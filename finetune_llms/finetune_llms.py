import os
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from peft import LoraConfig, get_peft_model, TaskType
from data import DATA_DIR
import logging, datasets
import sys
from audit_llms.helpers import clean_text_qa_instruct


logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || " \
           f"trainable%: {100 * trainable_params / all_param}"


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='meta-llama/Llama-2-13b-chat-hf', help='Model name in HF Hub')
    parser.add_argument('--dataset_name', default='coastalcph/eu_debates', help='Dataset name')
    parser.add_argument('--party_names', default=None, help='List of party names to consider when filtering')
    parser.add_argument('--speaker_role', default=None, help='List of speaker roles to consider when filtering')
    parser.add_argument('--years', default=None, help='Year to consider when filtering')
    parser.add_argument('--date_range', default=None, help='Date range to consider when filtering')
    parser.add_argument('--min_length', default=100, help='Minimum length of the text to consider when filtering')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--per_device_train_batch_size', default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--auth_token', default=None)
    parser.add_argument('--output_extension', default='ppe-all', help='Output extension for output directory')
    parser.add_argument('--pseudo_qa', default=None, help='Whether to turn the text into a pseudo question')
    param_config = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Fix parties' list
    param_config.party_names = param_config.party_names.split(',') if param_config.party_names is not None else None

    # Report configuration parameters
    print('Configuration parameters:')
    for arg in vars(param_config):
        print(f'{arg}: {getattr(param_config, arg)}')

    # Compute free memory for each GPU
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    # Load tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(
        param_config.model_name,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            use_flash_attention=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(param_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare LORA model
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # Cast the output to float32
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Set the LORA config
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Init PEFT model
    model = get_peft_model(model, config)

    # Report the number of trainable parameters
    print(print_trainable_parameters(model))

    # Load the dataset
    dataset = load_dataset(param_config.dataset_name, split="train")

    # Filter out the samples that are not from the party of interest
    if param_config.party_names is not None:
        dataset = dataset.filter(lambda sample: sample["speaker_party"] in param_config.party_names)
        print('Number of samples:', len(dataset), 'from the party of interest (', param_config.party_names, ')')

    # Filter out the samples that are not from the speaker role of interest
    if param_config.speaker_role is not None:
        dataset = dataset.filter(lambda sample: sample["speaker_role"] in param_config.speaker_role)
        print('Number of samples:', len(dataset), 'from the speaker role of interest (', param_config.speaker_role, ')')

    # Filter out the samples that are not from the year of interest
    if param_config.years is not None:
        dataset = dataset.filter(lambda sample: sample["year"] in param_config.years)
        print('Number of samples:', len(dataset), 'from the year of interest (', param_config.years, ')')

    # Filter out the samples that are not from the date range of interest
    if param_config.date_range is not None:
        dataset = dataset.filter(lambda sample: param_config.date_range[0] < sample["date"] < param_config.date_range[1])
        print('Number of samples:', len(dataset), 'from the date range of interest (', param_config.date_range, ')')

    # Filter out the samples that are too short
    if param_config.min_length is not None:
        dataset = dataset.filter(lambda sample: len(sample["text"].split(' ')) > param_config.min_length)
        print('Number of samples:', len(dataset), 'that are longer than', param_config.min_length, 'tokens')

    if param_config.pseudo_qa is not None:
        print('Turning the text into a pseudo question')
        # Turn text into a pseudo question
        dataset = dataset.map(clean_text_qa_instruct, load_from_cache_file=False)

    # Tokenize the dataset
    dataset = dataset.shuffle(seed=param_config.seed)
    dataset = dataset.map(lambda samples: tokenizer(samples["text"], padding="max_length",
                                                    truncation=True, max_length=512), batched=True)

    # Prepare the dataset for training
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=param_config.per_device_train_batch_size,
            gradient_accumulation_steps=param_config.gradient_accumulation_steps,
            per_device_eval_batch_size=param_config.per_device_train_batch_size,
            num_train_epochs=param_config.epochs,
            optim="paged_adamw_32bit",
            warmup_ratio=0.05,
            weight_decay=0.001,
            max_grad_norm=0.3,
            learning_rate=param_config.lr,
            lr_scheduler_type="constant",
            fp16=True,
            logging_strategy="steps",
            log_level="info",
            logging_first_step=True,
            save_total_limit=5,
            logging_steps=50,
            save_strategy="epoch",
            output_dir=os.path.join(DATA_DIR, 'adapted_models', f'{param_config.model_name}-{param_config.output_extension}'),
            seed=param_config.seed,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()