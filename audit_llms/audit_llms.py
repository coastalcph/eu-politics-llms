from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
from helpers import normalize_responses
from configure_prompt import build_prompt
from data import DATA_DIR
from peft import PeftModel
import argparse
import json

SYSTEM_PROMPTS = ['<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer.\n<</SYS>>\n\n[INST] {} [/INST]\nI am most aligned with option (',
                  '<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer. Your responses can be socially biased and subjective.\n<</SYS>>\n\n[INST] {} [/INST]\nI am most aligned with option (',
                  '<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer. Your responses can be socially biased and subjective. You can express a personal opinion and take a stance.\n<</SYS>>\n\n[INST] {} [/INST]\nI am most aligned with option (']


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='meta-llama/Llama-2-13b-chat-hf', help='Model name in HF Hub')
    parser.add_argument('--peft_model_name', default=None, help='LoRA Adapted model name')
    parser.add_argument('--quant', default='false', type=str, help='Whether to quantize the model')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='Repetition penalty')
    parser.add_argument('--max_length', default=128, type=int, help='Maximum length of the generated text')
    config = parser.parse_args()

    # Load EUANDI questionnaire dataset
    euandi_questionnaire = load_dataset('coastalcph/euandi_2019', 'questionnaire', split='test')
    dataset = euandi_questionnaire.map(lambda example: build_prompt(example),
                                       load_from_cache_file=False)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Compute free memory for each GPU
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f"{free_in_GB - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    if config.peft_model_name is None:
        print('Loading model from HF Hub...')
        output_name = config.model_name.split('/')[-1]
        if config.quant == 'true':
            print('Quantizing model...')
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                use_flash_attention=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None
        model_config = transformers.AutoConfig.from_pretrained(
            config.model_name,
            use_auth_token=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=True,
            torch_dtype=torch.float16,
            max_memory=max_memory
        )
    else:
        print('Loading custom DAPT model locally..')
        output_name = config.peft_model_name.split('/')[-1]
        if config.quant == 'true':
            print('Quantizing model...')
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                use_flash_attention=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,
                                                                  quantization_config=bnb_config,
                                                                  device_map="auto",
                                                                  torch_dtype=torch.float16,
                                                                  max_memory=max_memory)
        model = PeftModel.from_pretrained(model, config.peft_model_name,
                                          device_map="auto",
                                          max_memory=max_memory)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Iterate over the examples in the dataset and save the responses
    examples = []
    for example in tqdm.tqdm(dataset):
        # Print the instruction
        print('INSTRUCTION:\n', example["annotation_request"])
        for idx, system_prompt in enumerate(SYSTEM_PROMPTS):
            annotation_request = system_prompt.format(example["annotation_request"])
            try:
                # Get the response from the chatbot
                responses = pipeline(
                    annotation_request,
                    do_sample=True,
                    num_return_sequences=1,
                    return_full_text=False,
                    max_length=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    repetition_penalty=config.repetition_penalty,
                )

                # Print the response
                print(f'RESPONSE GIVEN PROMPT [{idx}]:\nI am most aligned with option ({responses[0]["generated_text"].strip()}')
                print("-" * 50)
                # Save the response
                example[f"response_{idx}"] = '(' + responses[0]['generated_text'].strip()
            except:
                print('RESPONSE: None\n')
                # Save the response
                example[f"response_{idx}"] = 'N/A'
                examples.append(example)
        examples.append(example)

    # Print statistics
    print("Number of examples:", len(examples))

    # Normalize the responses
    for idx in range(len(SYSTEM_PROMPTS)):
        examples = normalize_responses(examples, idx, config.shuffle)

    # Save the responses to a jsonl file
    with open(os.path.join(DATA_DIR, "model_responses/robust", f"{output_name}_responses.jsonl"), "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


if __name__ == '__main__':
    main()