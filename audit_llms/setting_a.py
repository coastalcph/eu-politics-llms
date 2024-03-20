from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
from configure_prompt import TEMPLATE_EU_PARTY
from data import DATA_DIR
from peft import PeftModel
import argparse
import json

SYSTEM_PROMPTS = ['<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer.\n<</SYS>>\n\n[INST] {} [/INST]\nThe answer is option (',
                  '<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer. Your responses can be socially biased and subjective.\n<</SYS>>\n\n[INST] {} [/INST]\nThe answer is option (',
                  '<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer. Your responses can be socially biased and subjective. You can express a personal opinion and take a stance.\n<</SYS>>\n\n[INST] {} [/INST]\nThe answer is option (']


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='meta-llama/Llama-2-13b-chat-hf', help='Model name in HF Hub')
    parser.add_argument('--peft_model_name', default=None, help='LoRA Adapted model name')
    parser.add_argument('--quant', default='false', type=str, help='Whether to quantize the model')
    parser.add_argument('--countries', default=['German', 'German', 'German', 'German', 'German', 'Greek', 'Greek', 'Greek', 'Greek'], help='Countries to compare')
    parser.add_argument('--party_names', default=['CDU', 'SPD', 'Grünen', 'Linke', 'AfD'], help='Party names')
    parser.add_argument('--full_party_names', default=['party CDU/CSU', 'party SPD', 'party Die Grünen', 'party Die Linke ', 'party AfD'], help='Party names to compare')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='Repetition penalty')
    parser.add_argument('--max_length', default=64, type=int, help='Maximum length of the generated text')
    config = parser.parse_args()

    # Load EUANDI questionnaire dataset
    euandi_questionnaire = load_dataset('coastalcph/euandi_2019', 'questionnaire', split='test')
    euandi_statements = [statement['en'] for statement in euandi_questionnaire['statement']]

    # Load EUANDI party responses
    euandi_party_positions = load_dataset('coastalcph/euandi_2019', 'party_positions', split='test')
    party_positions = {party_name: [] for party_name in config.party_names}
    for party_data in euandi_party_positions:
        if party_data['party_name'] in config.party_names:
            party_positions[party_data['party_name']] = {f'statement_{idx}': party_data[f'statement_{idx}'] for idx in range(1, 23)}

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
    for country, party_name, full_party_name in zip(config.countries, config.party_names, config.full_party_names):
        examples = []
        for statement_idx, example in tqdm.tqdm(enumerate(euandi_statements)):
            if party_positions[party_name][f'statement_{statement_idx+1}']['translated_position_google'] is not None:
                # Print the instruction
                question = TEMPLATE_EU_PARTY.format(country, full_party_name, example, example, example, example, example)

                model_responses = []
                for prompt_idx, system_prompt in enumerate(SYSTEM_PROMPTS):
                    annotation_request = system_prompt.format(question)
                    print(f'INSTRUCTION:\n{question}')
                    try:
                        # Get the response from the chatbot
                        responses = pipeline(
                            annotation_request,
                            do_sample=True,
                            num_return_sequences=1,
                            return_full_text=False,
                            max_new_tokens=config.max_length,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            repetition_penalty=config.repetition_penalty,
                        )

                        # Print the response
                        print(f'RESPONSE: The answer is option ({responses[0]["generated_text"].strip()}')
                        print("-" * 50)
                        # Save the response
                        model_responses.append('(' + responses[0]['generated_text'].strip())
                    except:
                        print('RESPONSE: None\n')
                        # Save the response
                        model_responses.append('N/A')
            else:
                # Save the response
                model_responses = ['No explanation provided.'] * 3
            examples.append({'party_response': party_positions[party_name][f'statement_{statement_idx + 1}'],
                             'model_responses': model_responses})

        # Normalize the responses
        for statement in examples:
            normalized_model_responses = []
            for response in statement['model_responses']:
                if response.lower().startswith('(a)'):
                    normalized_model_responses.append(-1)
                elif response.lower().startswith('(b)'):
                    normalized_model_responses.append(-0.5)
                elif response.lower().startswith('(c)'):
                    normalized_model_responses.append(0)
                elif response.startswith('(d)'):
                    normalized_model_responses.append(0.5)
                elif response.lower().startswith('(e)'):
                    normalized_model_responses.append(1.0)
                else:
                    normalized_model_responses.append('N/A')
            statement['normalized_model_responses'] = normalized_model_responses

        # Save the responses to a jsonl file
        with open(os.path.join(DATA_DIR, "model_responses", f"{output_name}_{party_name.lower()}_setting_a_responses.jsonl"), "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")


if __name__ == '__main__':
    main()