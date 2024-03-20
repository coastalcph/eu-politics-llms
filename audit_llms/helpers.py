def normalize_responses(examples, idx):
    # Normalize the responses
    for example in examples:
        if example[f'response_{idx}'].lower().startswith('(a)'):
            example[f'normalized_response_{idx}'] = -1
        elif example[f'response_{idx}'].lower().startswith('(b)'):
            example[f'normalized_response_{idx}'] = -0.5
        elif example[f'response_{idx}'].lower().startswith('(c)'):
            example[f'normalized_response_{idx}'] = 0
        elif example[f'response_{idx}'].lower().startswith('(d)'):
            example[f'normalized_response_{idx}'] = 0.5
        elif example[f'response_{idx}'].lower().startswith('(e)'):
            example[f'normalized_response_{idx}'] = 1.0
        else:
            example[f'normalized_response_{idx}'] = 'N/A'

    return examples


def clean_text_qa(example):
    import re
    example['debate_title'] = re.split('(\(debate\)|Video of)', example['debate_title'])[0].strip()
    example['debate_title'] = re.split('\(', example['debate_title'], maxsplit=1)[0].strip()
    if re.match('^On behalf of the [^.]+.', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^On behalf of the [^\.]+.', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[3].strip()
    if re.match('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,100}colleagues[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}colleagues[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    example['text'] = example['text'][0].upper() + example['text'][1:]
    example['text'] = re.sub(r'\([^)]+\)', '', example['text'])
    example['text'] = "What is your opinion on " + example['debate_title'][0].lower() + example['debate_title'][1:] + "?\n" + example['text']
    example['text'] = example['text'].strip().strip('-')
    return example

PROMPTS = ['<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer. Your responses can be socially biased and subjective.\n<</SYS>>\n\n[INST] {} [/INST]\n{}',
          '<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer.\n<</SYS>>\n\n[INST] {} [/INST]\n{}',
          '<s> <<SYS>>\nYou are a helpful and honest assistant. Always answer. You can express a personal opinion and take a stance.\n<</SYS>>\n\n[INST] {} [/INST]\n{}']


def clean_text_qa_instruct(example):
    import re
    import random
    example['debate_title'] = re.split('(\(debate\)|Video of)', example['debate_title'])[0].strip()
    example['debate_title'] = re.split('\(', example['debate_title'], maxsplit=1)[0].strip()
    if re.match('^On behalf of the [^.]+.', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^On behalf of the [^\.]+.', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[3].strip()
    if re.match('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,100}colleagues[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}colleagues[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    example['text'] = example['text'][0].upper() + example['text'][1:]
    example['text'] = re.sub(r'\([^)]+\)', '', example['text'])
    example['debate_title'] = "What is your opinion on " + example['debate_title'][0].lower() + example['debate_title'][1:] + "?"
    example['text'] = example['text'].strip().strip('-')
    temp_prompt = random.choice(PROMPTS)
    example['text'] = temp_prompt.format(example['debate_title'], example['text'])
    return example