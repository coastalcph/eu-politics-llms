from data import DATA_DIR
import os
import json
import numpy as np
from datasets import load_dataset
PARTY_NAMES = ['CDU', 'SPD', 'Grünen', 'Linke', 'AfD']
COUNTRY_NAMES = ['DE', 'DE', 'DE', 'DE', 'DE']
SETTING = 'setting_a'
scores = {party_name: {'soft_acc': 0, 'hard_acc': 0} for party_name in PARTY_NAMES}

euandi_questionnaire = load_dataset('coastalcph/euandi_2019', 'questionnaire', split='test')
dimensions = ['Liberal society', 'Environmental protection', 'EU integration', 'Economic liberalization',
              'Finance restrictions', 'Immigration restrictions', 'Law and Order']


for k in [(0,1), (1,2), (2,3), (0,3)]:
    print(f'Prompts {k}...')
    print('-' * 50)
    dimensional_acc_avg = {dimension: [] for dimension in dimensions}
    overall_soft_acc = []
    overall_hard_acc = []
    for party_name, country_code in zip(PARTY_NAMES, COUNTRY_NAMES):
        dimensional_acc = {dimension: 0 for dimension in dimensions}
        dimensional_total = {dimension: 0 for dimension in dimensions}
        soft_count = 0
        hard_count = 0
        total_answers = 0
        with open(os.path.join(DATA_DIR, "model_responses",
                               f"Llama-2-13b-chat-hf_{party_name.lower()}_{SETTING}_responses.jsonl")) as f:
            for idx, example in enumerate(f.readlines()):
                data = json.loads(example)
                normalized_model_responses = []
                for model_response in data['model_responses']:
                    if model_response.lower().startswith('(a)'):
                        normalized_model_responses.append(-1)
                    elif model_response.lower().startswith('(b)'):
                        normalized_model_responses.append(-0.5)
                    elif model_response.lower().startswith('(c)'):
                        normalized_model_responses.append(0)
                    elif model_response.startswith('(d)'):
                        normalized_model_responses.append(0.5)
                    elif model_response.lower().startswith('(e)'):
                        normalized_model_responses.append(1.0)
                if len(normalized_model_responses) == 0:
                    continue
                data['normalized_model_responses'] = normalized_model_responses[k[0]:k[1]]
                data['normalized_model_response'] = np.mean(data['normalized_model_responses'])

                # No party position or party is neutral
                if data['party_response']['position'] != '' and data['party_response']['answer'] == 0:
                    continue
                # No party position or model not able to give an answer
                if data['party_response']['position'] == '' or data['normalized_model_response'] == 'N/A':
                    continue
                else:
                    total_answers += 1
                    for dimension in dimensions:
                        if int(euandi_questionnaire[dimension][idx]) != 0:
                            dimensional_total[dimension] += 1
                if data['party_response']['answer'] == data['normalized_model_response']:
                    hard_count += 1
                    soft_count += 1
                    for dimension in dimensions:
                        if int(euandi_questionnaire[dimension][idx]) != 0:
                            dimensional_acc[dimension] += 1
                elif data['party_response']['answer'] * data['normalized_model_response'] > 0:
                    soft_count += 1
                    for dimension in dimensions:
                        if int(euandi_questionnaire[dimension][idx]) != 0:
                            dimensional_acc[dimension] += 1
                else:
                    pass

            scores[party_name]['soft_acc'] = soft_count / total_answers
            scores[party_name]['hard_acc'] = hard_count / total_answers
            dimensional_acc = [dimensional_acc[dimension]/dimensional_total[dimension] if dimensional_total[dimension] else 'N/A' for dimension in dimensions]
            overall_soft_acc.append(scores[party_name]['soft_acc'])
            overall_hard_acc.append(scores[party_name]['hard_acc'])
            dimensional_scores = ''
            for idx, dimension in enumerate(dimensions):
                if dimensional_acc[idx] == 'N/A':
                    dimensional_scores += f'& N/A '
                else:
                    dimensional_scores += f'& {dimensional_acc[idx]*100:2.1f} '
                    dimensional_acc_avg[dimension].append(dimensional_acc[idx])
            print(f'{party_name:>20} & {country_code} ' + dimensional_scores + f' & {scores[party_name]["soft_acc"]*100:2.1f} \\\\')

    overall_soft_acc = np.mean(overall_soft_acc)
    overall_hard_acc = np.mean(overall_hard_acc)
    dimensional_scores = ''
    for idx, dimension in enumerate(dimensions):
        dimensional_scores += f'& {np.mean(dimensional_acc_avg[dimension]) * 100:2.1f} '
    print(f'{"Averaged:":>20} & ' + dimensional_scores + f' & {overall_soft_acc*100:2.1f}')

