import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


import os
from data import DATA_DIR
from datasets import load_dataset
import json
from collections import Counter


def euandi_data(MODELS, PARTIES, HUMANS):
    q_dataset = load_dataset("csv",
                             data_files={"test": [os.path.join(DATA_DIR, "questionnaires/eu_elections_euandi.csv")]},
                             delimiter=",", column_names=["statement", "liberal society", "environment",
                                                          "eu integration", "economic liberal", "finance restrictions",
                                                          "immigration", "law and order", "right/left",
                                                          "pro-eu/anti-eu"], split='test')

    model_responses = {model_name: [] for model_name in MODELS}
    human_responses = {model_name: [] for model_name in MODELS}

    for model_name in model_responses:
        responses = []
        with open(os.path.join(DATA_DIR, f"model_responses/{model_name}_responses.jsonl"), "r") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                data = json.loads(line)
                statement_responses = [data[f'normalized_response_{idx_prompt}'] for idx_prompt in range(3) if
                                       data[f'normalized_response_{idx_prompt}'] != 'N/A']
                # Count signs
                response_signs = []
                for response in statement_responses:
                    if response == 0:
                        response_signs.append(0)
                    elif response > 0:
                        response_signs.append(1)
                    else:
                        response_signs.append(-1)
                counts = Counter(response_signs).most_common()[0]
                if counts[1] >= 2:
                    if counts[0] != 0:
                        ind_responses = [response for response in statement_responses if response * counts[0] > 0]
                        responses.append(sum(ind_responses) / len(ind_responses))
                    else:
                        responses.append(0)
                else:
                    responses.append(None)
        model_responses[model_name] = responses
        try:
            responses = {}
            for human in ['sb', 'ic']:
                responses[human] = []
                with open(os.path.join(DATA_DIR, f"model_responses/{model_name}_responses_annotated_{human}.jsonl"), "r") as f:
                    for idx, line in enumerate(f.readlines()):
                        data = json.loads(line)
                        statement_responses = [data[f'gold_label_{idx_prompt}'] for idx_prompt in range(3) if
                                               data[f'gold_label_{idx_prompt}'] != 'N/A']
                        # Count signs
                        response_signs = []
                        for response in statement_responses:
                            if response == 0:
                                response_signs.append(0)
                            elif response > 0:
                                response_signs.append(1)
                            else:
                                response_signs.append(-1)
                        counts = Counter(response_signs).most_common()[0]
                        if counts[1] >= 2:
                            if counts[0] != 0:
                                ind_responses = [response for response in statement_responses if response * counts[0] > 0]
                                responses[human].append(sum(ind_responses) / len(ind_responses))
                            else:
                                responses[human].append(0)
                        else:
                            responses[human].append(None)
            human_responses[model_name] = (np.nan_to_num(np.array(responses['sb'], dtype=np.float64)) + np.nan_to_num(
                np.array(responses['ilias'], dtype=np.float64))) / 2

        except FileNotFoundError:
            continue

    party_responses = {party: [] for party in PARTIES}
    with open(os.path.join(DATA_DIR, 'model_responses/aggregated_party_responses.csv'), 'r') as f:
        for line in f.readlines():
            responses = line.strip().split(',')
            if responses[0] in PARTIES:
                party_responses[responses[0]] = [float(response) if response != 'None' else None for response in
                                                 responses[1:]]

    dict_responses = {
        party: [0] * 7 for party in PARTIES + MODELS + [h + '_human' for h in HUMANS]
    }
    keys = ['liberal society', 'environment', 'eu integration', 'economic liberal',
            'finance restrictions', 'immigration', 'law and order']

    for idx_statement, row in enumerate(q_dataset):
        if idx_statement == 0:
            continue
        for party_idx, party in enumerate(party_responses):
            if party_responses[party][idx_statement - 1] is not None:
                for key_idx, key in enumerate(keys):
                    dict_responses[party][key_idx] += int(row[key]) * party_responses[party][idx_statement - 1]
            else:
                for key_idx, key in enumerate(keys):
                    dict_responses[party][key_idx] += 0

        for model_idx, model in enumerate(model_responses):
            if model_responses[model][idx_statement - 1] is not None:
                for key_idx, key in enumerate(keys):
                    dict_responses[model][key_idx] += int(row[key]) * model_responses[model][idx_statement - 1]
            else:
                for key_idx, key in enumerate(keys):
                    dict_responses[model][key_idx] += 0

        for model_idx, model in enumerate(human_responses):
            try:
                if human_responses[model][idx_statement - 1] is not None:
                    for key_idx, key in enumerate(keys):
                        dict_responses[model + '_human'][key_idx] += int(row[key]) * human_responses[model][idx_statement - 1]
                else:
                    for key_idx, key in enumerate(keys):
                        dict_responses[model + '_human'][key_idx] += 0
            except IndexError:
                if idx_statement==22:
                    continue

    data = [
        ['liberal society', 'environment', 'eu integration', 'economic liberal',
         'finance restrictions', 'immigration', 'law and order'],
        ('Survey', [dict_responses[party] for party in PARTIES + MODELS + [h + '_human' for h in HUMANS]]
         )
    ]
    return data


if __name__ == '__main__':
    PARTIES = ['PPE', 'S&D', 'ID', 'GUE/NGL', 'Greens']
    PARTIES_NAMES = {
        'PPE': "European People's Party",
        'S&D': "Progressive Alliance of Socialists and Democrats",
        'ID': "Identity and Democracy",
        'GUE/NGL': "The Left in the European Parliament",
        'Greens': "Greens–European Free Alliance"
    }

    MODELS = ['Llama-2-13b-chat-hf-ppe',
              'Llama-2-13b-chat-hf-ecr',
              'Llama-2-13b-chat-hf-id',
              'Llama-2-13b-chat-hf-gue',
              'Llama-2-13b-chat-hf-greens']
    HUMANS = MODELS
    MODELS_short = {
        'Llama-2-13b-chat-hf-ppe': 'hf-ppe',
        'Llama-2-13b-chat-hf-ecr': 'hf-ecr',
        'Llama-2-13b-chat-hf-id': 'hf-id',
        'Llama-2-13b-chat-hf-gue': 'hf-gue',
        'Llama-2-13b-chat-hf-greens': 'hf-greens'
    }
    COLORS = {'PPE': 'blue', 'S&D': 'red', 'ID': 'black',
              'GUE/NGL': 'darkred', 'Greens': 'green'}

    COLORS_HUMAN = {'PPE': 'lightblue', 'S&D': 'lightcoral', 'ID': 'dimgray',
              'GUE/NGL': 'lightcoral', 'Greens': 'lightgreen'}

    for (MODEL, PARTY, HUMAN) in zip(MODELS, PARTIES, HUMANS):
        COLORS_tmp = [COLORS[PARTY], 'gray', COLORS_HUMAN[PARTY]]
        N = 7
        theta = radar_factory(N, frame='polygon')

        data = euandi_data([MODEL], [PARTY], [HUMAN])
        spoke_labels = data.pop(0)

        fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                                subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        # Plot the four cases from the example data on separate axes
        for (title, case_data) in data:
            axs.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                          horizontalalignment='center', verticalalignment='center')
            for d, color in zip(case_data, COLORS_tmp):
                axs.plot(theta, d, color=color)
                axs.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
            axs.set_varlabels(spoke_labels)

        # add legend relative to top-left plot
        legend = axs.legend([PARTY, 'LLama_' + MODELS_short[MODEL], 'human_annotated'], loc=(0.7, .95),
                            labelspacing=0.1)

        axs.set_title(PARTIES_NAMES[PARTY])

        # plt.show()
        plt.savefig(f"radar_{PARTY.replace('/','-')}_avg.png", dpi=300)