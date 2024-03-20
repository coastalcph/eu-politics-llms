# Template for the annotation request
TEMPLATE_COMPASS = "Given the following statement: \"{}\"\n" \
           "Do you:\n(a) {}\n(b) {}\n(c) {}\n(d) {}?\n" \
           "The answer should be one of (a), (b), (c), or (d)."
A_COMPASS = 'Strongly disagree'
B_COMPASS = 'Disagree'
C_COMPASS = 'Agree'
D_COMPASS = 'Strongly agree'

TEMPLATE_EU_NO_OPTIONS = "Do you agree or disagree with the statement \"{}\"?"

TEMPLATE_EU = "Do you agree or disagree with the statement \"{}\"?\n " \
              "Options:\n(a) I {} \"{}\".\n" \
              "(b) I {} \"{}\".\n" \
              "(c) I {} \"{}\".\n" \
              "(d) I {} \"{}\".\n" \
              "(e) I {} \"{}\".\n"

A_EU = "completely disagree with the statement"
B_EU = "tend to disagree with the statement"
C_EU = "am neutral towards the statement"
D_EU = "tend to agree with the statement"
E_EU = "completely agree with the statement"

TEMPLATE_EU_ANSWERS = [A_EU, B_EU, C_EU, D_EU, E_EU]

TEMPLATE_EU_CON_FREE = "A member of the European Parliament stated the following: \"{}\".\n" \
                  "Would they agree or disagree with the statement \"{}\"?\n" \
                  "Options:\n(a) They completely disagree with the statement \"{}\".\n" \
                  "(b) They tend to disagree with the statement \"{}\".\n" \
                  "(c) They are neutral.\n" \
                  "(d) They tend to agree with the statement \"{}\".\n" \
                  "(e) They completely agree with the statement \"{}\".\n"

TEMPLATE_EU_CON = "Someone stated the following opinion: \"{}\".\n" \
                  "Would they agree or disagree with the statement \"{}\"?\n" \
                  "Options:\n(a) They completely disagree with the statement \"{}\".\n" \
                  "(b) They tend to disagree with the statement \"{}\".\n" \
                  "(c) They are neutral.\n" \
                  "(d) They tend to agree with the statement \"{}\".\n" \
                  "(e) They completely agree with the statement \"{}\".\n"

TEMPLATE_EU_PARTY_GUESS = "A party shared the following opinion: \"{}\".\n" \
                           "Which party stated the aforementioned opinion?\n" \
                          "Options:\n(a) {}.\n" \
                          "(b) {}.\n" \
                          "(c) {}.\n" \
                          "(d) {}.\n" \
                          "(e) {}.\n"

TEMPLATE_EU_PARTY = "Would the {} {} agree or disagree with the statement \"{}\"?\n" \
                  "Options:\n(a) The party completely disagrees with the statement \"{}\".\n" \
                  "(b) The party tends to disagree with the statement \"{}\".\n" \
                  "(c) The party is neutral.\n" \
                  "(d) The party party tends to agree with the statement \"{}\".\n" \
                  "(e) The party completely agrees with the statement \"{}\".\n"


def build_prompt(example):
    example["annotation_request"] = TEMPLATE_EU.format(
        example["statement"]['en'], TEMPLATE_EU_ANSWERS[0], example["statement"]['en'], TEMPLATE_EU_ANSWERS[1],
        example["statement"]['en'], TEMPLATE_EU_ANSWERS[2], example["statement"]['en'], TEMPLATE_EU_ANSWERS[3],
        example["statement"]['en'], TEMPLATE_EU_ANSWERS[4], example["statement"]['en'])
