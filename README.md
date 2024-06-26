# Llama meets EU: Investigating the European Political Spectrum through the Lens of LLMs

Instruction-finetuned Large Language Models inherit clear political leanings that have been shown to influence downstream task performance. In this work ([Chalkidis and Brandl, 2024](https://arxiv.org/abs/2403.13592)),  we expand this line of research beyond the two-party system in the US and audit *Llama Chat* ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)) on political debates from the European Parliament in various settings to analyze the model's political knowledge and its ability to reason in context. We adapt, i.e., further fine-tune, *Llama Chat*  on parliamentary debates of individual euro parties to reevaluate its political leaning based on the *EU-AND-I* questionnaire ([Michel et al., 2019](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3446677)). *Llama Chat* shows extensive prior knowledge of party positions and is capable of reasoning in context. The adapted, party-specific, models are substantially re-aligned towards respective positions which we see as a starting point for using chat-based LLMs as data-driven conversational engines to assist research in political science.

![Radar Plots](https://i.ibb.co/28Ptc37/Screenshot-2024-05-22-at-10-59-26.png)

## Datasets

As part of this work, we release the following datasets:

| Dataset | Dataset HF Alias |
| ---------- | ----------- |
| EU Debates | [`coastalcph/eu_debates`](https://huggingface.co/datasets/coastalcph/eu_debates)
| EUANDI 2019 | [`coastalcph/euandi_2019`](https://huggingface.co/datasets/coastalcph/euandi_2019)

## Models

As part of this work, we release the following LoRA adapters:

| Euro-party | Model HF Alias |
| ---------- | ----------- |
| EPP | [`coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-epp`](https://huggingface.co/coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-epp)
| S&D | [`coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-sd`](https://huggingface.co/coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-sd)
| ID  | [`coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-id`](https://huggingface.co/coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-id)
| GUE/NGL | [`coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-gue-ngl`](https://huggingface.co/coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-gue-ngl)
| Greens/EFA | [`coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-greens-efa`](https://huggingface.co/coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-greens-efa)

You can use the adapted models via the HuggingFace API:

```python

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-epp")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model = PeftModel.from_pretrained(base_model, "coastalcph/Llama-2-13b-chat-hf-LoRA-eu-debates-epp")
```

# Use Code

### Install dependencies

```shell
pip install -r requirements.txt
```

### Audit Llama Chat

```shell
python ./audit_llms/audit_llms.py
```

### Contextualize Auditing Settings A-C

```shel
python ./audit_llms/setting_a.py
python ./audit_llms/setting_b.py
python ./audit_llms/setting_c.py
```

### Fine-tune Llama Chat to EU Debates

```shell
python ./finetune_llms/finetune_llms.py
```


# Citation Information

*[Llama meets EU: Investigating the European political spectrum through the lens of LLMs. 
Ilias Chalkidis and Stephanie Brandl. 
In the Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 
Mexico City, Mexico, June 16–21, 2024.](https://arxiv.org/abs/2403.13592)*

```
@inproceedings{chalkidis-and-brandl-eu-llama-2024,
    title = "Llama meets EU: Investigating the European political spectrum through the lens of LLMs",
    author = "Chalkidis, Ilias  and Brandl, Stephanie",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
}

```
