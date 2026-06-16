# WEPO: Web Element Preference Optimization for LLM-based Web Navigation

Official repository of **WEPO** (AAAI 2025), a preference-optimization framework for
LLM-based web navigation agents. WEPO leverages **distance-based negative sampling**
over the DOM tree to build contrastive preference pairs/groups from Mind2Web data,
then fine-tunes an LLM with **DPO / nDPO** so it learns to pick the correct web
element given a user intent, current HTML, and action history.

> **[Hugging Face Model: WEPO-Llama-3-8b](https://huggingface.co/KLGR123/WEPO-llama-3-8b)**
> **[Arxiv Link](https://arxiv.org/pdf/2412.10742)**

---

## Installation

```bash
git clone https://github.com/KLGR123/WEPO.git
cd WEPO
pip install -r requirements.txt
```

---

## Data Preparation

```bash
# Training set (positive + sampled negatives)
python scripts/prepare_train.py \
    --data_dir   data/train_dataset \
    --output     data/mind2web_dpo_train.json \
    --sampling   distance \
    --n_neg      3 \
    --max_tokens 7900 \
    --tokenizer  meta-llama/Meta-Llama-3-8B-Instruct

# Test sets (DeBERTa-ranked candidate pruning)
python scripts/prepare_test.py \
    --data_dir   data/test_dataset \
    --scores_pkl data/scores_all_data.pkl \
    --output_dir data \
    --top_k      50 \
    --max_tokens 5000 \
    --tokenizer  meta-llama/Meta-Llama-3-8B-Instruct
```

---

## DeBERTa Candidate Ranker

```bash
# BCE
python deberta/train.py \
    --data_dir   data/train_dataset \
    --output_dir checkpoints/deberta_bce \
    --loss       bce \
    --model      microsoft/deberta-v3-base \
    --epochs 3 --batch_size 32 --lr 2e-5 --fp16

# InfoNCE
python deberta/train.py \
    --data_dir   data/train_dataset \
    --output_dir checkpoints/deberta_infonce \
    --loss        infonce \
    --max_neg     10 \
    --temperature 0.07 \
    --model       microsoft/deberta-v3-base \
    --epochs 3 --batch_size 8 --lr 2e-5 --fp16
```

---

## LLM Fine-tuning with (n)DPO

```bash
# nDPO (paper setting): Llama-3-8B, distance-based negatives, n=3
python llm/train_dpo.py \
    --model      meta-llama/Meta-Llama-3-8B-Instruct \
    --data       data/mind2web_dpo_train.json \
    --output     checkpoints/llama3_wepo \
    --loss_type  ndpo \
    --beta       0.95 \
    --n_neg      3 \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --batch_size 2 --grad_accum 8 --lr 1e-4 --epochs 3 \
    --bf16

# Pairwise DPO ablation
python llm/train_dpo.py \
    --model      meta-llama/Meta-Llama-3-8B-Instruct \
    --data       data/mind2web_dpo_train.json \
    --output     checkpoints/llama3_wepo_pairwise \
    --loss_type  pairwise \
    --beta       0.95 \
    --n_neg      3 \
    --bf16
```

---

## Evaluation

```bash
python llm/evaluate.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --model      checkpoints/llama3_wepo/best \
    --data       data/mind2web_dpo_test_domain_ranked_50.json \
                 data/mind2web_dpo_test_task_ranked_50.json \
                 data/mind2web_dpo_test_website_ranked_50.json \
    --splits     test_domain test_task test_website \
    --output     results/llama3_wepo.json \
    --bf16
```

Reports SSR (Step Success Rate), Op F1, and Element Distance per split and overall.
Pass `--save_predictions` to also dump raw generations for manual inspection.

---

## Quick Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("KLGR123/WEPO-llama-3-8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("KLGR123/WEPO-llama-3-8b", trust_remote_code=True).to("cuda:0")

messages = [
    {"role": "system", "content": "You are a web navigation intelligence who interacts with webpage environments to achieve human user intent."},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = model.generate(
    input_ids, max_new_tokens=128, eos_token_id=terminators,
    do_sample=True, temperature=0.2, top_p=0.9,
)
print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

If you fine-tuned your own LoRA adapter, load the base model and merge instead:

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "checkpoints/llama3_wepo/best").merge_and_unload()
```

---

## Cite Us

```bibtex
@inproceedings{liu2025wepo,
  title     = {WEPO: Web Element Preference Optimization for LLM-based Web Navigation},
  author    = {Liu, Jiarun and Hao, Jie and Zhang, Chi and Hu, Zitong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {25},
  pages     = {26614--26622},
  year      = {2025},
  doi       = {10.1609/aaai.v39i25.34863},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/34863}
}
```

For any inquiries, please reach out to us at [liujiarun01@bupt.edu.cn].

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
