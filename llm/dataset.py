"""
Dataset utilities for LLM fine-tuning (nDPO / SFT).

The DPO dataset produced here yields groups of the form:
  { "prompt": ..., "chosen": ..., "rejected": [r1, r2, ..., rn] }

where the "rejected" field is a list so that nDPO (Plackett-Luce) can
treat all n negatives jointly in a single training step.

If you need vanilla pairwise DPO compatible with TRL's DPOTrainer, call
  dataset.to_pairwise()
which explodes each group into n (prompt, chosen, rejected) pairs.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── tokenisation helpers ───────────────────────────────────────────────────────

def _apply_chat(tokenizer, system: str, user: str, assistant: str | None = None) -> str:
    """
    Build chat-template input. Returns raw string for nDPO (not tensors),
    because the trainer handles padding/truncation globally.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    # some tokenizers don't have chat templates; fall back to plain concat
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(assistant is None)
        )
    sep = tokenizer.eos_token or "\n"
    prompt = f"{system}{sep}{user}{sep}"
    if assistant is not None:
        return prompt + assistant
    return prompt


# ── nDPO dataset ───────────────────────────────────────────────────────────────

class WEPODataset(Dataset):
    """
    Loads a pre-built WEPO JSON file (output of scripts/prepare_train.py) and
    exposes it as a dataset whose items are nDPO groups:

      item["prompt"]    – str: system + user turn (no assistant token yet)
      item["chosen"]    – str: positive completion (assistant turn only)
      item["rejected"]  – list[str]: n negative completions

    Pass tokenizer=None to get raw strings; pass a tokenizer to get
    pre-tokenized dicts (useful for SFT or custom trainers).
    """

    def __init__(self, json_path: str | Path, tokenizer=None, max_prompt_len: int = 7936,
                 max_completion_len: int = 256):
        with open(json_path) as f:
            raw = json.load(f)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_completion_len = max_completion_len
        self.data = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        instruction = item["instruction"]
        user_input = item["input"]
        chosen_str, *rejected_list = (
            item["output"] if isinstance(item["output"], list) else [item["output"]]
        )

        if self.tokenizer is not None:
            prompt_str = _apply_chat(self.tokenizer, instruction, user_input)
        else:
            prompt_str = f"{instruction}\n\n{user_input}"

        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_list,
        }

    def to_pairwise(self) -> list[dict]:
        """Explode into classic (prompt, chosen, rejected) pairs for TRL DPOTrainer."""
        pairs = []
        for item in self:
            for rej in item["rejected"]:
                pairs.append({
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": rej,
                })
        return pairs


class SFTDataset(Dataset):
    """
    Supervised fine-tuning dataset: single (prompt, completion) pairs,
    where completion is the chosen (positive) action.
    Used for initialising / ablation MindAct-style baselines.
    """

    def __init__(self, json_path: str | Path, tokenizer, max_len: int = 8192):
        with open(json_path) as f:
            raw = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.samples = []
        for item in raw:
            chosen = item["output"][0] if isinstance(item["output"], list) else item["output"]
            self.samples.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": chosen,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        full = _apply_chat(self.tokenizer, s["instruction"], s["input"], s["output"])
        enc = self.tokenizer(
            full,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
