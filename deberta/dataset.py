"""
PyTorch Dataset for fine-tuning DeBERTa as a web element ranking model.

Each sample represents one (context, element) pair with a binary relevance label.
For InfoNCE training the dataset also exposes step_id so that the collator can
group all candidates belonging to the same step into one contrastive example.

Data format expected (Mind2Web processed JSONs):
  List of dicts, each with keys:
    confirmed_task, action_reprs, annotation_id, actions: [
      { action_uid, cleaned_html, pos_candidates: [{backend_node_id, ...}],
        neg_candidates: [{backend_node_id, ...}], operation: {op, value} }
    ]
"""

import json
import sys
from pathlib import Path
from typing import Optional

import lxml.etree as etree
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from dom_utils import get_tree_repr, clean_tree


def _element_text(tree: etree._Element, backend_node_id: str) -> str:
    """Return a compact text representation of a single element node."""
    nodes = tree.xpath(f'//*[@backend_node_id="{backend_node_id}"]')
    if not nodes:
        return ""
    node = nodes[0]
    # get_tree_repr expects a sub-tree; pass the node directly
    repr_str, _ = get_tree_repr(node, keep_html_brackets=True)
    return repr_str[:512]  # cap length


class ElementRankingDataset(Dataset):
    """
    Flat dataset: one row = one (context, element, label) triple.

    context  – "{intent} [SEP] {action_history}"
    element  – compact HTML repr of the candidate element
    label    – 1 (positive) or 0 (negative)
    step_id  – "{annotation_id}_{action_uid}" for grouping during InfoNCE
    """

    def __init__(
        self,
        data_paths: list[str | Path],
        tokenizer,
        max_context_len: int = 512,
        max_element_len: int = 256,
        max_total_len: int = 512,
        neg_per_step: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_total_len = max_total_len
        self.samples: list[dict] = []

        for path in data_paths:
            with open(path) as f:
                data = json.load(f)
            self._process(data, neg_per_step)

    def _process(self, data: list[dict], neg_per_step: Optional[int]):
        for dat in data:
            intent = dat["confirmed_task"]
            action_history_all = dat["action_reprs"]
            annotation_id = dat["annotation_id"]

            for index, d in enumerate(dat["actions"]):
                if not d["pos_candidates"]:
                    continue

                action_uid = d["action_uid"]
                step_id = f"{annotation_id}_{action_uid}"
                action_history = str(action_history_all[:index])
                context = f"{intent}\n{action_history}".strip()

                dom_tree = etree.fromstring(d["cleaned_html"])
                cleaned = clean_tree(dom_tree, set(
                    c["backend_node_id"] for c in d["pos_candidates"] + d["neg_candidates"]
                ))

                pos_id = d["pos_candidates"][0]["backend_node_id"]
                self.samples.append({
                    "context": context,
                    "element": _element_text(cleaned, pos_id),
                    "label": 1,
                    "step_id": step_id,
                })

                neg_candidates = d["neg_candidates"]
                if neg_per_step is not None:
                    neg_candidates = neg_candidates[:neg_per_step]

                for cand in neg_candidates:
                    self.samples.append({
                        "context": context,
                        "element": _element_text(cleaned, cand["backend_node_id"]),
                        "label": 0,
                        "step_id": step_id,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["context"],
            s["element"],
            max_length=self.max_total_len,
            truncation="only_first",
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids"),
            "label": s["label"],
            "step_id": s["step_id"],
        }


class InfoNCEBatch:
    """
    Collate function that groups flat samples by step_id into contrastive groups.

    Each group has exactly 1 positive (index 0) and up to max_neg negatives.
    Returns a dict ready for the InfoNCE loss:
      group_input_ids   : (B, G, L)  – B groups, G candidates (padded), L tokens
      group_attn_mask   : (B, G, L)
      group_token_types : (B, G, L)  – may be None
      group_labels      : (B, G)     – 1 or 0 per candidate
    """

    def __init__(self, tokenizer, max_neg: int = 10, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_neg = max_neg
        self.max_len = max_len

    def __call__(self, raw_batch: list[dict]) -> dict:
        import torch
        from collections import defaultdict

        groups: dict[str, list[dict]] = defaultdict(list)
        for item in raw_batch:
            groups[item["step_id"]].append(item)

        all_ids, all_mask, all_tt, all_labels = [], [], [], []
        for step_samples in groups.values():
            pos = [s for s in step_samples if s["label"] == 1]
            neg = [s for s in step_samples if s["label"] == 0][: self.max_neg]
            if not pos:
                continue
            group = pos[:1] + neg
            g_ids, g_mask, g_tt, g_lab = [], [], [], []
            for s in group:
                pad_len = self.max_len - len(s["input_ids"])
                g_ids.append(s["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
                g_mask.append(s["attention_mask"] + [0] * pad_len)
                if s["token_type_ids"] is not None:
                    g_tt.append(s["token_type_ids"] + [0] * pad_len)
                g_lab.append(s["label"])

            # pad group size
            pad_group = self.max_neg + 1 - len(group)
            for _ in range(pad_group):
                g_ids.append([self.tokenizer.pad_token_id] * self.max_len)
                g_mask.append([0] * self.max_len)
                if g_tt:
                    g_tt.append([0] * self.max_len)
                g_lab.append(-1)  # -1 = padding, ignored in loss

            all_ids.append(g_ids)
            all_mask.append(g_mask)
            if g_tt:
                all_tt.append(g_tt)
            all_labels.append(g_lab)

        result = {
            "group_input_ids": torch.tensor(all_ids, dtype=torch.long),
            "group_attn_mask": torch.tensor(all_mask, dtype=torch.long),
            "group_labels": torch.tensor(all_labels, dtype=torch.long),
        }
        if all_tt:
            result["group_token_types"] = torch.tensor(all_tt, dtype=torch.long)
        return result
