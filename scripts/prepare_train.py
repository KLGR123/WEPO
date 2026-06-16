"""
Build DPO training dataset from raw Mind2Web data.

Supports two negative sampling strategies:
  --sampling random     : shuffle neg_candidates, take first n
  --sampling distance   : take n negatives with smallest DOM-LCA distance to the positive

Usage:
  python scripts/prepare_train.py \
      --data_dir  data/train_dataset \
      --output    data/mind2web_dpo_train.json \
      --sampling  distance \
      --n_neg     3 \
      --max_tokens 7900
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import lxml.etree as etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dom_utils import prune_tree

# ── prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a web navigation intelligence who interacts with webpage environments "
    "to achieve human user intent.\n"
    "You always generate the next ACTION based on the user's INTENT, current cleaned "
    "webpage HTML and ACTION_HISTORY sequence which records the actions that have been performed.\n\n"
    "Given HTML, INTENT and ACTION_HISTORY, you should\n"
    "(1) Rely on your HTML code comprehension to analyze and understand what elements "
    "are on the current page.\n"
    "(2) Depend on your reasoning skills to parse the user's INTENT and infer the next "
    "action that should be taken in conjunction with the historical trajectory ACTION_HISTORY.\n"
    "(3) Select an element carefully from HTML code to interact with, thus bringing the "
    "goal closer to completion.\n\n"
    "Your output format should be strictly as follows\n"
    "Operation: ... (should be CLICK or TYPE or SELECT)\n"
    "Value: ... (optional textual value for the operation TYPE or SELECT)\n"
    "ID: ... (unique id number for the element to click or type into)\n\n"
    "Now, begin!"
)

USER_TEMPLATE = "INTENT:\n{intent}\n\nHTML:\n{html}\n\nACTION_HISTORY:\n{action_history}"
OUTPUT_TEMPLATE = "Operation: {op}\nValue: {value}\nID: {id}"


# ── DOM distance helpers ───────────────────────────────────────────────────────

def _ancestor_path(tree: etree._Element, backend_node_id: str) -> list[str]:
    """Return [node_id, parent_id, ..., root_id] for the given backend_node_id."""
    nodes = tree.xpath(f'//*[@backend_node_id="{backend_node_id}"]')
    if not nodes:
        return []
    path = []
    node = nodes[0]
    while node is not None:
        path.append(node.attrib.get("backend_node_id", ""))
        node = node.getparent()
    return path


def lca_distance(tree: etree._Element, id_a: str, id_b: str) -> int:
    """Sum of steps from node_a and node_b to their lowest common ancestor."""
    path_a = _ancestor_path(tree, id_a)
    path_b = _ancestor_path(tree, id_b)
    if not path_a or not path_b:
        # node missing from tree: treat as maximally distant so it sorts last
        return 10**6
    set_a = {nid: depth for depth, nid in enumerate(path_a)}
    for depth_b, nid in enumerate(path_b):
        if nid in set_a:
            return set_a[nid] + depth_b
    return len(path_a) + len(path_b)


def sample_by_distance(tree: etree._Element, pos_id: str, neg_ids: list[str], n: int) -> list[str]:
    """Return n negative ids closest in DOM distance to pos_id."""
    scored = [(nid, lca_distance(tree, pos_id, nid)) for nid in neg_ids]
    scored.sort(key=lambda x: x[1])
    return [nid for nid, _ in scored[:n]]


def sample_random(neg_ids: list[str], n: int) -> list[str]:
    pool = neg_ids.copy()
    random.shuffle(pool)
    return pool[:n]


# ── operation heuristic ────────────────────────────────────────────────────────

def assign_neg_op(pos_op: str, neg_value: str) -> tuple[str, str]:
    """
    With prob 0.33, replace TYPE/SELECT negative with CLICK (value cleared).
    This balances operation-type distribution without confusing the model
    about element functionality.
    """
    if pos_op != "CLICK" and random.random() < 0.33:
        return "CLICK", ""
    return pos_op, neg_value


# ── per-shard processing ───────────────────────────────────────────────────────

def process_shard(
    data: list[dict],
    sampling: str,
    n_neg: int,
    max_candidate_pool: int,
    max_tokens: Optional[int],
    tokenizer=None,
) -> tuple[list[dict], int]:
    dataset: list[dict] = []
    skipped_no_pos = 0
    skipped_too_long = 0

    for dat in tqdm(data, leave=False):
        intent = dat["confirmed_task"]
        action_history_all = dat["action_reprs"]

        for index, d in enumerate(dat["actions"]):
            if not d["pos_candidates"]:
                skipped_no_pos += 1
                continue

            gt_id = d["pos_candidates"][0]["backend_node_id"]
            neg_ids = [c["backend_node_id"] for c in d["neg_candidates"]]
            candidate_ids = [gt_id] + neg_ids[:max_candidate_pool]

            # prune HTML to candidate sub-trees
            dom_tree = etree.fromstring(d["cleaned_html"])
            pruned = prune_tree(dom_tree, candidate_ids)
            html = etree.tostring(pruned, pretty_print=True, method="html", encoding="unicode")
            html = html.replace("backend_node_id", "id")

            action_history = str(action_history_all[:index])

            if max_tokens and tokenizer is not None:
                n_tokens = (
                    len(tokenizer(html, add_special_tokens=False)["input_ids"])
                    + len(tokenizer(intent, add_special_tokens=False)["input_ids"])
                    + len(tokenizer(action_history, add_special_tokens=False)["input_ids"])
                )
                if n_tokens > max_tokens:
                    skipped_too_long += 1
                    continue

            op = d["operation"]["op"]
            value = d["operation"]["value"]
            chosen = OUTPUT_TEMPLATE.format(op=op, value=value, id=gt_id)
            input_text = USER_TEMPLATE.format(intent=intent, html=html, action_history=action_history)

            # negative sampling
            if sampling == "distance":
                raw_tree = etree.fromstring(d["cleaned_html"])
                sampled = sample_by_distance(raw_tree, gt_id, neg_ids, n_neg)
            else:
                sampled = sample_random(neg_ids, n_neg)

            for neg_id in sampled:
                neg_op, neg_value = assign_neg_op(op, value)
                rejected = OUTPUT_TEMPLATE.format(op=neg_op, value=neg_value, id=neg_id)
                dataset.append({
                    "instruction": SYSTEM_PROMPT,
                    "input": input_text,
                    "output": [chosen, rejected],
                })

    print(f"  skipped (no pos): {skipped_no_pos}, skipped (too long): {skipped_too_long}")
    return dataset, skipped_no_pos


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build DPO training dataset from Mind2Web")
    p.add_argument("--data_dir", required=True, help="Directory containing train_0.json … train_N.json")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--sampling", choices=["random", "distance"], default="distance",
                   help="Negative sampling strategy (default: distance)")
    p.add_argument("--n_neg", type=int, default=3, help="Negatives per positive (default: 3)")
    p.add_argument("--max_candidate_pool", type=int, default=20,
                   help="Max neg candidates before sampling (default: 20)")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Drop samples whose HTML+intent+history exceeds this token count")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer name/path for token counting (required if --max_tokens set)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    tokenizer = None
    if args.max_tokens:
        if not args.tokenizer:
            raise ValueError("--tokenizer is required when --max_tokens is set")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    data_dir = Path(args.data_dir)
    shard_files = sorted(data_dir.glob("train_*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No train_*.json files found in {data_dir}")

    all_samples: list[dict] = []
    for shard_path in shard_files:
        print(f"Processing {shard_path.name} …")
        with open(shard_path) as f:
            data = json.load(f)
        samples, _ = process_shard(data, args.sampling, args.n_neg,
                                   args.max_candidate_pool, args.max_tokens, tokenizer)
        all_samples.extend(samples)
        print(f"  → {len(samples)} samples  (total so far: {len(all_samples)})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_samples)} training samples to {out_path}")


if __name__ == "__main__":
    main()
