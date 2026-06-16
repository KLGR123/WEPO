"""
Build test datasets from raw Mind2Web data using DeBERTa-ranked candidate scores.

The DeBERTa ranking model (candidate generation stage) pre-scores all elements;
we take the top-k by score to build the pruned HTML snippet for each step.

Usage:
  python scripts/prepare_test.py \
      --data_dir   data/test_dataset \
      --scores_pkl data/scores_all_data.pkl \
      --output_dir data \
      --top_k      50 \
      --max_tokens 5000 \
      --tokenizer  meta-llama/Meta-Llama-3-8B-Instruct

The script generates three files:
  mind2web_dpo_test_domain_ranked.json
  mind2web_dpo_test_task_ranked.json
  mind2web_dpo_test_website_ranked.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import lxml.etree as etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dom_utils import prune_tree

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

# test split name → number of shards
SPLIT_SHARDS = {
    "test_domain": 10,
    "test_task": 3,
    "test_website": 2,
}


def build_split(
    data_dir: Path,
    split_name: str,
    n_shards: int,
    candidate_ranks: dict,
    top_k: int,
    max_tokens: int,
    tokenizer=None,
) -> list[dict]:
    dataset: list[dict] = []
    skipped_no_pos = 0
    skipped_too_long = 0

    for shard_id in range(n_shards):
        shard_path = data_dir / split_name / f"{split_name}_{shard_id}.json"
        with open(shard_path) as f:
            data = json.load(f)

        for dat in tqdm(data, desc=f"{split_name}[{shard_id}]", leave=False):
            intent = dat["confirmed_task"]
            action_history_all = dat["action_reprs"]
            annotation_id = dat["annotation_id"]

            for index, d in enumerate(dat["actions"]):
                if not d["pos_candidates"]:
                    skipped_no_pos += 1
                    continue

                action_uid = d["action_uid"]
                sample_id = f"{annotation_id}_{action_uid}"

                # use DeBERTa ranks to select top-k candidates
                step_ranks = candidate_ranks.get(sample_id, {})
                candidate_ids = [
                    c["backend_node_id"]
                    for candidates in [d["pos_candidates"], d["neg_candidates"]]
                    for c in candidates
                    if step_ranks.get(c["backend_node_id"], float("inf")) <= top_k
                ]

                if not candidate_ids:
                    # fall back: always include the ground truth
                    candidate_ids = [d["pos_candidates"][0]["backend_node_id"]]

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
                gt_id = d["pos_candidates"][0]["backend_node_id"]

                dataset.append({
                    "instruction": SYSTEM_PROMPT,
                    "input": USER_TEMPLATE.format(intent=intent, html=html, action_history=action_history),
                    "output": OUTPUT_TEMPLATE.format(op=op, value=value, id=gt_id),
                    # metadata for offline analysis
                    "meta": {
                        "sample_id": sample_id,
                        "annotation_id": annotation_id,
                        "action_uid": action_uid,
                        "gt_backend_node_id": gt_id,
                    },
                })

    print(f"{split_name}: {len(dataset)} samples "
          f"(skipped no-pos={skipped_no_pos}, too-long={skipped_too_long})")
    return dataset


def parse_args():
    p = argparse.ArgumentParser(description="Build ranked test datasets from Mind2Web")
    p.add_argument("--data_dir", required=True,
                   help="Root dir containing test_domain/, test_task/, test_website/")
    p.add_argument("--scores_pkl", required=True,
                   help="Pickle file with DeBERTa scores: {'ranks': {sample_id: {node_id: rank}}}")
    p.add_argument("--output_dir", required=True, help="Directory to write output JSON files")
    p.add_argument("--splits", nargs="+", default=list(SPLIT_SHARDS.keys()),
                   choices=list(SPLIT_SHARDS.keys()), help="Which splits to process")
    p.add_argument("--top_k", type=int, default=50, help="Top-k candidates by DeBERTa rank (default: 50)")
    p.add_argument("--max_tokens", type=int, default=5000, help="Max token count per sample (default: 5000)")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer for token counting")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.scores_pkl, "rb") as f:
        scores_data = pickle.load(f)
    candidate_ranks = scores_data["ranks"]

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        dataset = build_split(
            data_dir=data_dir,
            split_name=split,
            n_shards=SPLIT_SHARDS[split],
            candidate_ranks=candidate_ranks,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            tokenizer=tokenizer,
        )
        out_path = out_dir / f"mind2web_dpo_{split}_ranked_{args.top_k}.json"
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
