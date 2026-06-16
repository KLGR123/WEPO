"""
Evaluate a fine-tuned LLM on Mind2Web test sets.

Metrics reported (per split and overall):
  SSR          – Step Success Rate: element ID correct AND operation correct
  Op F1        – Operation-level F1 for TYPE/SELECT values
  Element Dist – Normalised DOM distance between predicted and GT element ID
                 (lower = closer in page structure)

Usage:
  python llm/evaluate.py \
      --model   checkpoints/llama3_wepo/best \
      --data    data/mind2web_dpo_test_website_ranked_50.json \
      --output  results/test_website.json \
      --split   test_website

  # Evaluate all three splits at once
  python llm/evaluate.py \
      --model  checkpoints/llama3_wepo/best \
      --data   data/mind2web_dpo_test_domain_ranked_50.json \
               data/mind2web_dpo_test_task_ranked_50.json \
               data/mind2web_dpo_test_website_ranked_50.json \
      --splits test_domain test_task test_website \
      --output results/all_splits.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── parsing ────────────────────────────────────────────────────────────────────

_ID_RE = re.compile(r"ID:\s*(\d+)", re.IGNORECASE)
_OP_RE = re.compile(r"Operation:\s*(\w+)", re.IGNORECASE)
_VAL_RE = re.compile(r"Value:\s*(.*?)(?:\n|$)", re.IGNORECASE)


def parse_output(text: str) -> dict:
    id_m = _ID_RE.search(text)
    op_m = _OP_RE.search(text)
    val_m = _VAL_RE.search(text)
    return {
        "id": int(id_m.group(1)) if id_m else None,
        "op": op_m.group(1).upper() if op_m else None,
        "value": val_m.group(1).strip() if val_m else "",
    }


# ── metrics ────────────────────────────────────────────────────────────────────

def char_f1(pred: str, gold: str) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    s1, s2 = set(pred.lower()), set(gold.lower())
    inter = len(s1 & s2)
    if not inter:
        return 0.0
    p = inter / len(s1)
    r = inter / len(s2)
    return 2 * p * r / (p + r)


def compute_op_f1(pred: dict, gold: dict) -> float | None:
    """
    Return character-level F1 for the value field when both pred and gold
    are TYPE or SELECT operations. Returns None for CLICK steps (not included
    in Op F1 averaging, consistent with Mind2Web evaluation).
    """
    g_op = gold["op"]
    p_op = pred["op"]
    if g_op == "CLICK":
        return None
    if p_op == "CLICK":
        return 0.0
    return char_f1(pred.get("value", ""), gold.get("value", ""))


def compute_element_distance(pred_id: int | None, gold_id: int) -> float:
    """
    Normalised distance: |pred_id - gold_id| / max(pred_id, gold_id).
    This is a proxy for structural closeness in the DOM (lower = better).
    Note: works on the remapped sequential IDs in the pruned HTML.
    """
    if pred_id is None:
        return 1.0
    denom = max(pred_id, gold_id)
    if denom == 0:
        return 0.0
    return abs(pred_id - gold_id) / denom


def aggregate(results: list[dict]) -> dict:
    step_sr = np.mean([r["step_sr"] for r in results])
    op_sr = np.mean([r["op_sr"] for r in results])
    el_dist = np.mean([r["element_dist"] for r in results])
    f1_vals = [r["op_f1"] for r in results if r["op_f1"] is not None]
    op_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")
    return {
        "n_steps": len(results),
        "SSR": float(step_sr),
        "Op_F1": op_f1,
        "Element_Dist": float(el_dist),
        "Op_SR": float(op_sr),
    }


# ── inference ──────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, item: dict, device: torch.device,
                  max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [
        {"role": "system", "content": item["instruction"]},
        {"role": "user", "content": item["input"]},
    ]

    if tokenizer.chat_template:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    else:
        text = f"{item['instruction']}\n\n{item['input']}\n"
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    eos_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    response_ids = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# ── evaluation loop ────────────────────────────────────────────────────────────

def evaluate_split(model, tokenizer, data_path: str | Path, device: torch.device,
                   args) -> tuple[list[dict], dict]:
    with open(data_path) as f:
        dataset = json.load(f)

    per_step: list[dict] = []
    for item in tqdm(dataset, desc=str(Path(data_path).stem)):
        gold_str = item["output"]
        gold = parse_output(gold_str)
        if gold["id"] is None:
            continue

        try:
            pred_str = run_inference(model, tokenizer, item, device,
                                     args.max_new_tokens, args.temperature, args.top_p)
            pred = parse_output(pred_str)
        except Exception as e:
            print(f"  [warn] generation error: {e}")
            pred = {"id": None, "op": None, "value": ""}
            pred_str = ""

        step_sr = (pred["id"] == gold["id"] and pred["op"] == gold["op"])
        op_sr = (pred["op"] == gold["op"])
        op_f1 = compute_op_f1(pred, gold)
        el_dist = compute_element_distance(pred["id"], gold["id"])

        row = {
            "step_sr": int(step_sr),
            "op_sr": int(op_sr),
            "op_f1": op_f1,
            "element_dist": el_dist,
        }
        if args.save_predictions:
            row["prediction"] = pred_str
            row["gold"] = gold_str
        per_step.append(row)

    return per_step, aggregate(per_step)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WEPO LLM on Mind2Web test sets")
    p.add_argument("--model", required=True, help="HF model or LoRA checkpoint path")
    p.add_argument("--base_model", default=None,
                   help="Base model path if --model is a LoRA adapter")
    p.add_argument("--data", nargs="+", required=True,
                   help="One or more test JSON files (output of scripts/prepare_test.py)")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Split names matching --data files (for reporting)")
    p.add_argument("--output", required=True, help="Output JSON path for results")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy, default: 0)")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--save_predictions", action="store_true",
                   help="Also save raw prediction strings in the output")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if args.base_model:
        print(f"Loading base model {args.base_model} + LoRA adapter {args.model} …")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
    else:
        print(f"Loading model {args.model} …")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    split_names = args.splits or [Path(d).stem for d in args.data]
    all_results = {}

    for split_name, data_path in zip(split_names, args.data):
        print(f"\n── {split_name} ──")
        per_step, summary = evaluate_split(model, tokenizer, data_path, device, args)
        all_results[split_name] = {"summary": summary, "per_step": per_step}
        print(
            f"  SSR={summary['SSR']:.4f}  "
            f"Op_F1={summary['Op_F1']:.4f}  "
            f"El_Dist={summary['Element_Dist']:.4f}  "
            f"n={summary['n_steps']}"
        )

    # overall average across splits
    if len(args.data) > 1:
        all_steps = [r for v in all_results.values() for r in v["per_step"]]
        overall = aggregate(all_steps)
        all_results["overall"] = {"summary": overall}
        print(f"\n── OVERALL ──")
        print(
            f"  SSR={overall['SSR']:.4f}  "
            f"Op_F1={overall['Op_F1']:.4f}  "
            f"El_Dist={overall['Element_Dist']:.4f}  "
            f"n={overall['n_steps']}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
