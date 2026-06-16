"""
Fine-tune DeBERTa as a web element ranking model.

Supports two loss functions:
  --loss bce      : Binary cross-entropy on individual (context, element) pairs.
                    Fast and simple; treats each pair independently.
  --loss infonce  : InfoNCE / in-step contrastive loss.
                    For each page step, one positive vs up to --max_neg negatives
                    are grouped and trained with a temperature-scaled cross-entropy
                    (softmax over the group), encouraging the model to rank the
                    positive element above all negatives simultaneously.

Usage:
  # BCE
  python deberta/train.py \
      --data_dir  data/train_dataset \
      --output_dir checkpoints/deberta_bce \
      --loss bce \
      --model microsoft/deberta-v3-base \
      --epochs 3 --batch_size 32 --lr 2e-5

  # InfoNCE
  python deberta/train.py \
      --data_dir  data/train_dataset \
      --output_dir checkpoints/deberta_infonce \
      --loss infonce \
      --max_neg 10 \
      --temperature 0.07 \
      --model microsoft/deberta-v3-base \
      --epochs 3 --batch_size 8 --lr 2e-5
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))
from deberta.dataset import ElementRankingDataset, InfoNCEBatch
from deberta.model import ElementRanker


# ── loss functions ─────────────────────────────────────────────────────────────

def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard binary cross-entropy for independent (context, element) pairs."""
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def infonce_loss(
    group_logits: torch.Tensor,  # (B, G)  – G = 1 pos + up-to-max_neg neg
    group_labels: torch.Tensor,  # (B, G)  – 1/0/-1(pad)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss over contrastive groups.

    For each group the positive is always the first valid element (label==1).
    Padding positions (label==-1) are masked out before softmax.

    L = -log( exp(s_pos/τ) / Σ_{j: label≠-1} exp(s_j/τ) )
    """
    B, G = group_logits.shape
    scaled = group_logits / temperature
    # mask padding
    pad_mask = (group_labels == -1)  # (B, G)
    scaled = scaled.masked_fill(pad_mask, float("-inf"))
    # positive index is 0 by construction (see InfoNCEBatch)
    log_softmax = F.log_softmax(scaled, dim=-1)  # (B, G)
    # loss = -log p(positive)
    loss = -log_softmax[:, 0]  # (B,)
    # only count groups that have a valid positive
    valid = (~pad_mask[:, 0]).float()
    return (loss * valid).sum() / valid.sum().clamp(min=1)


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bce(model: ElementRanker, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        tt_ids = batch.get("token_type_ids")
        if tt_ids is not None:
            tt_ids = tt_ids.to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids, attn_mask, tt_ids)
        total_loss += bce_loss(logits, labels).item() * len(labels)
        preds = (logits > 0).long()
        correct += (preds == labels).sum().item()
        total += len(labels)
    model.train()
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_infonce(model: ElementRanker, loader: DataLoader, device: torch.device,
                     temperature: float) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        B, G, L = batch["group_input_ids"].shape
        ids = batch["group_input_ids"].view(B * G, L).to(device)
        mask = batch["group_attn_mask"].view(B * G, L).to(device)
        tt = batch.get("group_token_types")
        if tt is not None:
            tt = tt.view(B * G, L).to(device)
        logits = model(ids, mask, tt).view(B, G)
        labels = batch["group_labels"].to(device)
        loss = infonce_loss(logits, labels, temperature)
        valid = (labels != -1).any(dim=-1)
        total_loss += loss.item() * valid.sum().item()
        # recall@1: positive (col 0) has highest logit in group
        ranked_first = logits.argmax(dim=-1) == 0
        correct += (ranked_first & valid).sum().item()
        total += valid.sum().item()
    model.train()
    return {"loss": total_loss / max(total, 1), "recall@1": correct / max(total, 1)}


# ── data helpers ───────────────────────────────────────────────────────────────

def _collate_bce(batch: list[dict]):
    """Standard collate for BCE: pad sequences to max length in batch."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [torch.tensor(b["input_ids"]) for b in batch], batch_first=True, padding_value=0
    )
    attn_mask = pad_sequence(
        [torch.tensor(b["attention_mask"]) for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    result = {"input_ids": input_ids, "attention_mask": attn_mask, "label": labels}
    if batch[0]["token_type_ids"] is not None:
        tt = pad_sequence(
            [torch.tensor(b["token_type_ids"]) for b in batch], batch_first=True, padding_value=0
        )
        result["token_type_ids"] = tt
    return result


# ── training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ElementRanker(base_model_name=args.model, dropout=args.dropout).to(device)

    data_paths = sorted(Path(args.data_dir).glob("train_*.json"))
    if not data_paths:
        raise FileNotFoundError(f"No train_*.json in {args.data_dir}")

    dataset = ElementRankingDataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_total_len=args.max_len,
        neg_per_step=args.max_neg if args.loss == "infonce" else None,
    )

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if args.loss == "bce":
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=_collate_bce, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                                shuffle=False, collate_fn=_collate_bce, num_workers=2)
    else:
        infonce_collator = InfoNCEBatch(tokenizer, max_neg=args.max_neg, max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=infonce_collator, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=infonce_collator, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    best_val_metric = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if args.loss == "bce":
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    tt_ids = batch.get("token_type_ids")
                    if tt_ids is not None:
                        tt_ids = tt_ids.to(device)
                    labels = batch["label"].to(device)
                    logits = model(input_ids, attn_mask, tt_ids)
                    loss = bce_loss(logits, labels)

                else:  # infonce
                    B, G, L = batch["group_input_ids"].shape
                    ids = batch["group_input_ids"].view(B * G, L).to(device)
                    mask = batch["group_attn_mask"].view(B * G, L).to(device)
                    tt = batch.get("group_token_types")
                    if tt is not None:
                        tt = tt.view(B * G, L).to(device)
                    logits = model(ids, mask, tt).view(B, G)
                    labels = batch["group_labels"].to(device)
                    loss = infonce_loss(logits, labels, args.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} step {step}/{len(train_loader)}  loss={avg:.4f}  lr={lr_now:.2e}")
                running_loss = 0.0

        # validation
        if args.loss == "bce":
            val_metrics = evaluate_bce(model, val_loader, device)
            key_metric = val_metrics["accuracy"]
            print(f"Epoch {epoch} val  loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")
        else:
            val_metrics = evaluate_infonce(model, val_loader, device, args.temperature)
            key_metric = val_metrics["recall@1"]
            print(f"Epoch {epoch} val  loss={val_metrics['loss']:.4f}  recall@1={val_metrics['recall@1']:.4f}")

        if key_metric > best_val_metric:
            best_val_metric = key_metric
            model.save(args.output_dir)
            print(f"  ✓ Saved best model (metric={key_metric:.4f}) to {args.output_dir}")

    print(f"\nTraining complete. Best val metric: {best_val_metric:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DeBERTa element ranker")
    p.add_argument("--data_dir", required=True, help="Dir with train_*.json shards")
    p.add_argument("--output_dir", required=True, help="Where to save the best checkpoint")
    p.add_argument("--model", default="microsoft/deberta-v3-base",
                   help="HF model name/path (default: deberta-v3-base)")
    p.add_argument("--loss", choices=["bce", "infonce"], default="bce",
                   help="Loss function (default: bce)")

    # InfoNCE-specific
    p.add_argument("--max_neg", type=int, default=10,
                   help="Max negatives per step for InfoNCE grouping (default: 10)")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature τ (default: 0.07)")

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (BCE: #pairs, InfoNCE: #steps)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=512, help="Max token length per sample")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--fp16", action="store_true", help="Use mixed-precision training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
