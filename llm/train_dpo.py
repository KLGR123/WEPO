"""
Fine-tune an LLM with DPO + LoRA for WEPO. Supports two loss objectives:

  --loss_type ndpo      (default) n-way Plackett-Luce DPO. Given prompt x,
                        chosen a_w, and n rejected {a_l1, …, a_ln}, all n+1
                        responses are scored jointly in a single softmax:

                            L = -E[ log( exp(r_w) / (exp(r_w) + Σ_i exp(r_li)) ) ]

                        where r_k = β · [log π_θ(a_k|x) - log π_ref(a_k|x)].
                        The model is penalised for ranking ANY negative above
                        the positive in one shot — this is the loss used to
                        reproduce the paper's results (β=0.95, n=3).

  --loss_type pairwise   Standard DPO (Rafailov et al., 2023), applied to each
                        of the n (chosen, rejected_i) pairs independently and
                        averaged:

                            L = -E[ 1/n Σ_i log σ(r_w - r_li) ]

                        Use this if you want a drop-in replacement for
                        TRL's DPOTrainer behaviour, or to ablate against nDPO.

Usage:
  # nDPO (paper default), Llama-3-8B, distance-based negatives (n=3)
  python llm/train_dpo.py \
      --model  meta-llama/Meta-Llama-3-8B-Instruct \
      --data   data/mind2web_dpo_train.json \
      --output checkpoints/llama3_wepo \
      --loss_type ndpo --beta 0.95 --n_neg 3 \
      --lora_r 16 --lora_alpha 32

  # Pairwise DPO ablation
  python llm/train_dpo.py \
      --model  meta-llama/Meta-Llama-3-8B-Instruct \
      --data   data/mind2web_dpo_train.json \
      --output checkpoints/llama3_wepo_pairwise \
      --loss_type pairwise --beta 0.95 --n_neg 3

LoRA defaults follow the paper (r=16, α=32, dropout=0.05, query/key/value/output).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm.dataset import WEPODataset


# ── log-probability computation ────────────────────────────────────────────────

def sequence_log_prob(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_start: int
) -> torch.Tensor:
    """
    Sum of token log-probabilities for the completion tokens only.

    Args:
        completion_start: index (in the un-shifted sequence) where the
                          completion begins; everything before is prompt
                          and excluded from the sum.
    Returns:
        log_prob: (B,)

    Note: gradient tracking follows whatever context is active at the call
    site (use `with torch.no_grad():` around calls for the frozen reference
    model; leave it enabled for the trainable policy).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)
    # shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]       # (B, L-1, V)
    shift_labels = input_ids[:, 1:]        # (B, L-1)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    tok_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    mask = attention_mask[:, 1:].float()   # (B, L-1), excludes right-padding
    if completion_start > 1:
        # token at position `completion_start` is first predicted by shifted
        # index `completion_start - 1`; zero out everything before that.
        mask[:, : completion_start - 1] = 0.0
    return (tok_lp * mask).sum(dim=-1)     # (B,)


# ── reward computation shared by both loss types ───────────────────────────────

def compute_rewards(
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    prompt_ids: torch.Tensor,         # (B, P)
    prompt_mask: torch.Tensor,        # (B, P)
    chosen_ids: torch.Tensor,         # (B, C)
    chosen_mask: torch.Tensor,        # (B, C)
    rejected_ids: torch.Tensor,       # (B, N, R)
    rejected_mask: torch.Tensor,      # (B, N, R)
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (r_w, r_l):
      r_w: (B,)    reward of the chosen response
      r_l: (B, N)  reward of each of the N rejected responses
    """
    B, N, _ = rejected_ids.shape
    P = prompt_ids.shape[1]

    def _cat(comp_ids, comp_mask):
        return (
            torch.cat([prompt_ids, comp_ids], dim=-1),
            torch.cat([prompt_mask, comp_mask], dim=-1),
        )

    # chosen
    chosen_full_ids, chosen_full_mask = _cat(chosen_ids, chosen_mask)
    chosen_lp_pol = sequence_log_prob(policy, chosen_full_ids, chosen_full_mask, P)
    with torch.no_grad():
        chosen_lp_ref = sequence_log_prob(ref_policy, chosen_full_ids, chosen_full_mask, P)
    r_w = beta * (chosen_lp_pol - chosen_lp_ref)  # (B,)

    # rejected — flatten (B, N) → (B*N) for a single forward pass
    rej_ids_flat = rejected_ids.reshape(B * N, -1)
    rej_mask_flat = rejected_mask.reshape(B * N, -1)
    prompt_ids_rep = prompt_ids.unsqueeze(1).expand(B, N, P).reshape(B * N, P)
    prompt_mask_rep = prompt_mask.unsqueeze(1).expand(B, N, P).reshape(B * N, P)

    rej_full_ids = torch.cat([prompt_ids_rep, rej_ids_flat], dim=-1)
    rej_full_mask = torch.cat([prompt_mask_rep, rej_mask_flat], dim=-1)

    rej_lp_pol = sequence_log_prob(policy, rej_full_ids, rej_full_mask, P)
    with torch.no_grad():
        rej_lp_ref = sequence_log_prob(ref_policy, rej_full_ids, rej_full_mask, P)
    r_l = (beta * (rej_lp_pol - rej_lp_ref)).view(B, N)

    return r_w, r_l


# ── loss functions ─────────────────────────────────────────────────────────────

def ndpo_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    n-way Plackett-Luce DPO: chosen must rank first among {chosen, n rejected}.
    Implemented as cross-entropy with target index 0 over [r_w, r_l_1..r_l_n].
    """
    B = r_w.shape[0]
    all_rewards = torch.cat([r_w.unsqueeze(1), r_l], dim=1)  # (B, N+1)
    target = torch.zeros(B, dtype=torch.long, device=r_w.device)
    loss = F.cross_entropy(all_rewards, target)
    return loss, _diagnostics(r_w, r_l)


def pairwise_dpo_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Standard pairwise DPO (Rafailov et al., 2023), averaged over the N
    (chosen, rejected_i) pairs per sample:
        L = E[ 1/N Σ_i -log σ(r_w - r_li) ]
    """
    diff = r_w.unsqueeze(1) - r_l  # (B, N)
    loss = -F.logsigmoid(diff).mean(dim=1).mean()
    return loss, _diagnostics(r_w, r_l)


def _diagnostics(r_w: torch.Tensor, r_l: torch.Tensor) -> dict:
    with torch.no_grad():
        rejected_mean = r_l.mean(dim=1)
        acc = (r_w > rejected_mean).float().mean()
        margin = (r_w - rejected_mean).mean()
    return {
        "chosen_reward": r_w.mean().item(),
        "rejected_reward": rejected_mean.mean().item(),
        "reward_margin": margin.item(),
        "accuracy": acc.item(),
    }


LOSS_FNS = {"ndpo": ndpo_loss, "pairwise": pairwise_dpo_loss}


# ── collator ───────────────────────────────────────────────────────────────────

class DPOCollator:
    """
    Tokenises and pads a batch of WEPODataset items.

    Each item: {"prompt": str, "chosen": str, "rejected": [str, …]}
    The rejected list is trimmed/padded (by repeating the last element) to
    exactly n_neg entries so the batch can be stacked into a regular tensor.
    """

    def __init__(self, tokenizer, max_prompt_len: int = 7936, max_comp_len: int = 256, n_neg: int = 3):
        self.tok = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_comp_len = max_comp_len
        self.n_neg = n_neg
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _encode(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(texts, max_length=max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def __call__(self, batch: list[dict]) -> dict:
        prompts = [b["prompt"] for b in batch]
        chosen = [b["chosen"] for b in batch]

        rejected_lists = []
        for b in batch:
            r = b["rejected"][: self.n_neg]
            if not r:
                raise ValueError("Sample has no rejected completions; check dataset.")
            while len(r) < self.n_neg:
                r = r + [r[-1]]
            rejected_lists.append(r)
        flat_rejected = [r for rl in rejected_lists for r in rl]

        p_ids, p_mask = self._encode(prompts, self.max_prompt_len)
        c_ids, c_mask = self._encode(chosen, self.max_comp_len)
        r_ids, r_mask = self._encode(flat_rejected, self.max_comp_len)

        B, N = len(batch), self.n_neg
        return {
            "prompt_ids": p_ids,
            "prompt_mask": p_mask,
            "chosen_ids": c_ids,
            "chosen_mask": c_mask,
            "rejected_ids": r_ids.view(B, N, -1),
            "rejected_mask": r_mask.view(B, N, -1),
        }


# ── training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}  |  Loss: {args.loss_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ── policy model with LoRA ─────────────────────────────────────────────────
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none",
    )
    policy = get_peft_model(base_model, lora_cfg)
    policy.print_trainable_parameters()

    # ── frozen reference model ──────────────────────────────────────────────────
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    ref_model.requires_grad_(False)
    ref_model.eval()

    # ── data ─────────────────────────────────────────────────────────────────
    dataset = WEPODataset(args.data, tokenizer=tokenizer)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    collator = DPOCollator(tokenizer, max_prompt_len=args.max_prompt_len,
                           max_comp_len=args.max_comp_len, n_neg=args.n_neg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator, num_workers=2)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    use_fp16_scaler = (not args.bf16) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    loss_fn = LOSS_FNS[args.loss_type]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    def run_batch(batch):
        r_w, r_l = compute_rewards(
            policy=policy, ref_policy=ref_model,
            prompt_ids=batch["prompt_ids"], prompt_mask=batch["prompt_mask"],
            chosen_ids=batch["chosen_ids"], chosen_mask=batch["chosen_mask"],
            rejected_ids=batch["rejected_ids"], rejected_mask=batch["rejected_mask"],
            beta=args.beta,
        )
        return loss_fn(r_w, r_l)

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = {"loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0}
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                enabled=torch.cuda.is_available(), dtype=autocast_dtype):
                loss, metrics = run_batch(batch)
                scaled_loss = loss / args.grad_accum

            scaler.scale(scaled_loss).backward()

            running["loss"] += loss.item()
            running["accuracy"] += metrics["accuracy"]
            running["reward_margin"] += metrics["reward_margin"]

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % (args.log_every * args.grad_accum) == 0:
                n = args.log_every * args.grad_accum
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch} step {step}/{len(train_loader)} | "
                    f"loss={running['loss']/n:.4f} | "
                    f"acc={running['accuracy']/n:.3f} | "
                    f"margin={running['reward_margin']/n:.3f} | "
                    f"lr={lr_now:.2e}"
                )
                running = {k: 0.0 for k in running}

        # ── validation ─────────────────────────────────────────────────────────
        policy.eval()
        val_acc, val_loss, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, metrics = run_batch(batch)
                val_loss += loss.item()
                val_acc += metrics["accuracy"]
                val_n += 1
        val_loss /= max(val_n, 1)
        val_acc /= max(val_n, 1)
        print(f"Epoch {epoch} val | loss={val_loss:.4f} | acc={val_acc:.4f}")
        policy.train()

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = output_dir / "best"
            policy.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Saved best checkpoint (acc={val_acc:.4f}) -> {save_path}")

        epoch_path = output_dir / f"epoch_{epoch}"
        policy.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)

    print(f"\nTraining complete. Best val reward-accuracy: {best_acc:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train LLM with DPO/nDPO + LoRA for WEPO")

    p.add_argument("--model", required=True,
                   help="HF model name or local path (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    p.add_argument("--data", required=True,
                   help="Path to the DPO training JSON (output of scripts/prepare_train.py)")
    p.add_argument("--output", required=True, help="Directory to save checkpoints")

    p.add_argument("--loss_type", choices=["ndpo", "pairwise"], default="ndpo",
                   help="ndpo: joint Plackett-Luce loss over 1+n responses (paper default). "
                        "pairwise: standard DPO averaged over n (chosen, rejected) pairs.")
    p.add_argument("--beta", type=float, default=0.95,
                   help="KL penalty coefficient beta (paper default: 0.95)")
    p.add_argument("--n_neg", type=int, default=3,
                   help="Number of negatives per positive (paper default: 3)")

    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    p.add_argument("--lora_targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="LoRA target module names (default: q/k/v/o projections)")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size x grad_accum)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (paper: 1e-4)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_prompt_len", type=int, default=7936,
                   help="Max tokens for the prompt (HTML + intent + history)")
    p.add_argument("--max_comp_len", type=int, default=256,
                   help="Max tokens for a completion (Operation/Value/ID line)")
    p.add_argument("--val_fraction", type=float, default=0.02)
    p.add_argument("--log_every", type=int, default=20, help="Log every N gradient steps")
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 (recommended for Ampere+ GPUs; A100/H100)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
