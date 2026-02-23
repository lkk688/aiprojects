# medusa_train_kd_adaptive.py
# ============================================================
# Medusa KD training (H100-friendly) with:
#   - Top-k distillation (teacher top-k only)
#   - Optional hard CE to ground truth
#   - Longer seq_len + optional FIM augmentation
#   - Better logging: per-head accuracy + prefix acceptance proxy
#   - Streaming-safe DataLoader settings
# ============================================================

import os
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# ============================================================
# 0) Args / Config
# ============================================================
@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-Coder-7B"

    # Data
    dataset_name: str = "bigcode/the-stack-smol"
    split: str = "train"
    data_dir: str = "data/python"   # for the-stack-smol subsets
    seq_len: int = 1024             # longer context
    batch_size: int = 4
    min_chars: int = 64

    # FIM augmentation (Qwen tokens)
    use_fim: bool = True
    fim_rate: float = 0.5
    fim_middle_min_ratio: float = 0.1
    fim_middle_max_ratio: float = 0.3

    # Medusa
    num_heads: int = 3

    # Optimization
    steps: int = 4000
    accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_updates: int = 50

    # Distillation
    temperature: float = 2.0
    top_k: int = 64                 # top-k distillation
    kd_alpha: float = 1.0           # weight on KD loss
    hard_ce_alpha: float = 0.1      # weight on hard CE loss
    head_weights: Optional[List[float]] = None  # e.g. [1.0, 0.6, 0.4]

    # Logging
    log_every_updates: int = 10
    eval_every_updates: int = 20    # compute accuracy metrics periodically
    save_path: str = "medusa_heads_kd_topk.pth"

    # Reproducibility
    seed: int = 42


# ============================================================
# 1) Medusa Architecture
# ============================================================
class MedusaResBlock(nn.Module):
    """A tiny residual block that learns to look ahead."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHeads(nn.Module):
    """
    K lookahead heads implemented as a chain of tiny residual blocks.
    Output of block i is used to predict token t + (i+1).
    """
    def __init__(self, hidden_size: int, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([MedusaResBlock(hidden_size) for _ in range(num_heads)])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = hidden_states
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


# ============================================================
# 2) FIM Utility
# ============================================================
class FIMAugmenter:
    """
    Applies Fill-In-the-Middle augmentation in raw text space.
    Qwen2.5-Coder FIM tokens:
      <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>
    """
    def __init__(
        self,
        tokenizer,
        fim_rate: float = 0.5,
        middle_min_ratio: float = 0.1,
        middle_max_ratio: float = 0.3,
    ):
        self.tokenizer = tokenizer
        self.fim_rate = fim_rate
        self.middle_min_ratio = middle_min_ratio
        self.middle_max_ratio = middle_max_ratio

        self.prefix_token = "<|fim_prefix|>"
        self.suffix_token = "<|fim_suffix|>"
        self.middle_token = "<|fim_middle|>"
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"

    def maybe_apply(self, text: str) -> str:
        if random.random() >= self.fim_rate:
            return text + self.eos_token

        if not isinstance(text, str) or len(text) < 32:
            return text + self.eos_token

        doc_len = len(text)
        middle_len = int(doc_len * random.uniform(self.middle_min_ratio, self.middle_max_ratio))
        middle_len = max(1, min(middle_len, doc_len - 1))

        start = random.randint(0, doc_len - middle_len)
        end = start + middle_len

        prefix = text[:start]
        middle = text[start:end]
        suffix = text[end:]

        return f"{self.prefix_token}{prefix}{self.suffix_token}{suffix}{self.middle_token}{middle}{self.eos_token}"


# ============================================================
# 3) Streaming Dataset (tokenization + optional FIM)
# ============================================================
class StreamingCodingDataset(IterableDataset):
    """
    Streaming dataset that yields fixed-length token sequences.
    For maximum throughput stability, we only yield sequences with exact seq_len.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        split: str,
        seq_len: int,
        data_dir: Optional[str] = None,
        min_chars: int = 64,
        fim: Optional[FIMAugmenter] = None,
    ):
        self.dataset = load_dataset(dataset_name, data_dir=data_dir, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_chars = min_chars
        self.fim = fim

    def _get_text(self, item: Dict[str, Any]) -> Optional[str]:
        return item.get("content") or item.get("code") or item.get("text")

    def __iter__(self):
        for item in self.dataset:
            text = self._get_text(item)
            if not isinstance(text, str) or len(text) < self.min_chars:
                continue

            if self.fim is not None:
                text = self.fim.maybe_apply(text)
            else:
                text = text + (self.tokenizer.eos_token or "")

            toks = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )

            if toks.input_ids.shape[1] == self.seq_len:
                yield {"input_ids": toks.input_ids[0]}


# ============================================================
# 4) Distillation Losses (Top-k KD + Hard CE)
# ============================================================
def topk_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    temperature: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Top-k distillation:
      - For each position, take teacher top-k tokens and their probabilities.
      - Compute student log-probabilities restricted to those tokens.
      - Compute KL(teacher || student) on the restricted support.

    Shapes:
      student_logits: [B, L, V]
      teacher_logits: [B, L, V]
      mask: [B, L] float (1.0 valid, 0.0 ignore)
    """
    T = temperature

    # Teacher top-k indices and logits
    t_logits = teacher_logits / T
    topk_vals, topk_idx = torch.topk(t_logits, k=top_k, dim=-1)  # [B,L,K]

    # Teacher probabilities on top-k (normalized over top-k)
    t_probs = F.softmax(topk_vals, dim=-1)  # [B,L,K]

    # Student log-probs on the same top-k support
    s_logits = (student_logits / T).gather(dim=-1, index=topk_idx)  # [B,L,K]
    s_log_probs = F.log_softmax(s_logits, dim=-1)  # [B,L,K]

    # KL(teacher || student) on top-k support:
    # sum_k t_probs * (log t_probs - log s_probs)
    kl = (t_probs * (torch.log(t_probs.clamp_min(1e-9)) - s_log_probs)).sum(dim=-1)  # [B,L]

    if mask is not None:
        loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = kl.mean()

    # Standard distillation scaling
    return loss * (T * T)


def hard_ce_loss(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hard CE on ground-truth tokens.
    student_logits: [B,L,V]
    target_ids: [B,L] (int64)
    mask: [B,L] float (1.0 valid, 0.0 ignore)
    """
    B, L, V = student_logits.shape
    loss = F.cross_entropy(student_logits.view(-1, V), target_ids.view(-1), reduction="none")
    loss = loss.view(B, L)

    if mask is not None:
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = loss.mean()
    return loss


# ============================================================
# 5) Metrics (Acceptance Proxy)
# ============================================================
@torch.no_grad()
def compute_acceptance_metrics(
    base_model,
    medusa,
    input_ids: torch.Tensor,
    num_heads: int,
) -> Dict[str, float]:
    """
    Compute:
      - acc_head_k: head k top-1 accuracy on token t+k
      - prefix_acc_m: probability that first m lookahead tokens are all correct
    """
    device = input_ids.device
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    out = base_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
    h = out.hidden_states[-1]
    feats = medusa(h)

    B, L = input_ids.shape
    # Store correctness masks per head at aligned positions
    correct_per_head = []

    for k in range(num_heads):
        shift = k + 1
        student_logits = base_model.lm_head(feats[k])  # [B,L,V]
        # Align
        s = student_logits[:, :-shift, :]            # predicts positions [0..L-shift-1] -> targets [shift..L-1]
        targets = input_ids[:, shift:]               # [B,L-shift]
        pred = torch.argmax(s, dim=-1)
        correct = (pred == targets).float()          # [B,L-shift]
        correct_per_head.append(correct)

    metrics = {}
    for k, corr in enumerate(correct_per_head):
        metrics[f"acc_head_{k+1}"] = corr.mean().item()

    # Prefix acceptance: for m=1..K, fraction of positions where heads 1..m are all correct
    # Use the shortest aligned length among first m heads (which is L-m)
    for m in range(1, num_heads + 1):
        min_len = L - m
        if min_len <= 0:
            continue
        prefix_ok = torch.ones((B, min_len), device=device)
        for k in range(m):
            prefix_ok = prefix_ok * correct_per_head[k][:, :min_len]
        metrics[f"prefix_acc_{m}"] = prefix_ok.mean().item()

    return metrics


# ============================================================
# 6) Training Loop
# ============================================================
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"Loading tokenizer/model: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    device = next(base_model.parameters()).device
    print(f"Base model device: {device}")

    print(f"Initializing Medusa heads: K={cfg.num_heads}")
    medusa = MedusaHeads(hidden_size=base_model.config.hidden_size, num_heads=cfg.num_heads)
    medusa.to(device=device, dtype=torch.bfloat16)
    medusa.train()

    if cfg.head_weights is None:
        # Farther heads are harder; start with smaller weights
        cfg.head_weights = [1.0] + [0.6 ** i for i in range(1, cfg.num_heads)]
    assert len(cfg.head_weights) == cfg.num_heads

    fim_aug = None
    if cfg.use_fim:
        fim_aug = FIMAugmenter(
            tokenizer,
            fim_rate=cfg.fim_rate,
            middle_min_ratio=cfg.fim_middle_min_ratio,
            middle_max_ratio=cfg.fim_middle_max_ratio,
        )

    print("Loading streaming training data...")
    train_dataset = StreamingCodingDataset(
        tokenizer=tokenizer,
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        seq_len=cfg.seq_len,
        data_dir=cfg.data_dir,
        min_chars=cfg.min_chars,
        fim=fim_aug,
    )

    # Streaming-safe: num_workers should be 0 (or 1) because dataset.num_shards is often 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        medusa.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Simple warmup + cosine LR (per update)
    def lr_for_update(u: int) -> float:
        if u < cfg.warmup_updates:
            return cfg.lr * (u + 1) / max(1, cfg.warmup_updates)
        # cosine decay
        progress = (u - cfg.warmup_updates) / max(1, (cfg.steps // cfg.accum_steps) - cfg.warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    scaler_enabled = (device.type == "cuda")
    autocast_dtype = torch.bfloat16

    data_iter = iter(train_loader)
    running_loss = 0.0
    updates = 0

    print(
        f"\n🚀 Medusa KD Training | steps={cfg.steps} | accum={cfg.accum_steps} | "
        f"seq_len={cfg.seq_len} | top_k={cfg.top_k} | T={cfg.temperature} | "
        f"kd_alpha={cfg.kd_alpha} | hard_ce_alpha={cfg.hard_ce_alpha}\n"
    )

    for step in range(1, cfg.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)  # [B,L]
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        # Teacher: final hidden + teacher logits
        with torch.no_grad():
            t_out = base_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            final_h = t_out.hidden_states[-1]                  # [B,L,H]
            teacher_logits = t_out.logits.to(torch.float32)    # [B,L,V]

        # Medusa forward + losses
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=scaler_enabled):
            feats = medusa(final_h)  # list of [B,L,H]

        total = 0.0
        for k in range(cfg.num_heads):
            shift = k + 1

            # Student logits from frozen lm_head
            student_logits = base_model.lm_head(feats[k]).to(torch.float32)  # [B,L,V]

            # Align sequences
            s = student_logits[:, :-shift, :].contiguous()     # [B,L-shift,V]
            t = teacher_logits[:, shift:, :].contiguous()      # [B,L-shift,V]
            y = input_ids[:, shift:].contiguous()              # [B,L-shift]

            # Valid mask (all ones here because we force exact seq_len)
            mask = torch.ones(y.shape, device=device, dtype=torch.float32)

            kd = topk_kd_loss(s, t, top_k=cfg.top_k, temperature=cfg.temperature, mask=mask)

            if cfg.hard_ce_alpha > 0.0:
                ce = hard_ce_loss(s, y, mask=mask)
                head_loss = cfg.kd_alpha * kd + cfg.hard_ce_alpha * ce
            else:
                head_loss = cfg.kd_alpha * kd

            total = total + cfg.head_weights[k] * head_loss

        total = total / cfg.accum_steps
        total.backward()
        running_loss += total.item() * cfg.accum_steps

        if step % cfg.accum_steps == 0:
            # Update LR
            lr_now = lr_for_update(updates)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            torch.nn.utils.clip_grad_norm_(medusa.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

            if updates % cfg.log_every_updates == 0:
                avg_loss = running_loss / cfg.log_every_updates
                print(f"Update {updates:05d} | loss={avg_loss:.4f} | lr={lr_now:.2e}")
                running_loss = 0.0

            # Periodic acceptance metrics (cheap-ish, but do not run too often)
            if updates % cfg.eval_every_updates == 0:
                metrics = compute_acceptance_metrics(base_model, medusa, input_ids, cfg.num_heads)
                msg = " | ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
                print(f"  Metrics @Update {updates:05d}: {msg}")

    print("✅ Training complete. Saving Medusa weights...")
    torch.save(medusa.state_dict(), cfg.save_path)
    print(f"Saved to: {cfg.save_path}")


if __name__ == "__main__":
    cfg = TrainConfig(
        # Recommended defaults for H100 experiments:
        seq_len=1024,          # try 1024 first; then 2048
        batch_size=4,          # adjust for memory
        steps=4000,
        accum_steps=4,
        lr=3e-4,
        top_k=64,
        temperature=2.0,
        kd_alpha=1.0,
        hard_ce_alpha=0.1,
        num_heads=3,
        use_fim=True,
        fim_rate=0.5,
    )
    train(cfg)