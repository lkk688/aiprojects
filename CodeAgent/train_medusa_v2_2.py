# medusa_train_kd_adaptive.py
# ============================================================
# Medusa KD training for code speculative decoding (H100-friendly)
#
# Upgrades included:
#  1) Projection-path alignment with MLX inference:
#       student_logits = lm_head(norm(feat))   (optional switch)
#     This must match your inference verifier path (lm_head(norm(vx))).
#
#  2) Top-k distillation (teacher top-k only) + temperature scaling.
#
#  3) Hard CE:
#       - Head-1 uses ground-truth targets (stable)
#       - Head>=2 uses teacher-top1 targets (maximizes acceptance under verifier)
#
#  4) Head weighting to protect head-1 and reduce far-head interference.
#
#  5) CE coefficient scheduling (decays over training).
#
#  6) Longer seq_len + optional FIM augmentation.
#
#  7) Better logging: per-head accuracy + prefix acceptance proxy, plus best-checkpoint saving.
#
# Notes:
#  - Base model is frozen. Only Medusa heads are trained.
#  - Dataset is streaming by default; keep DataLoader num_workers=0.
# ============================================================

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
# 0) Config
# ============================================================
@dataclass
class TrainConfig:
    # Model
    model_id: str = "Qwen/Qwen2.5-Coder-7B"

    # Data
    dataset_name: str = "bigcode/the-stack-smol"
    split: str = "train"
    data_dir: Optional[str] = "data/python"
    seq_len: int = 1024
    batch_size: int = 4
    min_chars: int = 64

    # FIM augmentation
    use_fim: bool = True
    fim_rate: float = 0.5
    fim_middle_min_ratio: float = 0.1
    fim_middle_max_ratio: float = 0.3

    # Medusa
    num_heads: int = 3
    # Stronger downweight far heads; reduces interference and improves prefix acceptance stability
    head_weights: Optional[List[float]] = None

    # Optimization
    steps: int = 4000                 # gradient steps (micro-steps)
    accum_steps: int = 4              # gradient accumulation
    lr_peak: float = 1e-4             # lower than before (tiny heads do not need large LR)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_updates: int = 200         # more warmup for stability
    log_every_updates: int = 10
    eval_every_updates: int = 20

    # Distillation
    temperature: float = 2.0
    top_k: int = 64                   # teacher top-k for KD
    kd_alpha: float = 1.0             # KD weight
    # Hard CE schedule: starts higher then decays
    hard_ce_alpha_start: float = 0.2
    hard_ce_alpha_end: float = 0.05

    # Alignment with inference
    # If your MLX verifier uses lm_head(norm(vx)), set this True (recommended).
    apply_norm_before_lm_head: bool = True

    # Acceptance-oriented hard targets:
    # Head-1 uses ground-truth; head>=2 uses teacher-top1 by default.
    use_teacher_top1_ce_for_far_heads: bool = True

    # Saving
    save_path: str = "medusa_heads_kd_topk.pth"
    save_best: bool = True
    best_metric: str = "prefix_acc_2"  # save checkpoint that maximizes this metric

    # Repro
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
# 2) FIM augmenter (raw-text)
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
# 3) Streaming dataset (fixed-length tokens)
# ============================================================
class StreamingCodingDataset(IterableDataset):
    """
    Streaming dataset that yields fixed-length token sequences.
    For stable shapes (and speed), only yield sequences with exact seq_len.
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
# 4) Distillation losses
# ============================================================
def topk_kd_loss(
    student_logits: torch.Tensor,     # [B,L,V]
    teacher_logits: torch.Tensor,     # [B,L,V]
    top_k: int,
    temperature: float,
    mask: Optional[torch.Tensor] = None,  # [B,L]
) -> torch.Tensor:
    """
    Top-k KD:
      - take teacher top-k logits per position
      - compute KL(teacher || student) restricted to that support
      - multiply by T^2
    """
    T = temperature

    # Teacher top-k
    t = teacher_logits / T
    topk_vals, topk_idx = torch.topk(t, k=top_k, dim=-1)          # [B,L,K]
    t_probs = torch.softmax(topk_vals, dim=-1)                     # [B,L,K]

    # Student logits restricted to the same top-k support
    s = (student_logits / T).gather(dim=-1, index=topk_idx)        # [B,L,K]
    s_log_probs = torch.log_softmax(s, dim=-1)                     # [B,L,K]

    # KL(teacher || student) over top-k support
    kl = (t_probs * (torch.log(t_probs.clamp_min(1e-9)) - s_log_probs)).sum(dim=-1)  # [B,L]

    if mask is not None:
        loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = kl.mean()

    return loss * (T * T)


def hard_ce_loss(
    student_logits: torch.Tensor,  # [B,L,V]
    targets: torch.Tensor,         # [B,L]
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, L, V = student_logits.shape
    loss = F.cross_entropy(student_logits.view(-1, V), targets.view(-1), reduction="none")
    loss = loss.view(B, L)
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = loss.mean()
    return loss


# ============================================================
# 5) Metrics: per-head accuracy and prefix acceptance proxy
# ============================================================
@torch.no_grad()
def compute_acceptance_metrics(
    base_model,
    medusa: MedusaHeads,
    input_ids: torch.Tensor,
    apply_norm_before_lm_head: bool,
) -> Dict[str, float]:
    device = input_ids.device
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    out = base_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
    h = out.hidden_states[-1]  # [B,L,H]
    feats = medusa(h)          # list of [B,L,H]

    B, L = input_ids.shape
    correct_per_head = []

    for k in range(medusa.num_heads):
        shift = k + 1
        proj = base_model.model.norm(feats[k]) if apply_norm_before_lm_head else feats[k]
        logits = base_model.lm_head(proj)  # [B,L,V]
        s = logits[:, :-shift, :]
        targets = input_ids[:, shift:]
        pred = torch.argmax(s, dim=-1)
        correct = (pred == targets).float()  # [B,L-shift]
        correct_per_head.append(correct)

    metrics = {}
    for k, corr in enumerate(correct_per_head):
        metrics[f"acc_head_{k+1}"] = corr.mean().item()

    for m in range(1, medusa.num_heads + 1):
        min_len = L - m
        if min_len <= 0:
            continue
        prefix_ok = torch.ones((B, min_len), device=device)
        for k in range(m):
            prefix_ok = prefix_ok * correct_per_head[k][:, :min_len]
        metrics[f"prefix_acc_{m}"] = prefix_ok.mean().item()

    return metrics


# ============================================================
# 6) Scheduler utilities
# ============================================================
def lr_for_update(update_idx: int, total_updates: int, lr_peak: float, warmup_updates: int) -> float:
    """
    Warmup + cosine decay.
    """
    if update_idx < warmup_updates:
        return lr_peak * (update_idx + 1) / max(1, warmup_updates)
    progress = (update_idx - warmup_updates) / max(1, total_updates - warmup_updates)
    progress = min(max(progress, 0.0), 1.0)
    return lr_peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def ce_alpha_for_update(update_idx: int, total_updates: int, start: float, end: float) -> float:
    """
    Linear decay schedule for hard CE weight.
    """
    progress = update_idx / max(1, total_updates)
    return start * (1.0 - progress) + end * progress


# ============================================================
# 7) Training
# ============================================================
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"Loading tokenizer/model: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # Use torch_dtype explicitly; transformers warns that torch_dtype is deprecated in some wrappers,
    # but for AutoModelForCausalLM this is standard and stable.
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
        # Strong downweighting of farther heads to stabilize training and improve prefix acceptance
        # Suggested default for K=3: [1.0, 0.35, 0.15]
        if cfg.num_heads == 3:
            cfg.head_weights = [1.0, 0.35, 0.15]
        else:
            cfg.head_weights = [1.0] + [0.35 * (0.6 ** (i - 1)) for i in range(2, cfg.num_heads + 1)]
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

    # Streaming dataset: num_workers should be 0 (often only 1 shard)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        medusa.parameters(),
        lr=cfg.lr_peak,
        weight_decay=cfg.weight_decay,
    )

    total_updates = cfg.steps // cfg.accum_steps
    best_val = -1.0
    best_state = None

    print(
        f"\n🚀 Medusa KD Training | steps={cfg.steps} | accum={cfg.accum_steps} | "
        f"seq_len={cfg.seq_len} | top_k={cfg.top_k} | T={cfg.temperature} | "
        f"kd_alpha={cfg.kd_alpha} | CE {cfg.hard_ce_alpha_start}->{cfg.hard_ce_alpha_end} | "
        f"norm_before_lm_head={cfg.apply_norm_before_lm_head}\n"
    )

    data_iter = iter(train_loader)
    running_loss = 0.0
    updates = 0

    # Use autocast for speed
    use_cuda = (device.type == "cuda")
    autocast_dtype = torch.bfloat16

    for step in range(1, cfg.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        # Teacher forward (frozen base model)
        with torch.no_grad():
            t_out = base_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            final_h = t_out.hidden_states[-1]                  # [B,L,H]
            teacher_logits = t_out.logits.to(torch.float32)    # [B,L,V]

        # Medusa forward
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_cuda):
            feats = medusa(final_h)  # list of [B,L,H]

        # Hard CE schedule per update (not per micro-step)
        ce_alpha = ce_alpha_for_update(updates, total_updates, cfg.hard_ce_alpha_start, cfg.hard_ce_alpha_end)

        total_loss = 0.0
        for k in range(cfg.num_heads):
            shift = k + 1

            # Student projection aligned with inference:
            # student_logits = lm_head(norm(feat)) if apply_norm_before_lm_head else lm_head(feat)
            proj = base_model.model.norm(feats[k]) if cfg.apply_norm_before_lm_head else feats[k]
            student_logits_full = base_model.lm_head(proj).to(torch.float32)  # [B,L,V]

            # Align sequences: head k predicts token at t+shift
            s = student_logits_full[:, :-shift, :].contiguous()   # [B,L-shift,V]
            t = teacher_logits[:, shift:, :].contiguous()         # [B,L-shift,V]
            y = input_ids[:, shift:].contiguous()                 # [B,L-shift]

            # Valid mask (all ones: we enforce exact seq_len)
            mask = torch.ones(y.shape, device=device, dtype=torch.float32)

            # KD loss (top-k)
            kd = topk_kd_loss(s, t, top_k=cfg.top_k, temperature=cfg.temperature, mask=mask)

            # Hard CE targets:
            # - head-1 uses ground-truth (y)
            # - head>=2 uses teacher top-1 by default (better aligns with verifier acceptance)
            if ce_alpha > 0.0:
                if (k == 0) or (not cfg.use_teacher_top1_ce_for_far_heads):
                    ce_targets = y
                else:
                    ce_targets = torch.argmax(t, dim=-1)  # teacher top1 at aligned positions

                ce = hard_ce_loss(s, ce_targets, mask=mask)
                head_loss = cfg.kd_alpha * kd + ce_alpha * ce
            else:
                head_loss = cfg.kd_alpha * kd

            total_loss = total_loss + cfg.head_weights[k] * head_loss

        total_loss = total_loss / cfg.accum_steps
        total_loss.backward()
        running_loss += total_loss.item() * cfg.accum_steps

        if step % cfg.accum_steps == 0:
            # LR schedule
            lr_now = lr_for_update(updates, total_updates, cfg.lr_peak, cfg.warmup_updates)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            torch.nn.utils.clip_grad_norm_(medusa.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

            if updates % cfg.log_every_updates == 0:
                avg_loss = running_loss / cfg.log_every_updates
                print(f"Update {updates:05d} | loss={avg_loss:.4f} | lr={lr_now:.2e} | ce_alpha={ce_alpha:.3f}")
                running_loss = 0.0

            if updates % cfg.eval_every_updates == 0:
                metrics = compute_acceptance_metrics(
                    base_model=base_model,
                    medusa=medusa,
                    input_ids=input_ids,
                    apply_norm_before_lm_head=cfg.apply_norm_before_lm_head,
                )
                msg = " | ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
                print(f"  Metrics @Update {updates:05d}: {msg}")

                # Save best checkpoint based on chosen metric
                if cfg.save_best and (cfg.best_metric in metrics):
                    val = metrics[cfg.best_metric]
                    if val > best_val:
                        best_val = val
                        best_state = {k: v.detach().cpu() for k, v in medusa.state_dict().items()}
                        print(f"  ✅ New best {cfg.best_metric}={best_val:.4f} at update {updates:05d}")

    print("✅ Training complete.")

    # Save final
    print("Saving final Medusa weights...")
    torch.save(medusa.state_dict(), cfg.save_path)
    print(f"Saved to: {cfg.save_path}")

    # Save best (if available)
    if cfg.save_best and best_state is not None:
        best_path = cfg.save_path.replace(".pth", f".best_{cfg.best_metric}.pth")
        torch.save(best_state, best_path)
        print(f"Saved best checkpoint to: {best_path} (best {cfg.best_metric}={best_val:.4f})")


if __name__ == "__main__":
    cfg = TrainConfig(
        # Core
        model_id="Qwen/Qwen2.5-Coder-7B",
        num_heads=3,

        # Data / context
        seq_len=1024,          # try 1024, then 2048 once stable
        batch_size=4,

        # Optimization
        steps=4000,
        accum_steps=4,
        lr_peak=1e-4,
        warmup_updates=200,

        # Distillation
        top_k=64,
        temperature=2.0,
        kd_alpha=1.0,
        hard_ce_alpha_start=0.2,
        hard_ce_alpha_end=0.05,

        # Inference alignment (set True if MLX verifier uses lm_head(norm(vx)))
        apply_norm_before_lm_head=True,

        # Acceptance-oriented CE for far heads
        use_teacher_top1_ce_for_far_heads=True,

        # FIM
        use_fim=True,
        fim_rate=0.5,

        # Saving
        save_path="medusa_heads_kd_topk.pth",
        save_best=True,
        best_metric="prefix_acc_2",
    )
    train(cfg)