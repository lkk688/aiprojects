import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# 1) MEDUSA ARCHITECTURE
# ==========================================
class MedusaResBlock(nn.Module):
    """A tiny residual block that learns to look ahead with minimal overhead."""
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

    def forward(self, hidden_states: torch.Tensor):
        feats = []
        x = hidden_states
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


# ==========================================
# 2) STREAMING DATASET
# ==========================================
class StreamingCodingDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset_name="bigcode/the-stack-smol",
        split="train",
        seq_len=256,
        data_dir="data/python",
        min_chars=64,
    ):
        self.dataset = load_dataset(dataset_name, data_dir=data_dir, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_chars = min_chars

    def __iter__(self):
        for item in self.dataset:
            code_text = item.get("content") or item.get("code") or item.get("text")
            if not isinstance(code_text, str) or len(code_text) < self.min_chars:
                continue

            toks = self.tokenizer(
                code_text,
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )
            # Require full length to keep shapes stable (good for throughput)
            if toks.input_ids.shape[1] == self.seq_len:
                yield {"input_ids": toks.input_ids[0]}


# ==========================================
# 3) KD TRAINING LOOP (FIXED + IMPROVED)
# ==========================================
def kd_kl_loss(student_logits, teacher_logits, T: float, mask=None):
    """
    KL(teacher || student) using teacher probs and student log-probs.
    Returns scalar loss.
    """
    student_logp = F.log_softmax(student_logits / T, dim=-1)
    teacher_p = F.softmax(teacher_logits / T, dim=-1)

    # KLDiv expects input=log_probs, target=probs
    kl = F.kl_div(student_logp, teacher_p, reduction="none")  # [B, L, V]
    kl = kl.sum(dim=-1)  # [B, L]

    if mask is not None:
        kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        kl = kl.mean()

    return kl * (T * T)


def train_medusa_kd(
    base_model,
    medusa,
    train_loader,
    steps=2000,
    lr=1e-3,
    accum_steps=4,
    T=2.0,
    kd_alpha=1.0,          # 1.0 = pure KD, 0.0 = pure hard CE (not typical here)
    hard_ce_alpha=0.0,     # optionally add some hard CE to true tokens
    head_weights=None,     # e.g., [1.0, 0.7, 0.5] to downweight harder farther heads
    max_grad_norm=1.0,
    log_every_updates=10,
):
    device = next(base_model.parameters()).device
    optimizer = torch.optim.AdamW(medusa.parameters(), lr=lr)

    if head_weights is None:
        head_weights = [1.0] * medusa.num_heads
    assert len(head_weights) == medusa.num_heads

    medusa.train()
    base_model.eval()

    print(f"\n🚀 Medusa KD Training | steps={steps} | accum={accum_steps} | T={T} | kd_alpha={kd_alpha}")

    data_iter = iter(train_loader)
    running = 0.0
    updates = 0

    autocast_dtype = torch.bfloat16 if base_model.dtype == torch.bfloat16 else torch.float16

    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)  # [B, L]
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            # Teacher forward (base model) - we need final hidden states and teacher logits
            teacher_out = base_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            final_h = teacher_out.hidden_states[-1]                 # [B, L, H]
            teacher_logits = teacher_out.logits.to(torch.float32)   # [B, L, V]

        # Student features from Medusa heads (cheap)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
            medusa_feats = medusa(final_h)  # list of [B, L, H]

            total_loss = 0.0
            for k, feats in enumerate(medusa_feats):
                shift = k + 1

                # Project to vocab using frozen lm_head
                student_logits = base_model.lm_head(feats).to(torch.float32)  # [B, L, V]

                # Align: head k predicts token at t+shift
                s = student_logits[:, :-shift, :].contiguous()
                t = teacher_logits[:, shift:, :].contiguous()

                # Mask for valid positions
                mask = torch.ones(s.shape[:2], device=device, dtype=torch.float32)

                kd = kd_kl_loss(s, t, T=T, mask=mask)

                # Optional hard CE to true next token (helps sometimes)
                if hard_ce_alpha > 0.0:
                    # True targets are input_ids shifted by 'shift'
                    true_targets = input_ids[:, shift:].contiguous()
                    ce = F.cross_entropy(
                        s.view(-1, s.size(-1)),
                        true_targets.view(-1),
                        reduction="mean",
                    )
                    head_loss = kd_alpha * kd + hard_ce_alpha * ce
                else:
                    head_loss = kd_alpha * kd

                total_loss = total_loss + head_weights[k] * head_loss

        total_loss = total_loss / accum_steps
        total_loss.backward()
        running += total_loss.item() * accum_steps

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(medusa.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            updates += 1
            if updates % log_every_updates == 0:
                avg = running / (log_every_updates * 1.0)
                print(f"Update {updates:05d} | KD loss: {avg:.4f}")
                running = 0.0

    print("✅ Medusa training complete. Saving weights...")
    torch.save(medusa.state_dict(), "medusa_heads_kd.pth")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B"

    print("Loading base model (BF16 + SDPA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    print("Initializing Medusa heads (K=3)...")
    medusa = MedusaHeads(hidden_size=base_model.config.hidden_size, num_heads=3)
    medusa.to(next(base_model.parameters()).device, dtype=torch.bfloat16)

    print("Loading streaming training data...")
    train_dataset = StreamingCodingDataset(tokenizer, seq_len=256)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,          # increase on H100 box if CPU allows
        pin_memory=True,
        persistent_workers=True,
    )

    # Head weights: farther heads are harder; downweight them initially
    head_weights = [1.0, 0.7, 0.5]

    train_medusa_kd(
        base_model,
        medusa,
        train_loader,
        steps=2000,
        lr=1e-3,
        accum_steps=4,
        T=2.0,
        kd_alpha=1.0,
        hard_ce_alpha=0.0,
        head_weights=head_weights,
        log_every_updates=10,
    )