import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. THE MEDUSA ARCHITECTURE
# ==========================================
class MedusaResBlock(nn.Module):
    """A tiny residual block that learns to look ahead."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class MedusaHeads(nn.Module):
    def __init__(self, hidden_size, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        # We create a sequence of ResBlocks. 
        # Output of Block 1 -> Head 1. Output of Block 2 -> Head 2.
        self.blocks = nn.ModuleList([MedusaResBlock(hidden_size) for _ in range(num_heads)])

    def forward(self, hidden_states):
        features = []
        x = hidden_states
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

# ==========================================
# 2. EXACT-MATCH STREAMING DATASET
# ==========================================
class StreamingCodingDataset(IterableDataset):
    def __init__(self, tokenizer, dataset_name="bigcode/the-stack-smol", split="train", seq_len=256, data_dir="data/python"):
        self.dataset = load_dataset(dataset_name, data_dir=data_dir, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        for item in self.dataset:
            code_text = item.get('content') or item.get('code') or item.get('text')
            if not code_text: continue
                
            tokens = self.tokenizer(code_text, truncation=True, max_length=self.seq_len, return_tensors="pt")
            if tokens.input_ids.shape[1] == self.seq_len:
                yield {"input_ids": tokens.input_ids[0]}

# ==========================================
# 3. KNOWLEDGE DISTILLATION TRAINING LOOP
# ==========================================
def train_medusa_kd_h100(base_model, medusa, train_loader, steps=2000, lr=1e-3, accum_steps=4):
    device = base_model.device
    optimizer = torch.optim.AdamW(medusa.parameters(), lr=lr)
    
    medusa.train()
    print(f"\n🚀 Launching Medusa KD Training (Targeting {steps} steps...)")
    
    data_iterator = iter(train_loader)
    total_loss = 0
    
    for step in range(1, steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)
            
        input_ids = batch['input_ids'].to(device)
        
        # 1. Get the "Teacher" outputs from the frozen base model
        with torch.no_grad():
            outputs = base_model(input_ids, output_hidden_states=True)
            # Medusa attaches to the absolute final hidden state
            final_hidden_states = outputs.hidden_states[-1] 
            teacher_logits = outputs.logits.float() # Cast to float32 for stable KD math
            
        # 2. Get the "Student" features from Medusa
        medusa_features = medusa(final_hidden_states)
        
        loss = 0
        # 3. Calculate Distillation Loss for each head
        for k, features in enumerate(medusa_features):
            # Project features to vocab using the frozen base model's lm_head
            student_logits = base_model.lm_head(features).float()
            
            # The Alignment Shift: 
            # Head k (0-indexed) predicts the token at t + k + 1
            shift = k + 1
            
            s_logits = student_logits[:, :-shift, :].contiguous()
            t_logits = teacher_logits[:, shift:, :].contiguous()
            
            # KNOWLEDGE DISTILLATION: Cross Entropy with Soft Targets
            # We force the Medusa head to match the Base Model's probability distribution
            teacher_probs = F.softmax(t_logits, dim=-1)
            
            head_loss = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)), 
                teacher_probs.view(-1, teacher_probs.size(-1))
            )
            loss += head_loss
            
        # Scale for accumulation
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(medusa.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if (step // accum_steps) % 10 == 0:
                print(f"Update Step {step // accum_steps} | Medusa KD Loss: {total_loss / accum_steps:.4f}")
            total_loss = 0

    print("Medusa Training Complete! Saving weights...")
    torch.save(medusa.state_dict(), "h100_medusa_heads.pth")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B" 
    
    print("Loading 7B Base Model (BF16 & SDPA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" 
    )
    # Freeze the base model
    for param in base_model.parameters(): param.requires_grad = False
    base_model.eval()
    
    print("Initializing Medusa Heads (K=3)...")
    # Qwen 7B hidden size is 3584
    medusa = MedusaHeads(hidden_size=base_model.config.hidden_size, num_heads=3)
    medusa.to(base_model.device, dtype=torch.bfloat16)
    
    print("Loading Streaming Training Data...")
    train_dataset = StreamingCodingDataset(tokenizer, seq_len=256)
    train_loader = DataLoader(train_dataset, batch_size=8)
    
    train_medusa_kd_h100(base_model, medusa, train_loader, steps=2000, lr=1e-3, accum_steps=4)