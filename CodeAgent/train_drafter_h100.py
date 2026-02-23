import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. H100-OPTIMIZED DRAFTER ARCHITECTURE
# ==========================================

class LocalAttentionDraftLayer(nn.Module):
    def __init__(self, config, window_size=32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.window_size = window_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )

    def forward(self, hidden_states):
        seq_len = hidden_states.size(1)
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Sliding Window Causal Mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device)
        mask = torch.tril(mask)
        window_mask = torch.triu(torch.ones_like(mask), diagonal=-self.window_size + 1)
        local_mask = mask & window_mask
        
        attn_bias = torch.zeros_like(local_mask, dtype=hidden_states.dtype)
        attn_bias.masked_fill_(~local_mask, float('-inf'))
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn_weights = attn_weights + attn_bias
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = self.o_proj(attn_output)
        
        draft_features = hidden_states + attn_output
        final_features = draft_features + self.adapter(draft_features)
        
        return final_features

class MultiLayerDraftBlock(nn.Module):
    """A deeper drafter for 7B/8B models"""
    def __init__(self, config, window_size=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalAttentionDraftLayer(config, window_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Stage1SpeculativeModel(nn.Module):
    def __init__(self, base_model, exit_layer_idx=20, window_size=32, num_draft_layers=2):
        super().__init__()
        self.base_model = base_model 
        self.exit_layer_idx = exit_layer_idx
        
        # 2-Layer Drafter to handle complex 7B representations
        self.drafter = MultiLayerDraftBlock(base_model.config, window_size=window_size, num_layers=num_draft_layers)
        self.lm_head = base_model.lm_head 
        
        # Freeze the 7B base model completely
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            
        intermediate_features = outputs.hidden_states[self.exit_layer_idx]
        draft_features = self.drafter(intermediate_features)
        
        # THE FIX: Apply the base model's final RMSNorm to stabilize the features!
        normalized_features = self.base_model.model.norm(draft_features)
        
        draft_logits = self.lm_head(normalized_features)
        #draft_logits = self.lm_head(draft_features)
        
        return draft_logits

# ==========================================
# 2. STREAMING DATA PIPELINE
# ==========================================

class StreamingCodingDataset(IterableDataset):
    """Streams data infinitely to prevent CPU RAM crashes on massive datasets."""
    def __init__(self, tokenizer, dataset_name="bigcode/the-stack-smol", split="train", seq_len=256, data_dir="data/python"):
        self.dataset = load_dataset(dataset_name, data_dir=data_dir, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        for item in self.dataset:
            code_text = item.get('content') or item.get('code') or item.get('text')
            if not code_text:
                continue
                
            tokens = self.tokenizer(code_text, truncation=True, max_length=self.seq_len, return_tensors="pt")
            
            # # Yield only if it has a decent amount of tokens to learn from
            # if tokens.input_ids.shape[1] >= self.seq_len // 2:
            #     yield {"input_ids": tokens.input_ids[0]}
            # THE FIX: Only yield the tensor if it is exactly `seq_len` long
            if tokens.input_ids.shape[1] == self.seq_len:
                yield {"input_ids": tokens.input_ids[0]}

# ==========================================
# 3. TRAINING LOOP (WITH ACCUMULATION)
# ==========================================

def train_drafter_h100(model, train_loader, steps=2000, lr=5e-4, accum_steps=4):
    device = next(model.drafter.parameters()).device
    optimizer = torch.optim.AdamW(model.drafter.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"\n🚀 Launching H100 Training (Targeting {steps} steps with accumulation...)")
    
    total_loss = 0
    optimizer.zero_grad()
    
    data_iterator = iter(train_loader)
    
    for step in range(1, steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)
            
        input_ids = batch['input_ids'].to(device)
        
        # Mixed Precision Forward Pass (Handled automatically by bfloat16 base model)
        draft_logits = model(input_ids)
        
        # shift_logits = draft_logits[:, :-1, :].contiguous()
        # shift_labels = input_ids[:, 1:].contiguous()
        # THE FIX: Cast logits to .float() to prevent Cross-Entropy overflow
        shift_logits = draft_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        
        if step % accum_steps == 0:
            # THE FIX: Clip gradients to max_norm 1.0 to prevent any future spikes
            torch.nn.utils.clip_grad_norm_(model.drafter.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if (step // accum_steps) % 20 == 0:
                print(f"Update Step {step // accum_steps} | Drafter Loss: {total_loss / accum_steps:.4f}")
            total_loss = 0

    print("Training Complete! Saving weights...")
    torch.save(model.drafter.state_dict(), "h100_multi_layer_drafter.pth")

# ==========================================
# 4. EVALUATION BENCHMARK
# ==========================================

def run_h100_evaluation(stage1_model, tokenizer, test_prompts, max_new_tokens=40, K=3):
    print("\n" + "="*50)
    print("🚀 H100 SPECULATIVE DECODING BENCHMARK")
    print("="*50)
    
    stage1_model.eval()
    
    for i, prompt in enumerate(test_prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(stage1_model.base_model.device)
        
        start_time = time.time()
        tokens_generated = 0
        forward_steps = 0
        stats = {"drafts_generated": 0, "drafts_accepted": 0}
        
        with torch.no_grad():
            while tokens_generated < max_new_tokens:
                draft_tokens = []
                current_input = input_ids
                
                # --- DRAFTING ---
                for _ in range(K):
                    outputs = stage1_model.base_model(current_input, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[stage1_model.exit_layer_idx]
                    
                    # Pass full sequence through the multi-layer SWA drafter
                    draft_features = stage1_model.drafter(hidden_states)
                    
                    # Project only the final token to vocab
                    next_draft_token = torch.argmax(stage1_model.lm_head(draft_features[:, -1:, :]), dim=-1)
                    
                    draft_tokens.append(next_draft_token)
                    current_input = torch.cat([current_input, next_draft_token], dim=1)
                    stats["drafts_generated"] += 1
                
                draft_tensor = torch.cat(draft_tokens, dim=1)
                spec_input_ids = torch.cat([input_ids, draft_tensor], dim=1)
                
                # --- VERIFICATION ---
                slow_outputs = stage1_model.base_model(spec_input_ids)
                forward_steps += 1 
                slow_logits = slow_outputs.logits
                seq_len = input_ids.shape[1]
                
                accepted_list = []
                hit_eos = False
                
                for i in range(K):
                    true_token = torch.argmax(slow_logits[:, seq_len - 1 + i, :], dim=-1).unsqueeze(1)
                    draft_tok = draft_tensor[:, i].unsqueeze(1)
                    
                    if true_token.item() == draft_tok.item():
                        accepted_list.append(draft_tok)
                        stats["drafts_accepted"] += 1
                        if draft_tok.item() == tokenizer.eos_token_id:
                            hit_eos = True
                            break
                    else:
                        accepted_list.append(true_token)
                        if true_token.item() == tokenizer.eos_token_id:
                            hit_eos = True
                        break
                
                if len(accepted_list) == K and not hit_eos:
                    bonus_token = torch.argmax(slow_logits[:, -1:, :], dim=-1)
                    accepted_list.append(bonus_token)
                    if bonus_token.item() == tokenizer.eos_token_id:
                        hit_eos = True
                        
                accepted_tensor = torch.cat(accepted_list, dim=1)
                
                # Truncate overshoot
                remaining = max_new_tokens - tokens_generated
                if accepted_tensor.shape[1] > remaining:
                    accepted_tensor = accepted_tensor[:, :remaining]
                    
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
                tokens_generated += accepted_tensor.shape[1]
                
                if hit_eos or input_ids[0, -1].item() == tokenizer.eos_token_id:
                    break
                    
        wall_time = time.time() - start_time
        
        acc_rate = (stats['drafts_accepted'] / max(1, stats['drafts_generated'])) * 100
        step_speedup = tokens_generated / max(1, forward_steps)
        
        print(f"\n[Prompt {i+1}]: {prompt.strip()}")
        print(f"  🎯 Acceptance Rate: {acc_rate:.1f}% ({stats['drafts_accepted']}/{stats['drafts_generated']})")
        print(f"  ⚡ Model Steps: {forward_steps} (Tokens generated: {tokens_generated})")
        print(f"  🔥 Theoretical Speedup: {step_speedup:.2f}x fewer forward passes!")
        print("-" * 40)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # We step up to the 7B class model for serious research validation
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B" 
    
    print("Loading 7B Model and Tokenizer (BF16 & FlashAttention)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # We embrace bfloat16 and FlashAttention-2 to saturate the H100 Tensor Cores
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2" ,
        attn_implementation="sdpa"
    )
    
    # Extract at Layer 20 (Deep representations) and use 2 SWA Layers
    model = Stage1SpeculativeModel(base_model, exit_layer_idx=20, window_size=32, num_draft_layers=2)
    model.drafter.to(base_model.device, dtype=torch.bfloat16)
    
    print("Initializing Streaming Dataset...")
    # Batch size 8 is a good start. If you OOM on the 80GB VRAM, drop to 4. 
    train_dataset = StreamingCodingDataset(tokenizer, seq_len=256)
    train_loader = DataLoader(train_dataset, batch_size=8)
    
    # Run 4,000 forward steps (1,000 actual optimizer updates)
    train_drafter_h100(model, train_loader, steps=4000, lr=3e-4, accum_steps=4)
    
    # Final Sanity Check Benchmark
    test_prompts = [
        "def quicksort(arr):",
        "class TransformerLayer(nn.Module):\n    def __init__(self, d_model):",
    ]
    
    run_h100_evaluation(model, tokenizer, test_prompts, max_new_tokens=40, K=3)