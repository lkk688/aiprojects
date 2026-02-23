import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. DEPENDENCIES (Copy-Pasted from Stage 1)
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
        return draft_features + self.adapter(draft_features)

class MultiLayerDraftBlock(nn.Module):
    def __init__(self, config, window_size=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([LocalAttentionDraftLayer(config, window_size) for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

# ==========================================
# 2. THE ELASTIC ROUTER
# ==========================================
class ElasticComputeRouter(nn.Module):
    def __init__(self, hidden_size, num_lanes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_lanes, bias=False)
        )
        # Bias the network towards Lane 1 (Heavy Global) initially
        self.safety_bias = nn.Parameter(torch.tensor([-1.0, 1.0])) 

    def forward(self, hidden_state, temperature=1.0):
        logits = self.net(hidden_state) + self.safety_bias
        routed_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        return routed_probs, logits

class Stage2RoutedModel(nn.Module):
    def __init__(self, base_model, drafter_weights_path, exit_layer_idx=20):
        super().__init__()
        self.base_model = base_model
        self.exit_layer_idx = exit_layer_idx
        
        # Load Frozen Drafter
        self.drafter = MultiLayerDraftBlock(base_model.config, window_size=32, num_layers=2)
        self.drafter.load_state_dict(torch.load(drafter_weights_path))
        self.lm_head = base_model.lm_head
        
        # Initialize Trainable Router
        self.router = ElasticComputeRouter(base_model.config.hidden_size, num_lanes=2)
        
        # FREEZE Base Model AND Drafter. Only the Router trains!
        for param in self.base_model.parameters(): param.requires_grad = False
        for param in self.drafter.parameters(): param.requires_grad = False

    def forward(self, input_ids, temperature=1.0):
        # We wrap this in no_grad because the base model and drafter are frozen.
        # This saves massive amounts of VRAM on the H100!
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            true_logits = outputs.logits
            intermediate_features = outputs.hidden_states[self.exit_layer_idx]
            
            draft_features = self.drafter(intermediate_features)
            
            # THE STAGE 1 FIX: Apply RMSNorm to the features before the LM Head!
            normalized_features = self.base_model.model.norm(draft_features)
            draft_logits = self.lm_head(normalized_features)
            
        # The Router analyzes the hidden features to predict Drafter success
        # This is the ONLY part of the graph that tracks gradients.
        routed_probs, router_logits = self.router(intermediate_features, temperature=temperature)
        
        return router_logits, routed_probs, draft_logits, true_logits

# ==========================================
# 3. EXACT-MATCH STREAMING DATASET
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
            
            # Exact length match to prevent Jagged Batch errors!
            if tokens.input_ids.shape[1] == self.seq_len:
                yield {"input_ids": tokens.input_ids[0]}

# ==========================================
# 4. ORACLE TRAINING LOOP
# ==========================================
def train_router_h100(model, train_loader, steps=2000, lr=1e-3, accum_steps=2):
    device = next(model.router.parameters()).device
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr)
    
    # Weight Lane 1 higher to penalize the Router for skipping when it shouldn't
    # Cast to float32 to prevent BF16 CrossEntropy crashes
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.train()
    print(f"\n🚀 Launching Stage 2 Oracle Training (Targeting {steps} steps...)")
    
    data_iterator = iter(train_loader)
    total_loss = 0
    total_correct = 0
    total_preds = 0
    optimizer.zero_grad()
    
    for step in range(1, steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)
            
        input_ids = batch['input_ids'].to(device)
        
        router_logits, routed_probs, draft_logits, true_logits = model(input_ids)
        
        # --- CREATE DYNAMIC GROUND TRUTH (The Oracle) ---
        with torch.no_grad():
            draft_tokens = torch.argmax(draft_logits[:, :-1, :], dim=-1)
            true_tokens = torch.argmax(true_logits[:, :-1, :], dim=-1)
            
            # If Drafter is right -> Label 0 (Skip is safe)
            # If Drafter is wrong -> Label 1 (Must route to Slow Brain)
            target_lanes = torch.where(draft_tokens == true_tokens, 0, 1).view(-1).long()
            
        # Cast router logits to float32 for stable loss calculation
        valid_router_logits = router_logits[:, :-1, :].reshape(-1, 2).float()
        
        loss = criterion(valid_router_logits, target_lanes)
        loss = loss / accum_steps
        loss.backward()
        
        total_loss += loss.item() * accum_steps
        
        # Calculate running accuracy
        with torch.no_grad():
            predictions = torch.argmax(valid_router_logits, dim=-1)
            total_correct += (predictions == target_lanes).sum().item()
            total_preds += target_lanes.numel()
        
        if step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if (step // accum_steps) % 20 == 0:
                accuracy = (total_correct / total_preds) * 100
                print(f"Update Step {step // accum_steps} | Router Loss: {total_loss / accum_steps:.4f} | Routing Accuracy: {accuracy:.1f}%")
                
                # Reset rolling accuracy trackers
                total_correct = 0
                total_preds = 0
                
            total_loss = 0

    print("Stage 2 Training Complete! Saving Router weights...")
    torch.save(model.router.state_dict(), "h100_elastic_router.pth")

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
        dtype=torch.bfloat16,
        attn_implementation="sdpa" 
    )
    
    print("Initializing Stage 2 Model...")
    # Make sure to point this to the drafter you just finished training!
    model = Stage2RoutedModel(base_model, "h100_multi_layer_drafter.pth", exit_layer_idx=20)
    
    model.drafter.to(base_model.device, dtype=torch.bfloat16)
    model.router.to(base_model.device, dtype=torch.bfloat16)
    
    print("Loading Streaming Training Data...")
    train_dataset = StreamingCodingDataset(tokenizer, seq_len=256)
    
    # We can use a massive batch size because the base model and drafter are completely frozen!
    train_loader = DataLoader(train_dataset, batch_size=8)
    
    # Train the dynamic oracle (Runs incredibly fast since no heavy backprop is needed)
    train_router_h100(model, train_loader, steps=2000, lr=1e-3, accum_steps=2)