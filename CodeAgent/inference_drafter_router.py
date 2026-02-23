import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
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

class ElasticComputeRouter(nn.Module):
    def __init__(self, hidden_size, num_lanes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_lanes, bias=False)
        )
        self.safety_bias = nn.Parameter(torch.tensor([-1.0, 1.0])) 

    def forward(self, hidden_state, temperature=1.0):
        logits = self.net(hidden_state) + self.safety_bias
        routed_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        return routed_probs, logits

class ElasticSpeculativeEngine(nn.Module):
    def __init__(self, base_model, drafter_path, router_path, exit_layer_idx=20):
        super().__init__()
        self.base_model = base_model
        self.exit_layer_idx = exit_layer_idx
        
        self.drafter = MultiLayerDraftBlock(base_model.config, window_size=32, num_layers=2)
        self.drafter.load_state_dict(torch.load(drafter_path, weights_only=True))
        
        self.router = ElasticComputeRouter(base_model.config.hidden_size, num_lanes=2)
        self.router.load_state_dict(torch.load(router_path, weights_only=True))
        
        self.lm_head = base_model.lm_head
        self.base_model.eval()
        self.drafter.eval()
        self.router.eval()
        
        for param in self.parameters():
            param.requires_grad = False

# ==========================================
# 2. KV CACHE & GENERATION LOOP
# ==========================================
# def rollback_kv_cache(cache, keep_len):
#     """Physically slices the KV Cache memory to erase rejected drafted tokens."""
#     if isinstance(cache, DynamicCache):
#         new_cache = DynamicCache()
#         new_cache.key_cache = [k[:, :, :keep_len, :] for k in cache.key_cache]
#         new_cache.value_cache = [v[:, :, :keep_len, :] for v in cache.value_cache]
#         new_cache._seen_tokens = keep_len
#         return new_cache
#     else: 
#         return tuple(tuple(t[:, :, :keep_len, :] for t in layer) for layer in cache)

def rollback_kv_cache(cache, keep_len):
    """Physically slices the KV Cache memory to erase rejected drafted tokens."""
    
    # 1. The New API (Transformers v4.44+)
    # Hugging Face now natively supports in-place cache cropping!
    if hasattr(cache, "crop"):
        cache.crop(keep_len)
        return cache
        
    # 2. The Legacy API (Transformers v4.38 to v4.43)
    elif hasattr(cache, "key_cache"):
        new_cache = DynamicCache()
        new_cache.key_cache = [k[:, :, :keep_len, :] for k in cache.key_cache]
        new_cache.value_cache = [v[:, :, :keep_len, :] for v in cache.value_cache]
        new_cache._seen_tokens = keep_len
        return new_cache
        
    # 3. The Ancient Fallback (Standard PyTorch Tuple-of-Tuples)
    else: 
        return tuple(tuple(t[:, :, :keep_len, :] for t in layer) for layer in cache)

def generate_baseline_standard(base_model, tokenizer, prompt, max_new_tokens=60):
    """The standard Autoregressive Baseline using perfectly optimized KV-Caching."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(base_model.device)
    
    start_time = time.time()
    tokens_generated = 0
    
    with torch.no_grad():
        # --- PREFILL ---
        outputs = base_model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        tokens_generated += 1
        
        # --- DECODING LOOP ---
        while tokens_generated < max_new_tokens:
            outputs = base_model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            tokens_generated += 1
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    wall_time = time.time() - start_time
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, tokens_generated, wall_time

def generate_elastic_fast_v1(engine, tokenizer, prompt, max_new_tokens=60, K=3):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.base_model.device)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    stats = {"trusted_skips": 0, "heavy_routes": 0, "drafts_generated": 0, "drafts_accepted": 0}
    
    with torch.no_grad():
        # --- PREFILL ---
        outputs = engine.base_model(input_ids, use_cache=True, output_hidden_states=True)
        past_kv = outputs.past_key_values
        hidden_states = outputs.hidden_states[engine.exit_layer_idx]
        next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        tokens_generated += 1
        forward_steps += 1
        
        while tokens_generated < max_new_tokens:
            routed_probs, _ = engine.router(hidden_states[:, -1:, :])
            lane_choice = torch.argmax(routed_probs, dim=-1).item()
            
            verified_len = input_ids.shape[1]
            
            if lane_choice == 0:
                # --- PATH A: DRAFT & VERIFY ---
                stats["trusted_skips"] += 1
                draft_tokens = []
                current_token = next_token
                draft_kv = past_kv 
                
                # Draft Loop
                for _ in range(K):
                    draft_out = engine.base_model(
                        current_token, past_key_values=draft_kv, use_cache=True, output_hidden_states=True
                    )
                    draft_kv = draft_out.past_key_values
                    draft_hidden = draft_out.hidden_states[engine.exit_layer_idx]
                    
                    d_features = engine.drafter(draft_hidden)
                    # The crucial norm fix applied for inference
                    norm_features = engine.base_model.model.norm(d_features[:, -1:, :])
                    d_token = torch.argmax(engine.lm_head(norm_features), dim=-1)
                    
                    draft_tokens.append(d_token)
                    current_token = d_token
                    stats["drafts_generated"] += 1
                    
                draft_tensor = torch.cat(draft_tokens, dim=1)
                
                # Rollback cache for parallel verification
                past_kv = rollback_kv_cache(draft_kv, verified_len - 1)
                
                # Verification
                verification_input = torch.cat([input_ids[:, -1:], draft_tensor], dim=1)
                slow_outputs = engine.base_model(
                    verification_input, past_key_values=past_kv, use_cache=True
                )
                forward_steps += 1
                slow_logits = slow_outputs.logits 
                
                # Cascading Reject
                accepted_list = []
                hit_eos = False
                for i in range(K):
                    true_token = torch.argmax(slow_logits[:, i, :], dim=-1).unsqueeze(1)
                    draft_tok = draft_tensor[:, i].unsqueeze(1)
                    if true_token.item() == draft_tok.item():
                        accepted_list.append(draft_tok)
                        stats["drafts_accepted"] += 1
                        if draft_tok.item() == tokenizer.eos_token_id: hit_eos = True; break
                    else:
                        accepted_list.append(true_token)
                        if true_token.item() == tokenizer.eos_token_id: hit_eos = True; break
                        
                        # THE FIX: We must halt the sequence evaluation on rejection!
                        break
                
                if len(accepted_list) == K and not hit_eos:
                    bonus_token = torch.argmax(slow_logits[:, K, :], dim=-1).unsqueeze(1)
                    accepted_list.append(bonus_token)
                    if bonus_token.item() == tokenizer.eos_token_id: hit_eos = True
                        
                accepted_tensor = torch.cat(accepted_list, dim=1)
                
                # Apply tokens and Rollback rejected cache
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
                tokens_generated += accepted_tensor.shape[1]
                past_kv = rollback_kv_cache(slow_outputs.past_key_values, input_ids.shape[1] - 1)
                
                # Setup next loop state
                last_accepted = accepted_tensor[:, -1:]
                setup_out = engine.base_model(
                    last_accepted, past_key_values=past_kv, use_cache=True, output_hidden_states=True
                )
                past_kv = setup_out.past_key_values
                hidden_states = setup_out.hidden_states[engine.exit_layer_idx]
                next_token = torch.argmax(setup_out.logits[:, -1:, :], dim=-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                tokens_generated += 1
                
                if hit_eos or input_ids[0, -1].item() == tokenizer.eos_token_id: break

            else: 
                # --- PATH B: HEAVY GLOBAL (Router Skip) ---
                stats["heavy_routes"] += 1
                forward_steps += 1
                
                outputs = engine.base_model(
                    next_token, past_key_values=past_kv, use_cache=True, output_hidden_states=True
                )
                past_kv = outputs.past_key_values
                hidden_states = outputs.hidden_states[engine.exit_layer_idx]
                next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                tokens_generated += 1
                
                if next_token.item() == tokenizer.eos_token_id: break
                    
    wall_time = time.time() - start_time
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, tokens_generated, wall_time, forward_steps, stats

import copy

def clone_kv_cache(cache):
    """Deepcopies the KV Cache safely regardless of the Hugging Face API version."""
    return copy.deepcopy(cache)

def generate_elastic_fast(engine, tokenizer, prompt, max_new_tokens=60, K=3):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.base_model.device)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    stats = {"trusted_skips": 0, "heavy_routes": 0, "drafts_generated": 0, "drafts_accepted": 0}
    
    # Extract the layer lists for the surgical swap
    original_layers = engine.base_model.model.layers
    draft_layers = original_layers[:engine.exit_layer_idx]
    
    with torch.no_grad():
        # --- 1. PREFILL ---
        outputs = engine.base_model(input_ids, use_cache=True, output_hidden_states=True)
        past_kv = outputs.past_key_values
        
        # We need the hidden state of the last token in the prompt for the router
        hidden_states = outputs.hidden_states[engine.exit_layer_idx]
        next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        tokens_generated += 1
        forward_steps += 1
        
        # --- 2. DECODING LOOP ---
        while tokens_generated < max_new_tokens:
            # Route based on the hidden state of the token we just verified
            routed_probs, _ = engine.router(hidden_states[:, -1:, :])
            lane_choice = torch.argmax(routed_probs, dim=-1).item()
            
            if lane_choice == 0:
                # --- PATH A: TRUE EARLY EXIT DRAFTING ---
                stats["trusted_skips"] += 1
                draft_tokens = []
                current_token = next_token
                
                # Deepcopy the clean cache so the Drafter can safely play with it
                draft_kv = clone_kv_cache(past_kv)
                
                # 🔥 TRUNCATE THE BASE MODEL 🔥
                engine.base_model.model.layers = draft_layers
                
                for _ in range(K):
                    # Call the inner model (runs only 20 layers!)
                    draft_out = engine.base_model.model(
                        input_ids=current_token, 
                        past_key_values=draft_kv, 
                        use_cache=True,
                        return_dict=True
                    )
                    draft_kv = draft_out.past_key_values
                    draft_hidden = draft_out.last_hidden_state
                    
                    # Parasitic Drafter Output
                    d_features = engine.drafter(draft_hidden)
                    norm_features = engine.base_model.model.norm(d_features[:, -1:, :])
                    d_token = torch.argmax(engine.lm_head(norm_features), dim=-1)
                    
                    draft_tokens.append(d_token)
                    current_token = d_token
                    stats["drafts_generated"] += 1
                    
                draft_tensor = torch.cat(draft_tokens, dim=1)
                
                # 🔥 RESTORE THE BASE MODEL FOR VERIFICATION 🔥
                engine.base_model.model.layers = original_layers
                
                # We verify the Sequence [next_token, draft1, draft2, draft3]
                # Using the original, untouched past_kv!
                verification_input = torch.cat([next_token, draft_tensor], dim=1)
                slow_outputs = engine.base_model(
                    input_ids=verification_input, 
                    past_key_values=past_kv, 
                    use_cache=True,
                    output_hidden_states=True
                )
                forward_steps += 1
                slow_logits = slow_outputs.logits 
                
                # Cascading Rejection
                accepted_list = []
                hit_eos = False
                for i in range(K):
                    true_token = torch.argmax(slow_logits[:, i, :], dim=-1).unsqueeze(1)
                    draft_tok = draft_tensor[:, i].unsqueeze(1)
                    if true_token.item() == draft_tok.item():
                        accepted_list.append(draft_tok)
                        stats["drafts_accepted"] += 1
                        if draft_tok.item() == tokenizer.eos_token_id: hit_eos = True; break
                    else:
                        accepted_list.append(true_token)
                        if true_token.item() == tokenizer.eos_token_id: hit_eos = True; break
                        break # Stop evaluating on the first failure
                
                if len(accepted_list) == K and not hit_eos:
                    bonus_token = torch.argmax(slow_logits[:, K, :], dim=-1).unsqueeze(1)
                    accepted_list.append(bonus_token)
                    if bonus_token.item() == tokenizer.eos_token_id: hit_eos = True
                        
                accepted_tensor = torch.cat(accepted_list, dim=1)
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
                tokens_generated += accepted_tensor.shape[1]
                
                # Setup Memory for Next Loop
                past_kv = slow_outputs.past_key_values
                past_kv.crop(input_ids.shape[1] - 1) 
                
                next_token = accepted_tensor[:, -1:]
                last_accepted_idx = len(accepted_list) - 1
                hidden_states = slow_outputs.hidden_states[engine.exit_layer_idx][:, last_accepted_idx:last_accepted_idx+1, :]
                
                if hit_eos or input_ids[0, -1].item() == tokenizer.eos_token_id: break

            else: 
                # --- PATH B: HEAVY GLOBAL (Router Skip) ---
                stats["heavy_routes"] += 1
                forward_steps += 1
                
                outputs = engine.base_model(
                    next_token, past_key_values=past_kv, use_cache=True, output_hidden_states=True
                )
                past_kv = outputs.past_key_values
                hidden_states = outputs.hidden_states[engine.exit_layer_idx]
                next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                tokens_generated += 1
                
                if next_token.item() == tokenizer.eos_token_id: break
                    
    # Failsafe: Always restore layers
    engine.base_model.model.layers = original_layers
    
    wall_time = time.time() - start_time
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, tokens_generated, wall_time, forward_steps, stats

# ==========================================
# 3. OPTIMIZED GENERATION (Surgical Forward)
# ==========================================

def _clone_cache_partial(past_kv, num_layers):
    """Clone only the first num_layers of a DynamicCache using fast tensor .clone().

    Avoids copy.deepcopy (Python-level recursion) and only copies the layers
    needed for drafting (e.g. 20 out of 28), saving both time and memory.
    """
    draft_kv = DynamicCache()
    # Override lazy-init so layers are pre-populated
    draft_kv.layers = []
    draft_kv.layer_class_to_replicate = None  # Disable lazy layer creation
    for i in range(num_layers):
        src_layer = past_kv.layers[i]
        from transformers.cache_utils import DynamicLayer
        new_layer = DynamicLayer()
        new_layer.keys = src_layer.keys.clone()
        new_layer.values = src_layer.values.clone()
        new_layer.dtype = src_layer.dtype
        new_layer.device = src_layer.device
        new_layer.is_initialized = True
        draft_kv.layers.append(new_layer)
    return draft_kv


def _surgical_forward_partial(model, input_ids, past_kv, exit_layer_idx):
    """Run only layers [0, exit_layer_idx) of the base model.

    Bypasses HuggingFace's full forward machinery (mask creation, output
    packaging, hidden_states collection) for maximum speed.
    Returns the raw hidden state at the exit layer.
    """
    inner = model.model  # Qwen2Model

    # Embed
    hidden_states = inner.embed_tokens(input_ids)

    # Compute position info
    past_seen = past_kv.get_seq_length() if past_kv is not None else 0
    seq_len = hidden_states.shape[1]
    cache_position = torch.arange(
        past_seen, past_seen + seq_len, device=hidden_states.device
    )
    position_ids = cache_position.unsqueeze(0)

    # Causal mask: for single-token decoding with SDPA, None works
    # For multi-token sequences, build a minimal causal mask
    if seq_len > 1:
        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=inner.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_kv,
        )
    else:
        causal_mask = None

    # Rotary embeddings (computed once, shared across all layers)
    position_embeddings = inner.rotary_emb(hidden_states, position_ids)

    # Run only the first exit_layer_idx layers
    for i in range(exit_layer_idx):
        hidden_states = inner.layers[i](
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    return hidden_states


def _surgical_forward_full(model, input_ids, past_kv, exit_layer_idx):
    """Run ALL layers of the base model, capturing the hidden state at exit_layer_idx.

    Replaces the pattern of calling model(..., output_hidden_states=True) and
    then indexing into the huge hidden_states list. Only stores 1 intermediate
    tensor instead of num_layers+1.
    Returns (logits, exit_hidden_state, past_kv).
    """
    inner = model.model  # Qwen2Model

    # Embed
    hidden_states = inner.embed_tokens(input_ids)

    # Compute position info
    if past_kv is None:
        past_kv = DynamicCache(config=inner.config)
    past_seen = past_kv.get_seq_length()
    seq_len = hidden_states.shape[1]
    cache_position = torch.arange(
        past_seen, past_seen + seq_len, device=hidden_states.device
    )
    position_ids = cache_position.unsqueeze(0)

    # Causal mask
    if seq_len > 1:
        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=inner.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_kv,
        )
    else:
        causal_mask = None

    # Rotary embeddings
    position_embeddings = inner.rotary_emb(hidden_states, position_ids)

    # Run all layers, capturing exit_layer_idx
    exit_hidden = None
    num_layers = len(inner.layers)
    for i in range(num_layers):
        hidden_states = inner.layers[i](
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if i == exit_layer_idx - 1:
            exit_hidden = hidden_states

    # Final norm + lm_head
    hidden_states = inner.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return logits, exit_hidden, past_kv


def _router_decide(router, hidden_state):
    """Deterministic routing decision (no gumbel_softmax noise at inference)."""
    logits = router.net(hidden_state) + router.safety_bias
    # Lane 0 = draft (trusted skip), Lane 1 = heavy (full model)
    return logits[..., 0] > logits[..., 1]


def generate_elastic_optimized(engine, tokenizer, prompt, max_new_tokens=60, K=3):
    """Optimized elastic speculative decoding — zero-cost drafting.

    The critical insight: running 20 transformer layers per draft token makes
    drafting nearly as expensive as the full model. Instead, use the drafter
    head as a standalone predictor with NO transformer layers:
      - 1st draft: drafter(real layer-20 hidden) -> norm -> lm_head
      - 2nd draft: drafter(prev drafter output) -> norm -> lm_head
      - 3rd draft: drafter(prev drafter output) -> norm -> lm_head

    Draft cost drops from 3x20=60 layer-forwards to ~0.
    Verification of K+1 tokens in parallel costs barely more than 1 token.
    Even with lower draft accuracy, net throughput improves.
    """
    device = engine.base_model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    max_len = prompt_len + max_new_tokens + K + 2
    token_buf = torch.zeros(1, max_len, dtype=torch.long, device=device)
    token_buf[0, :prompt_len] = input_ids[0]
    buf_len = prompt_len

    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    stats = {"trusted_skips": 0, "heavy_routes": 0, "drafts_generated": 0, "drafts_accepted": 0}

    exit_idx = engine.exit_layer_idx
    eos_id = tokenizer.eos_token_id
    norm_fn = engine.base_model.model.norm
    lm_head = engine.lm_head
    drafter = engine.drafter

    with torch.inference_mode():
        # --- 1. PREFILL ---
        logits, exit_hidden, past_kv = _surgical_forward_full(
            engine.base_model, token_buf[:, :buf_len], None, exit_idx
        )
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        token_buf[0, buf_len] = next_token_id
        buf_len += 1
        tokens_generated += 1
        forward_steps += 1

        router_hidden = exit_hidden[:, -1:, :]

        # --- 2. DECODING LOOP ---
        while tokens_generated < max_new_tokens:
            use_draft = _router_decide(engine.router, router_hidden).item()

            if use_draft:
                # --- PATH A: ZERO-COST DRAFTING ---
                stats["trusted_skips"] += 1

                # Draft K tokens using ONLY the drafter head — no transformer layers!
                # No KV cache clone needed.
                draft_token_ids = torch.zeros(1, K, dtype=torch.long, device=device)
                d_hidden = router_hidden  # Real layer-20 hidden state

                for k_idx in range(K):
                    d_features = drafter(d_hidden)
                    d_logits = lm_head(norm_fn(d_features[:, -1:, :]))
                    d_token = torch.argmax(d_logits, dim=-1)
                    draft_token_ids[0, k_idx] = d_token[0, 0]
                    # Feed drafter output back as next input
                    d_hidden = d_features
                    stats["drafts_generated"] += 1

                # Verify [last_verified_token, draft1, ..., draftK] through full model
                verification_input = torch.cat(
                    [token_buf[:, buf_len - 1:buf_len], draft_token_ids], dim=1
                )
                verify_logits, verify_exit_hidden, past_kv = _surgical_forward_full(
                    engine.base_model, verification_input, past_kv, exit_idx
                )
                forward_steps += 1

                # Cascading rejection
                accepted_count = 0
                hit_eos = False
                for i in range(K):
                    true_token = torch.argmax(verify_logits[:, i, :], dim=-1)
                    draft_tok = draft_token_ids[0, i]
                    if true_token.item() == draft_tok.item():
                        token_buf[0, buf_len] = draft_tok
                        buf_len += 1
                        accepted_count += 1
                        stats["drafts_accepted"] += 1
                        if draft_tok.item() == eos_id:
                            hit_eos = True
                            break
                    else:
                        token_buf[0, buf_len] = true_token
                        buf_len += 1
                        accepted_count += 1
                        if true_token.item() == eos_id:
                            hit_eos = True
                        break

                # Bonus token if all K drafts accepted
                if accepted_count == K and not hit_eos:
                    bonus = torch.argmax(verify_logits[:, K, :], dim=-1)
                    token_buf[0, buf_len] = bonus
                    buf_len += 1
                    accepted_count += 1
                    if bonus.item() == eos_id:
                        hit_eos = True

                tokens_generated += accepted_count

                # Crop KV cache to match accepted tokens
                past_kv.crop(buf_len - 1)

                # Router hidden from verification output
                last_idx = accepted_count - 1
                router_hidden = verify_exit_hidden[:, last_idx:last_idx + 1, :]

                if hit_eos:
                    break

            else:
                # --- PATH B: HEAVY FULL-MODEL STEP ---
                stats["heavy_routes"] += 1
                forward_steps += 1

                current_token = token_buf[:, buf_len - 1:buf_len]
                logits, exit_hidden, past_kv = _surgical_forward_full(
                    engine.base_model, current_token, past_kv, exit_idx
                )
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                token_buf[0, buf_len] = next_token_id
                buf_len += 1
                tokens_generated += 1

                router_hidden = exit_hidden[:, -1:, :]

                if next_token_id.item() == eos_id:
                    break

    wall_time = time.time() - start_time
    text = tokenizer.decode(token_buf[0, :buf_len], skip_special_tokens=True)
    return text, tokens_generated, wall_time, forward_steps, stats


# ==========================================
# 4. BENCHMARK
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B"

    print("Loading Base Model (BF16 & SDPA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    print("Loading Unified Elastic Engine...")
    engine = ElasticSpeculativeEngine(
        base_model,
        drafter_path="h100_multi_layer_drafter.pth",
        router_path="h100_elastic_router.pth",
        exit_layer_idx=20
    )
    engine.drafter.to(base_model.device, dtype=torch.bfloat16)
    engine.router.to(base_model.device, dtype=torch.bfloat16)

    # Compile drafter and router for reduced kernel launch overhead
    # Use default mode (not "reduce-overhead") to avoid CUDA graph conflicts
    # with dynamic control flow in the generation loop
    try:
        engine.drafter = torch.compile(engine.drafter)
        engine.router.net = torch.compile(engine.router.net)
        print("torch.compile applied to drafter & router")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    # --- WARMUP ---
    print("\nWarming up GPU caches for all engines...")
    _, _, _ = generate_baseline_standard(base_model, tokenizer, "def add(a, b):", max_new_tokens=5)
    _, _, _, _, _ = generate_elastic_fast(engine, tokenizer, "def add(a, b):", max_new_tokens=5, K=3)
    _, _, _, _, _ = generate_elastic_optimized(engine, tokenizer, "def add(a, b):", max_new_tokens=5, K=3)
    # Extra warmup for torch.compile graph capture
    for _ in range(2):
        _, _, _, _, _ = generate_elastic_optimized(engine, tokenizer, "def add(a, b):", max_new_tokens=10, K=3)

    prompts = [
        "def quicksort(arr):",
        "import torch\nimport torch.nn as nn\nclass MultiHeadAttention(nn.Module):"
    ]

    MAX_TOKENS = 120

    print("\n" + "="*70)
    print(" BASELINE vs ORIGINAL ELASTIC vs OPTIMIZED ELASTIC")
    print("="*70)

    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}]: {prompt.strip()}")

        # 1. Standard Baseline
        base_text, base_tokens, base_time = generate_baseline_standard(
            base_model, tokenizer, prompt, max_new_tokens=MAX_TOKENS
        )
        base_tps = base_tokens / base_time

        # 2. Original Elastic Engine
        orig_text, orig_tokens, orig_time, orig_steps, orig_stats = generate_elastic_fast(
            engine, tokenizer, prompt, max_new_tokens=MAX_TOKENS, K=3
        )
        orig_tps = orig_tokens / orig_time

        # 3. Optimized Elastic Engine
        opt_text, opt_tokens, opt_time, opt_steps, opt_stats = generate_elastic_optimized(
            engine, tokenizer, prompt, max_new_tokens=MAX_TOKENS, K=3
        )
        opt_tps = opt_tokens / opt_time

        print(f"\n--- STANDARD BASELINE ---")
        print(f"  Wall Time: {base_time:.3f}s | Speed: {base_tps:.1f} tok/s | Tokens: {base_tokens}")

        print(f"\n--- ORIGINAL ELASTIC ---")
        print(f"  Wall Time: {orig_time:.3f}s | Speed: {orig_tps:.1f} tok/s | Tokens: {orig_tokens}")
        print(f"  Steps: {orig_steps} | Draft: {orig_stats['trusted_skips']}x | Heavy: {orig_stats['heavy_routes']}x", end="")
        if orig_stats['drafts_generated'] > 0:
            print(f" | Accuracy: {orig_stats['drafts_accepted']/orig_stats['drafts_generated']*100:.1f}%")
        else:
            print()
        print(f"  Speedup vs Baseline: {orig_tps/base_tps:.2f}x")

        print(f"\n--- OPTIMIZED ELASTIC ---")
        print(f"  Wall Time: {opt_time:.3f}s | Speed: {opt_tps:.1f} tok/s | Tokens: {opt_tokens}")
        print(f"  Steps: {opt_steps} | Draft: {opt_stats['trusted_skips']}x | Heavy: {opt_stats['heavy_routes']}x", end="")
        if opt_stats['drafts_generated'] > 0:
            print(f" | Accuracy: {opt_stats['drafts_accepted']/opt_stats['drafts_generated']*100:.1f}%")
        else:
            print()
        print(f"  Speedup vs Baseline: {opt_tps/base_tps:.2f}x")
        print(f"  Speedup vs Original: {opt_tps/orig_tps:.2f}x")

        print("-" * 70)