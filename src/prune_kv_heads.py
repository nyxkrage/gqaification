import torch
import torch.nn as nn
import pickle
from transformers import AutoModelForCausalLM
from constants import MODEL_NAME, TARGET_NUM_KV_HEADS

def get_head_indices(head_idx, head_dim):
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    return list(range(start, end))

# Load importance scores and model
with open('kv_head_importance_scores_min.pkl', 'rb') as f:
    kv_importance_scores = pickle.load(f)



model = AutoModelForCausalLM.from_pretrained(f"./models/{MODEL_NAME.replace('/', '_')}-Corrected", torch_dtype=torch.bfloat16)
model.cpu()

# Update config
num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads
group_size = num_heads // TARGET_NUM_KV_HEADS  # Number of q heads per kv head
model.config.num_key_value_heads = TARGET_NUM_KV_HEADS

# For each layer, prune the least important K/V heads
for layer_idx, importance_scores in kv_importance_scores.items():
    print(f"Processing layer {layer_idx}")
    
    # Get sorted indices of K/V heads (descending order)
    sorted_indices = torch.argsort(importance_scores, descending=True)
    selected_indices = sorted_indices[:TARGET_NUM_KV_HEADS]
    
    attn = model.model.layers[layer_idx].self_attn
    orig_k_proj = attn.k_proj.weight.data
    orig_v_proj = attn.v_proj.weight.data
    orig_q_proj = attn.q_proj.weight.data
    
    # Get the indices for the selected K/V heads
    selected_kv_indices = []
    for idx in selected_indices:
        selected_kv_indices.extend(get_head_indices(idx.item(), head_dim))
    
    # Create new K/V projection matrices
    new_k_proj = orig_k_proj[:, selected_kv_indices]
    new_v_proj = orig_v_proj[:, selected_kv_indices]
    
    # Update the k_proj and v_proj weights
    attn.k_proj = nn.Linear(hidden_size, TARGET_NUM_KV_HEADS * head_dim, bias=model.config.attention_bias)
    attn.v_proj = nn.Linear(hidden_size, TARGET_NUM_KV_HEADS * head_dim, bias=model.config.attention_bias)
    # Note the transpose back to Linear layer expected shape
    attn.k_proj.weight.data = new_k_proj.T  # [TARGET_NUM_KV_HEADS * head_dim, hidden_size]
    attn.v_proj.weight.data = new_v_proj.T

# Save the pruned model
model.save_pretrained(f"./models/{MODEL_NAME.replace('/', '_')}-Pruned")
print(f"Model saved to './models/{MODEL_NAME.replace('/', '_')}-Pruned'")