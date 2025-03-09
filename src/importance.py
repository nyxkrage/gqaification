import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_loader import load_datasets_from_json
import pickle
from tqdm import tqdm
from typing import Dict, List, Literal

# Configuration
MODEL_NAME = './models/'
DATASET_CONFIG = None
SPLIT = 'train'
BATCH_SIZE = 16
MAX_SAMPLES = 1024  # Number of samples to process

# Load the dataset
dataset = load_datasets_from_json("dataset_configs_correction.json")
dataset = dataset.select(range(MAX_SAMPLES))  # Limit the number of samples
dataset = dataset.shuffle()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(f"./models/{MODEL_NAME.replace("/", "_")}", torch_dtype=torch.bfloat16)
model.cuda()  # Move the model to GPU

# Set the model to evaluation mode
model.eval()

# Dictionary to store activation data for each attention head
activation_data = {}

def aggregate_simple_mean(k_scores, v_scores):
    return (k_scores + v_scores) / 2

def aggregate_geometric_mean(k_scores, v_scores):
    return torch.sqrt(k_scores * v_scores)

def aggregate_max(k_scores, v_scores):
    return torch.maximum(k_scores, v_scores)

def aggregate_min(k_scores, v_scores):
    return torch.minimum(k_scores, v_scores)

def aggregate_weighted(k_scores, v_scores, alpha=0.5):
    return alpha * k_scores + (1 - alpha) * v_scores

def aggregate_correlation_weighted(k_scores, v_scores):
    # Compute correlation between K and V scores
    corr = torch.corrcoef(torch.stack([k_scores, v_scores]))[0,1]
    # If correlation is high, use average; if low, use maximum
    weight = (1 + corr) / 2  # maps correlation [-1,1] to weights [0,1]
    return weight * (k_scores + v_scores)/2 + (1-weight) * torch.maximum(k_scores, v_scores)

AGGREGATION_FUNCTIONS={
    'simple_mean': aggregate_simple_mean,
    'geometric_mean': aggregate_geometric_mean,
    'max': aggregate_max,
    'min': aggregate_min,
    'weighted': lambda k, v: aggregate_weighted(k, v, alpha=0.6),
    'correlation': aggregate_correlation_weighted
}

# Function to register hooks on key and value projection layers
def get_kv_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'k_proj' in name or 'v_proj' in name:
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
    return hooks

# Hook function to capture activations
def create_hook(name):
    def hook(module, input, output):
        # output: (batch_size, seq_len, head_dim * num_heads)
        with torch.no_grad():
            # Reshape output to (batch_size, seq_len, num_heads, head_dim)
            batch_size, seq_len, total_dim = output.shape

            # Get the number of attention heads
            num_heads = model.config.num_attention_heads
            head_dim = total_dim // num_heads

            # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
            output = output.view(batch_size, seq_len, num_heads, head_dim)            # Permute to (num_heads, batch_size, seq_len, head_dim)
            # Permute to (num_heads, batch_size, seq_len, head_dim)
            output = output.permute(2, 0, 1, 3)

            # Compute L2 norm across batch dimension
            l2_norm = torch.norm(output, p=2, dim=1)  # (num_heads, seq_len, head_dim)
            # Compute mean(abs) across seq_len
            mean_abs = torch.mean(torch.abs(l2_norm), dim=1)  # (num_heads, head_dim)
            # Sum over head_dim to get a scalar per head
            importance_scores = torch.sum(mean_abs, dim=1)  # (num_heads,)
            # print(importance_scores)

            # Store the importance scores
            proj_type = 'k_proj' if 'k_proj' in name else 'v_proj'
            layer_idx = int(name.split('.')[2])
            if layer_idx not in activation_data:
                activation_data[layer_idx] = {}
            if proj_type not in activation_data[layer_idx]:
                activation_data[layer_idx][proj_type] = importance_scores.detach().cpu().pow(2)  # Store squared values
            else:
                activation_data[layer_idx][proj_type] += importance_scores.detach().cpu().pow(2)  # Add squared values
    return hook

# Register hooks on key and value projections
hooks = get_kv_hooks(model)

# Process the dataset
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch_texts = dataset[i:i+BATCH_SIZE]['text']
    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}  # Move inputs to GPU

    with torch.no_grad():
        outputs = model(**inputs)

for layer_idx in activation_data.keys():
    for proj_type in activation_data[layer_idx].keys():
        activation_data[layer_idx][proj_type] = torch.sqrt(activation_data[layer_idx][proj_type])

# Remove hooks
for hook in hooks:
    hook.remove()

min_importance  = {}
for layer_idx, head_scores in activation_data.items():
    k_scores = head_scores['k_proj']
    v_scores = head_scores['v_proj']
    min_importance[layer_idx] = AGGREGATION_FUNCTIONS['min'](k_scores, v_scores)

# Save the importance scores
with open('kv_head_importance_scores_min.pkl', 'wb') as f:
    pickle.dump(min_importance, f)


print("Importance scores saved to 'kv_head_importance_scores.pkl'")