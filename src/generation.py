from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from constants import MODEL_NAME

# Define the models you want to use
model_names = [
    f"{MODEL_NAME}-Corrected",
    f"{MODEL_NAME}-GQA",
    f"{MODEL_NAME}-Pruned",
]

def set_pad_token(model):
    gen_config = model.generation_config
    gen_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config = gen_config
    return model

# Load the tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id
models = [set_pad_token(AutoModelForCausalLM.from_pretrained("./models/" + model_name, torch_dtype=torch.bfloat16, device_map="auto")) for model_name in model_names]


# Define the prompt
prompt = "Once upon a time"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text from each model
for i, model in enumerate(models):
    print(f"Generating text with model: {model_names[i]}")
    output = model.generate(input_ids=inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device), do_sample=True, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"\033[1m{prompt}\033[0m", end="")
    print(generated_text)
    print("-" * 80)