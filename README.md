# Download Dataset Splits
This will stream the datasets from Huggingface counting the tokens, and saving them to `datasets/`

```
python src/download_dataset.py --tokenizer MiniLLM/MiniPLM-llama3.1-212M --dataset bigcode/the-stack-smol --num-tokens 10000000 --output-path datasets/thestack_10m --text-column content
python src/download_dataset.py --tokenizer MiniLLM/MiniPLM-llama3.1-212M --dataset mlfoundations/dclm-baseline-1.0 --num-tokens 40000000 --output-path datasets/dclm_40m
python src/download_dataset.py --tokenizer MiniLLM/MiniPLM-llama3.1-212M --dataset bigcode/the-stack-smol --num-tokens 200000000 --output-path datasets/thestack_200m --text-column content
python src/download_dataset.py --tokenizer MiniLLM/MiniPLM-llama3.1-212M --dataset mlfoundations/dclm-baseline-1.0 --num-tokens 800000000 --output-path datasets/dclm_800m
```

# Run Correction
This will train the model on a small subset of the full dataset (50M tokens vs 1B tokens), this is to align the teacher with the dataset for the distillation.

```
accelerate launch src/correct.py
```

# Prune Model
This will estimate the importance of the models kv heads and prune them, this is done by running the model over a tiny subset of the dataset and saving the activations of the k and v tensors.

```
python src/importance.py
python src/prune.py
```

# Distill Model
This will use knowledge distillation to train the pruned model on the full dataset, using the corrected model as the teacher.

```
accelerate launch src/distill.py
```

# Test Healed Model

```
python src/generatation.py
```