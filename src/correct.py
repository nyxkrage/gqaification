import torch
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithFlattening,
)
from accelerate import Accelerator
from dataset_loader import load_datasets_from_json
from datasets import Dataset
from constants import MODEL_NAME

def main():
    accelerator = Accelerator()
    model = AutoLigerKernelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        rope=True,
        rms_norm=True,
        swiglu=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    tokenized_dataset = Dataset.load_from_disk("./datasets/correction")

    data_collator = DataCollatorWithFlattening()

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=f"./models/{MODEL_NAME.replace("/", "_")}-Corrected",
        learning_rate=4e-4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        per_device_train_batch_size=144,
        eval_strategy="no",
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        save_strategy="epoch",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        torch_compile=True,
        torch_compile_mode="default",
        torch_compile_backend="inductor",
        dataloader_num_workers=8,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Start training
    trainer.train()
    
if __name__ == "__main__":
    main()