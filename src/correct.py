import torch
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset, concatenate_datasets
from constants import MODEL_NAME
from accelerate import Accelerator
from lib.muon import Muon

torch._dynamo.config.capture_scalar_outputs = True

def main():
    accelerator = Accelerator()
    model = AutoLigerKernelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        rope=True,
        rms_norm=True,
        swiglu=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    tokenized_dataset = concatenate_datasets(
        [
            Dataset.load_from_disk("./datasets/dclm_40m"),
            Dataset.load_from_disk("./datasets/thestack_10m"),
        ]
    ).shuffle()

    batch_size = 16
    num_epochs = 1
    grad_accum_steps = 1
    weight_decay = 0.1
    learning_rate = 4e-4
    num_devices = accelerator.num_processes
    warmup_steps = 100
    steps_per_epoch = len(tokenized_dataset) // batch_size // grad_accum_steps // num_devices

    muon_params = [
        p
        for name, p in model.named_parameters()
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
    ]
    adamw_params = [
        p
        for name, p in model.named_parameters()
        if not (
            p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        )
    ]

    optimizer = Muon(
        lr=learning_rate,
        wd=weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps_per_epoch * num_epochs,
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=f"./models/{MODEL_NAME.replace('/', '_')}-Corrected",
        learning_rate=learning_rate,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="no",
        logging_steps=1,
        bf16=True,
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
        optimizers=(optimizer, lr_scheduler),
        train_dataset=tokenized_dataset,
    )
    
    # Start training
    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(f"./models/{MODEL_NAME.replace('/', '_')}-Corrected")
    accelerator.wait_for_everyone()
    
if __name__ == "__main__":
    main()