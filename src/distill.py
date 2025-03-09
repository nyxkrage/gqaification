import torch
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import Dataset
from loss import ForwardKLLoss
from accelerate import Accelerator
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from constants import MODEL_NAME

# Custom Trainer for logit distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = self.accelerator.prepare(teacher_model)
        self.teacher_model.eval()
        self.kd_loss = ForwardKLLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v.to(device=self.accelerator.device) for k, v in inputs.items()}
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # Get student outputs
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get teacher outputs with automatic device placement
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids.to(device=self.teacher_model.device),
                attention_mask=attention_mask.to(device=self.teacher_model.device),
            )
        loss = self.kd_loss(student_outputs.logits, teacher_outputs.logits, labels) / self.args.gradient_accumulation_steps
        
        return (loss, student_outputs) if return_outputs else loss
    
class AccelerateHFLM(HFLM):
    def __init__(
            self,
            model,
            accelerator,
            tokenizer,
            truncation = False,
            logits_cache = True,
            max_length = None,
            max_batch_size = 64,
            add_bos_token = False,
            batch_size = 1,
        ):
        super(type(self).__bases__[0], self).__init__()
        self._model = model
        self.accelerator = accelerator
        self._device = self.accelerator.device
        self._config = self.model.config
        self.backend = "causal"
        self.tokenizer = tokenizer

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size

        self._max_length = max_length
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
        self.AUTO_MODEL_CLASS = AutoModelForCausalLM

        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

    def get_model_info(self) -> dict:
        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1
        
        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""
            
        model_info = {
            "model_num_parameters": get_model_num_params(self.model),
            "model_dtype": get_model_dtype(self.model),
            "model_name": self.model.name_or_path,
        }
        
        return model_info

# PIQA evaluation callback using lm-eval-harness
class PIQACallback(TrainerCallback):
    def __init__(self, trainer: Trainer, eval_steps, tokenizer):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self.trainer.model.eval()
            model = self.trainer.accelerator.unwrap_model(self.trainer.model)
            if self.trainer.accelerator.is_main_process:
                self.evaluate_piqa(model)
            self.trainer.accelerator.wait_for_everyone()
        return control
    
    def evaluate_piqa(self, model):
            wrapper = HFLM(
                pretrained=model,
                parallelize=False,
                tokenizer=self.tokenizer,
            )

            results = evaluator.simple_evaluate(
                model=wrapper,
                tasks=["piqa"],
                batch_size=4,
                num_fewshot=0,
            )

            self.model.train()
        
            self.trainer.log({
                "eval/piqa": results["results"]["piqa"]["acc,none"]
            })

def main():
    accelerator = Accelerator()
    
    # Load models with flash attention and device_map="auto"
    teacher = AutoModelForCausalLM.from_pretrained(
        f"./models/{MODEL_NAME.replace('/', '_')}-Corrected",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    
    student = AutoLigerKernelForCausalLM.from_pretrained(
        f"./models/{MODEL_NAME.replace('/', '_')}-Pruned",
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
    tokenized_dataset = Dataset.load_from_disk("datasets/distillation")
    
    # Calculate evaluation steps
    batch_size = 1
    grad_accum_steps = 2
    num_devices = accelerator.num_processes
    steps_per_epoch = len(tokenized_dataset) // (batch_size * grad_accum_steps * num_devices)
    eval_steps = max(1, steps_per_epoch // 4)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=f"./models/{MODEL_NAME.replace('/', '_')}-GQA",
        learning_rate=4e-4,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        eval_strategy="no",
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        bf16=True,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        report_to=None,
        save_strategy="steps",
        save_steps=eval_steps,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        torch_compile=True,
        torch_compile_mode="default",
        torch_compile_backend="inductor",
        dataloader_num_workers=8,
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # LmEval callback currently doesn't work and causes as deadlock/NCCL to hang
    # Add PIQA evaluation callback
    # trainer.add_callback(PIQACallback(
    #     trainer=trainer,
    #     tokenizer=tokenizer,
    #     eval_steps=eval_steps,
    # ))

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()