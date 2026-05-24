import random
import torch
from transformers import Trainer
from training.data_collator import DataCollatorForCausalLMWithLabels
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    set_seed,
)
import os
from peft import LoraConfig, get_peft_model, PeftModel
from preprocessing.preprocess import preprocess_dataset

def build_trainer(model, training_args, train_dataset, eval_dataset, data_collator, tokenizer):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

def load_model(model_name, torch_dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        #device_map="auto",
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

def get_lora_config(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
 
def train(model_name, args, training_args, train_path, eval_path, max_seq_len):
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    train_ds=preprocess_dataset(train_path, tokenizer, max_seq_len)
    eval_ds=preprocess_dataset(eval_path, tokenizer, max_seq_len)

    model = load_model(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = get_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer)
    
    trainer = build_trainer(model, training_args, train_ds, eval_ds, data_collator, tokenizer)
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Train
    train_result = trainer.train()
    if hasattr(train_result, "metrics"):
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Save LoRA adapter
    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    print(f"[INFO] Saving LoRA adapter to: {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Optional: merge LoRA into base model
    if args.merge_model:
        print("[INFO] Merging LoRA adapter into base model (CPU)...")
        base = load_model(model_name)
        merged = PeftModel.from_pretrained(base, adapter_dir)
        merged = merged.merge_and_unload()

        merged_dir = os.path.join(args.output_dir, "merged_model")
        os.makedirs(merged_dir, exist_ok=True)
        print(f"[INFO] Saving merged model to: {merged_dir}")
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)