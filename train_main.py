from training.build_trainer import train
from transformers import TrainingArguments
from types import SimpleNamespace

TRAIN_DATA_PATH = "./data/lora/train.jsonl"
VAL_DATA_PATH = "./data/lora/validation.jsonl"
OUTPUT_DIR = "./fine-tuned-models/fine-tuned-gemma"

MODEL_NAME = "./models/gemma-3-4b-it"
MAX_SEQ_LEN = 7500
SEED = 42

# LoRA defaults
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,       
    # eval_strategy="epoch",
    # save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=100,         
    save_strategy="steps",
    save_steps=100,  

    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
    optim="adamw_torch",
    bf16=True,
    gradient_checkpointing=True,
)

args = SimpleNamespace(
    output_dir = OUTPUT_DIR,
    lora_r = LORA_R,
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    seed = SEED,
    merge_model = False,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
)

train(MODEL_NAME, args, training_args, TRAIN_DATA_PATH, VAL_DATA_PATH, MAX_SEQ_LEN)