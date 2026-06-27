import torch
import json
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import math
from collections import defaultdict
import csv

def save_accuracy_csv(rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "difficulty", "correct", "id"])
        writer.writeheader()
        writer.writerows(rows)
    return

def load_dataset_samples(jsonl_path: str) -> dict[str, list[dict]]:
    """Uploads the dataset samples from a JSONL file and returns an index mapping task_id to list of samples."""
    index = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            task_id = sample.get("task_id", "")
            if task_id:
                index[task_id].append(sample)
    return dict(index)

def build_chat_text(question, tokenizer):
    message = question["messages"]
    # Separate prompt and assistant response
    message_prompt = message[:-1]
    return tokenizer.apply_chat_template(message_prompt, tokenize=False, add_generation_prompt=True)

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    question,
    max_new_tokens,
    do_sample = False,      # False = deterministic (greedy)
    temperature = 0.7,     # used only if do_sample=True
    top_p = 0.9,           # used only if do_sample=True
    repetition_penalty = 1.05 
):
    """"Generates an answer using the given model and tokenizer."""
    text = build_chat_text(question, tokenizer)
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        #eos_token_id=tokenizer.eos_token_id,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),  # force stop on im_end token
        ],
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    out = model.generate(**inputs, **gen_kwargs)
    # decode only newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_first_json(raw)

def generate_answer_vllm(question, client, model_name, max_new_tokens=512):
    """Generates an answer using vLLM."""
    messages = question["messages"][:-1]  
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,  # greedy
    )
    
    raw = response.choices[0].message.content.strip()
    return extract_first_json(raw)

def load_fine_tuned_model(model_name, adapter_dir, torch_dtype=torch.bfloat16):
    """Loads the fine-tuned model by merging the LoRA adapter with the base model."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base=AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    # Load the LoRA adapter and merge it with the base model
    model_ft = PeftModel.from_pretrained(base, adapter_dir)
    model_ft = model_ft.merge_and_unload()
    model_ft = model_ft.to("cuda")
    model_ft.eval()
    return model_ft, tokenizer

def filter_questions(question, max_len, max_new_tokens, tokenizer):
    """Filters out questions that would exceed the model's max_input_length when combined with the expected answer."""
    prompt_text = tokenizer.apply_chat_template(
        question["messages"][:-1],  
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + max_new_tokens <= max_len

def extract_first_json(text: str) -> str:
    """Extracts the first JSON object from the given text."""
    text = text.strip()
    if not text.startswith("{"):
        idx = text.find("{")
        if idx == -1:
            return text
        text = text[idx:]
    try:
        decoder = json.JSONDecoder()
        parsed, end_idx = decoder.raw_decode(text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return text

def eval_format_classification(answer):
    """Evaluates the format classification task by checking if both the predicted classification and decodability match the expected values."""
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    classification_ok = predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower()
    decodable_ok = predicted.get("is_decodable", False) == expected.get("is_decodable", False)
    return 1.0 if classification_ok and decodable_ok else 0.0

def eval_validation_judgement(answer):
    """Evaluates the validation judgement task by checking if the predicted accept/reject decision matches the expected one."""
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    return 1.0 if predicted.get("accept", False) == expected.get("accept", False) else 0.0

def eval_refusal(answer):
    """Evaluates the refusal task by checking if the model correctly identifies non-decodable blobs and doesn't generate code."""
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    not_decodable = predicted.get("is_decodable") == False
    no_code = predicted.get("decoder_code") is None
    return 1.0 if not_decodable and no_code else 0.0

