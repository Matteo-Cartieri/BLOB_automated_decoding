import torch
import json
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

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
    max_new_tokens = 512,
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
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base=AutoModelForCausalLM.from_pretrained(
        model_name,
        #device_map="auto",
        dtype=torch_dtype,
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

def check_encoding_family(predicted, expected):
    pred_enc = predicted.get("classification", "").strip().lower()
    exp_enc = expected.get("classification", "").strip().lower()
    return pred_enc == exp_enc 

def check_expected_fields(predicted, expected):
    pred = set(predicted)
    exp  = set(expected)
    if not exp:
        return 0.0
    recall    = len(pred & exp) / len(exp)
    precision = len(pred & exp) / len(pred) if pred else 0.0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def eval_decoder_synthesis(answer):
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    class_score = predicted.get("expected_fields", "") == expected.get("expected_fields", "")
    encoding_score = check_encoding_family(predicted, expected)
    return 1.0 if class_score and encoding_score else 0.0

def eval_blob_to_json(answer):
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    return 1.0 if predicted == expected else 0.0

def eval_format_classification(answer):
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    encoding_score = check_encoding_family(predicted, expected)
    decodable_score =predicted.get("is_decodable", False) == expected.get("is_decodable", False)
    return 1.0 if encoding_score and decodable_score else 0.0

def eval_validation_judgement(answer):
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    return 1.0 if predicted.get("accept", False) == expected.get("accept", False) else 0.0

def eval_refusal(answer):
    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}
    return 1.0 if predicted.get("is_decodable", False) == expected.get("is_decodable", False) else 0.0

