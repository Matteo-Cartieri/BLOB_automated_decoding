import json
from datasets import Dataset

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# def apply_chat_template(ex, tokenizer, max_seq_len):
#     messages = ex["messages"]
#     full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#     prompt_messages = messages[:-1]
#     prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

#     full_enc = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len, return_tensors=None)
#     prompt_enc = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_seq_len, return_tensors=None)

#     input_ids = full_enc["input_ids"]
#     attention_mask = full_enc["attention_mask"]
#     labels = input_ids.copy()

#     prompt_len = len(prompt_enc["input_ids"])
#     for i in range(min(prompt_len, len(labels))):
#         labels[i] = -100

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#     }
def apply_chat_template(ex, tokenizer, max_seq_len):
    messages = ex["messages"]

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize prompt and full without truncation
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    full_ids_notrunc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # Real response length (before any truncation)
    real_response_len = len(full_ids_notrunc) - len(prompt_ids)

    # Tokenize full text with truncation to fit max_seq_len
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt_ids), len(input_ids))
    available_for_response = len(input_ids) - prompt_len

    # Discard if the response is truncated 
    if available_for_response < real_response_len:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [-100] * len(input_ids),
            "__skip__": True,
        }

    labels = input_ids.copy()
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "__skip__": False,
    }

def filter_by_length(example, max_tokens):
    """Discard examples that exceed max_tokens."""
    return len(example["input_ids"]) <= max_tokens

def preprocess_dataset(data_path, tokenizer, max_length):
    raw_data = load_jsonl(data_path)
    print(f"Raw dataset: {len(raw_data)} examples")
    
    # Convert to Dataset
    dataset = Dataset.from_dict({
        'id': [d['id'] for d in raw_data],
        'task_type': [d.get('task_type', 'unknown') for d in raw_data],
        'messages': [d['messages'] for d in raw_data],
        'metadata': [d.get('metadata', {}) for d in raw_data],
    })
    
    # Apply chat template
    processed = dataset.map(
        lambda ex: apply_chat_template(ex, tokenizer, max_length),
        remove_columns=['messages', 'metadata','id', 'task_type']
    )

    processed = processed.filter(lambda ex: not ex["__skip__"])
    processed = processed.remove_columns(["__skip__"])
    
    # Filter by length
    # processed = processed.filter(lambda ex: filter_by_length(ex, max_length),)
    print(f"Dataset after processing: {len(processed)} examples")
    
    return processed