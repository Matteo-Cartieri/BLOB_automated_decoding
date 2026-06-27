from evaluation.acc_eval import generate_answer_vllm, load_fine_tuned_model, filter_questions
from preprocessing.preprocess import load_jsonl
from openai import OpenAI
import json
from transformers import AutoTokenizer

ADAPTER_DIR = "./fine-tuned-models/fine-tuned-gemma/adapter"
MODEL_NAME = "./models/gemma-3-4b-it" 
RESULTS_FILE = "./results/gemma/ft_gemma_3_results_2.jsonl"
MAX_SEQ_LEN = 7800
MAX_NEW_TOKENS = 850

test_data = load_jsonl("./data/lora/test.jsonl")
#model, tokenizer = load_fine_tuned_model(MODEL_NAME, ADAPTER_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
test_data = [q for q in test_data if filter_questions(q, MAX_SEQ_LEN, MAX_NEW_TOKENS, tokenizer)]
print(f"Test samples after length filter: {len(test_data)}")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

with open(RESULTS_FILE, "w") as f:
    for i, question in enumerate(test_data):
        answer_str = generate_answer_vllm(question, client, "my-gemma", max_new_tokens=MAX_NEW_TOKENS)

        try:
            answer_json = json.loads(answer_str)
            parse_ok = True
        except json.JSONDecodeError:
            answer_json = None
            parse_ok = False

        record = {
            "id": question["id"],
            "task_type": question.get("task_type"),
            "metadata": question.get("metadata", {}),
            "expected": json.loads(question["messages"][-1]["content"]),
            "predicted_str": answer_str,
            "predicted_json": answer_json,
            "parse_ok": parse_ok,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()  

        if i % 10 == 0:
            print(f"[{i}/{len(test_data)}] completed")
