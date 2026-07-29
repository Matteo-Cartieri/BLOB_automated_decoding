from evaluation.acc_eval import generate_answer_vllm, load_fine_tuned_model, filter_questions
from preprocessing.preprocess import load_jsonl,load_csv
from execution_feedback_repair.execution_feedback_repair import repair_prompt
from openai import OpenAI
import json
from transformers import AutoTokenizer
from collections import Counter

ADAPTER_DIR = "./fine-tuned-models/fine-tuned-gemma/adapter"
MODEL_NAME = "./models/gemma-3-4b-it" 
ANSWERS_FILE = "./results/gemma/ft_gemma_3_results_ood_new.jsonl"
REPORT_FILE = "./eval_reports/gemma/ft_report_results_ood.csv"
RESULTS_FILE = "./results/gemma/execution_feedback_repair/ft_gemma_3_results_ood_new.jsonl"
MAX_SEQ_LEN = 8400
MAX_NEW_TOKENS = 810

test_data = load_jsonl("./data/lora/new_ood_test.jsonl")
answers_data = load_jsonl(ANSWERS_FILE)
report_data = load_csv(REPORT_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)

answers_by_id = {item.get("id", ""): item for item in answers_data}
report_by_id = {item.get("id", ""): dict(item) for item in report_data}
test_data = [q for q in test_data if q.get("task_type") == "decoder_synthesis"]
repaired_data = []
for q in test_data:
    if q.get("id", "") not in answers_by_id or q.get("id", "") not in report_by_id :
        continue
    answer = answers_by_id.get(q.get("id", ""))
    if not answer.get("parse_ok") or not answer.get("predicted_json", {}).get("decoder_code"):
        continue
    report = report_by_id.get(q.get("id", ""))
    repaired_q = repair_prompt(q, answer, report)
    repaired_data.append(repaired_q)

test_data = [q for q in repaired_data if filter_questions(q, MAX_SEQ_LEN, MAX_NEW_TOKENS, tokenizer)]
print(f"Test samples after length filter: {len(test_data)}")

# for i in range(5):
#     print(f"Sample {i}:")
#     print(json.dumps(test_data[i].get("messages")[1], indent=2, ensure_ascii=False))

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
