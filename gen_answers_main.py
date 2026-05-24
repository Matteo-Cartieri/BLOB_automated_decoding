from evaluation.evaluate import generate_answer_vllm, load_fine_tuned_model, filter_questions
from preprocessing.preprocess import load_jsonl
from openai import OpenAI
import json
from transformers import AutoTokenizer

ADAPTER_DIR = "./fine-tuned-model/adapter"
MODEL_NAME = "./models/Qwen2.5-7B"
RESULTS_FILE = "./results/base_qwen_2_5_results.jsonl"
MAX_SEQ_LEN = 7000

test_data = load_jsonl("./data/lora/test.jsonl")
#model, tokenizer = load_fine_tuned_model(MODEL_NAME, ADAPTER_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
test_data = [q for q in test_data if filter_questions(q, MAX_SEQ_LEN, 512, tokenizer)]
print(f"Test samples after length filter: {len(test_data)}")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

with open(RESULTS_FILE, "w") as f:
    for i, question in enumerate(test_data):
        answer_str = generate_answer_vllm(question, client, MODEL_NAME)

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


# for i in range(22,23):
#     question = test_data[i]
#     print("Question:", question["messages"][1]["content"][:200], "\n***\n")
#     answer = generate_answer(model, tokenizer, question)
#     print("Answer:", answer)
#     print("-" * 50)