from evaluation.acc_eval import generate_answer_vllm, load_fine_tuned_model, filter_questions
from preprocessing.preprocess import load_jsonl
from retrieval_template.retrieve import load_templates, augment_prompt
from openai import OpenAI
import json
from transformers import AutoTokenizer
from collections import Counter
from sentence_transformers import SentenceTransformer

ADAPTER_DIR = "./fine-tuned-models/fine-tuned-gemma/adapter"
MODEL_NAME = "./models/gemma-3-4b-it" 
RESULTS_FILE = "./results/gemma/retrieval_template/ft_gemma_3_results_similarity_k_3.jsonl"
MAX_SEQ_LEN = 8700
MAX_NEW_TOKENS = 730

embedder = SentenceTransformer("all-MiniLM-L6-v2")
templates = load_templates("./templates", embedder)

test_data = load_jsonl("./data/lora/new_ood_test.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
test_data = [augment_prompt(q, templates, "semantic_similarity",embedder,top_k=3) for q in test_data if q.get("task_type") == "decoder_synthesis"]
#test_data = [augment_prompt(q, templates, "magic_bytes",embedder) for q in test_data if q.get("task_type") == "decoder_synthesis"]
test_data = [q for q in test_data if filter_questions(q, MAX_SEQ_LEN, MAX_NEW_TOKENS, tokenizer)]
print(f"Test samples after length filter: {len(test_data)}")

for i in range(3):
    print(f"\nSample {i} {test_data[i].get('id')}:")
    print(json.dumps(test_data[i].get("messages")[1].get("content", "")[:1100], indent=2, ensure_ascii=False))

# client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# with open(RESULTS_FILE, "w") as f:
#     for i, question in enumerate(test_data):
#         answer_str = generate_answer_vllm(question, client, "my-gemma", max_new_tokens=MAX_NEW_TOKENS)

#         try:
#             answer_json = json.loads(answer_str)
#             parse_ok = True
#         except json.JSONDecodeError:
#             answer_json = None
#             parse_ok = False

#         record = {
#             "id": question["id"],
#             "task_type": question.get("task_type"),
#             "metadata": question.get("metadata", {}),
#             "expected": json.loads(question["messages"][-1]["content"]),
#             "predicted_str": answer_str,
#             "predicted_json": answer_json,
#             "parse_ok": parse_ok,
#         }

#         f.write(json.dumps(record, ensure_ascii=False) + "\n")
#         f.flush()  

#         if i % 10 == 0:
#             print(f"[{i}/{len(test_data)}] completed")
