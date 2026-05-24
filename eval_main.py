from preprocessing.preprocess import load_jsonl
from evaluation.evaluate import (
    eval_decoder_synthesis,
    eval_blob_to_json,
    eval_format_classification,
    eval_validation_judgement,
    eval_refusal
)

RESULTS_FILE = "./results/base_qwen_2_5_results.jsonl"

results = load_jsonl(RESULTS_FILE)

correct = {"decoder_synthesis": 0, "blob_to_json": 0, "format_classification": 0, "validation_judgement": 0, "refusal": 0}
total = {"decoder_synthesis": 0, "blob_to_json": 0, "format_classification": 0, "validation_judgement": 0, "refusal": 0}

for i, answer in enumerate(results):
    total[answer["task_type"]] += 1
    if answer["parse_ok"]:
        if answer["task_type"] == "decoder_synthesis":
            correct["decoder_synthesis"] += eval_decoder_synthesis(answer)
        elif answer["task_type"] == "blob_to_json":
            correct["blob_to_json"] += eval_blob_to_json(answer)
        elif answer["task_type"] == "format_classification":
            correct["format_classification"] += eval_format_classification(answer)
        elif answer["task_type"] == "validation_judgement":
            correct["validation_judgement"] += eval_validation_judgement(answer)
        elif answer["task_type"] == "refusal":
            correct["refusal"] += eval_refusal(answer)
        else:
            print(f"Sample {i} - Task: {answer['task_type']} (unknown)")

print("Evaluation Results:")
for task in total.keys():
    acc = correct[task] / total[task] if total[task] > 0 else 0
    print(f"{task}: {correct[task]}/{total[task]} - Accuracy: {acc:.2%}")
    
print("Total Accuracy:", sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0)
