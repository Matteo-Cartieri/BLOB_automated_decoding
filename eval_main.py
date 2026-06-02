from preprocessing.preprocess import load_jsonl
from evaluation.acc_eval import (
    eval_format_classification,
    eval_validation_judgement,
    eval_refusal,
    load_dataset_samples,
)
from evaluation.struct_funct_eval import (mark_blob_to_json, mark_decoder_synthesis)

RESULTS_FILE = "./results/gemma/base_gemma_3_results_ood.jsonl"
DATASET_PATH = "./data/blob_ground_truth_rows.jsonl"

results = load_jsonl(RESULTS_FILE)
dataset_index  = load_dataset_samples(DATASET_PATH)

correct = {"format_classification": 0, "validation_judgement": 0, "refusal": 0}
marks = {"decoder_synthesis": 0,"blob_to_json": 0,}
total1 = {"format_classification": 0, "validation_judgement": 0, "refusal": 0}
total2 = {"decoder_synthesis": 0, "blob_to_json": 0}
# evaluate accuracy
for i, answer in enumerate(results):
    if answer["parse_ok"]:
        if answer["task_type"] == "format_classification":
            total1["format_classification"] += 1
            correct["format_classification"] += eval_format_classification(answer)
        elif answer["task_type"] == "validation_judgement":
            total1["validation_judgement"] += 1
            correct["validation_judgement"] += eval_validation_judgement(answer)
        elif answer["task_type"] == "refusal":
            total1["refusal"] += 1
            correct["refusal"] += eval_refusal(answer)
        else:
            continue

print("Evaluation Results:")
for task in total1.keys():
    acc = correct[task] / total1[task] if total1[task] > 0 else 0
    print(f"{task}: {correct[task]}/{total1[task]} - Accuracy: {acc:.2%}")
#print("Total Accuracy:", sum(correct.values()) / sum(total1.values()) if sum(total1.values()) > 0 else 0)

# evaluate structural and functional correctness
for i, answer in enumerate(results):
    if answer["task_type"] == "decoder_synthesis":
        total2["decoder_synthesis"] += 1
        task_id = answer.get("metadata", {}).get("task_id", "")
        samples = dataset_index.get(task_id, [])
        if not samples:
            print(f"[WARN] Sample {i} ({task_id}): no JSON reference found for decoder synthesis evaluation.")
        else:
            # use the first sample as primary reference and the others as generalization samples
            primary = samples[0]
            blob_hex = primary["blob_hex"]
            expected_output = primary["decoded_reference"]
            gen_samples = [{"blob_hex": s["blob_hex"], "expected_output": s["decoded_reference"]} for s in samples[1:]]

            mark = mark_decoder_synthesis(
                answer,
                generalization_samples=gen_samples or None,
                blob_hex=blob_hex,
                expected_output=expected_output,
            )

            marks["decoder_synthesis"] += mark          
    elif answer["task_type"] == "blob_to_json":
        total2["blob_to_json"] += 1
        marks["blob_to_json"] += mark_blob_to_json(answer)

print("\nStructural and Functional Correctness Marks:")
for task in marks.keys():
    mark = marks[task] / (total2[task]) if total2[task] > 0 else 0
    print(f"{task}: Mean evaluation - Mark: {mark:.2f} / 5.00")