from preprocessing.preprocess import load_jsonl
from evaluation.acc_eval import (
    eval_format_classification,
    eval_validation_judgement,
    eval_refusal,
    load_dataset_samples,
    save_accuracy_csv,
)
from evaluation.struct_funct_eval import (
    mark_blob_to_json, 
    mark_decoder_synthesis, 
    mark_decoder_synthesis_undecodable, 
    save_report_csv
)

RESULTS_FILE = "./results/gemma/ft_gemma_3_results_2.jsonl"
REPORT_PATH = "./eval_reports/gemma/"
DATASET_PATH = "./data/blob_ground_truth_rows.jsonl"
ONTHOLOGY_PATH = "./data/onthology/field_synonyms.json"

results = load_jsonl(RESULTS_FILE)
dataset_index  = load_dataset_samples(DATASET_PATH)

correct = {"format_classification": 0, "validation_judgement": 0, "refusal": 0}
marks = {"decoder_synthesis": 0,"decoder_synthesis_undecodable": 0,"blob_to_json": 0,}
total1 = {"format_classification": 0, "validation_judgement": 0, "refusal": 0}
total2 = {"decoder_synthesis": 0, "decoder_synthesis_undecodable": 0, "blob_to_json": 0}

accuracy_rows = []   # for format_classification, validation_judgement, refusal
report_rows = []   # for decoder_synthesis, decoder_synthesis_undecodable, blob_to_json

# evaluate accuracy
for i, answer in enumerate(results):
    task = answer.get("task_type", "")
    difficulty = answer.get("metadata", {}).get("difficulty", "unknown")
    score = 0
    if answer["parse_ok"]:
        if answer["task_type"] == "format_classification":
            total1["format_classification"] += 1
            score = eval_format_classification(answer)
            correct["format_classification"] += score
            accuracy_rows.append({"task": task, "difficulty": difficulty, "correct": score, "id": answer.get("id", "")})
        elif answer["task_type"] == "validation_judgement":
            total1["validation_judgement"] += 1
            score = eval_validation_judgement(answer)
            correct["validation_judgement"] += score
            accuracy_rows.append({"task": task, "difficulty": difficulty, "correct": score, "id": answer.get("id", "")})
        elif answer["task_type"] == "refusal":
            total1["refusal"] += 1
            score = eval_refusal(answer)
            correct["refusal"] += score
            accuracy_rows.append({"task": task, "difficulty": difficulty, "correct": score, "id": answer.get("id", "")})
        else:
            continue
print("Accuracy evaluation results:")
for task in total1.keys():
    acc = correct[task] / total1[task] if total1[task] > 0 else 0
    print(f"{task}: {correct[task]}/{total1[task]} - Accuracy: {acc:.2%}")

# evaluate structural and functional correctness
for i, answer in enumerate(results):
    task = answer.get("task_type", "")
    difficulty = answer.get("metadata", {}).get("difficulty", "unknown")
    eval_report = {}
    
    if answer["task_type"] == "decoder_synthesis" and difficulty != "hard_negative_control":
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

            eval_report = mark_decoder_synthesis(
                answer,
                generalization_samples=gen_samples or None,
                blob_hex=blob_hex,
                expected_output=expected_output,
                ontology_path=ONTHOLOGY_PATH
            )

            marks["decoder_synthesis"] += eval_report["mark"]
            report_rows.append({"task": task, "difficulty": difficulty, **eval_report, "id": answer.get("id", "")})

    if answer["task_type"] == "decoder_synthesis" and difficulty == "hard_negative_control":
        total2["decoder_synthesis_undecodable"] += 1
        eval_report = mark_decoder_synthesis_undecodable(answer)
        marks["decoder_synthesis_undecodable"] += eval_report["mark"]
        report_rows.append({"task": task, "difficulty": difficulty, **eval_report, "id": answer.get("id", "")})
    
    elif answer["task_type"] == "blob_to_json":
        total2["blob_to_json"] += 1
        eval_report = mark_blob_to_json(answer, ontology_path=ONTHOLOGY_PATH)
        marks["blob_to_json"] += eval_report["mark"]
        report_rows.append({"task": task, "difficulty": difficulty, **eval_report, "id": answer.get("id", "")})

print("\nStructural and Functional Correctness Marks:")
for task in marks.keys():
    mark = marks[task] / (total2[task]) if total2[task] > 0 else 0
    print(f"{task}: Mean evaluation - Mark: {mark:.2f} / 5.00")
    
# sort results by task and difficulty
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "hard_negative_control": 3, "unknown": 4}
accuracy_rows.sort(key=lambda r: (r["task"], DIFFICULTY_ORDER.get(r["difficulty"], 99)))
report_rows.sort(key=lambda r: (r["task"], DIFFICULTY_ORDER.get(r["difficulty"], 99)))

# # save results to CSV
save_accuracy_csv(accuracy_rows, REPORT_PATH + "base_accuracy_results.csv")
save_report_csv(report_rows, REPORT_PATH + "ft_report_results.csv")