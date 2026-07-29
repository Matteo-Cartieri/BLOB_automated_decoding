import math
import ast
import subprocess
import tempfile
from collections import defaultdict
import textwrap
import os
import json
import csv
from evaluation.acc_eval import load_dataset_samples
from preprocessing.preprocess import load_jsonl, load_csv
from evaluation.struct_funct_eval import (
    load_ontology,
    build_reverse_index,
    lists_similarity,
    field_level_score,
    is_syntactically_valid,
    is_hardcoded,
    is_empty_code,
    run_decoder,
    make_report,
    mark_decoder_synthesis,
    mark_decoder_synthesis_undecodable,
)

def save_ood_report_csv(rows, filename):
    fields = ["sample_id", "encoding_family", "initial_decoder_parse_success", "repaired_decoder_parse_success", "initial_runtime_success", 
              "repaired_runtime_success", "initial_json_parse_success", "repaired_json_parse_success", "initial_value_accuracy", 
              "repaired_value_accuracy", "initial_manual_score", "repaired_manual_score", "initial_automatic_score", "repaired_automatic_score",
              "repair_success_rate", "repair_regression_rate"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(serialized)
    return

def create_ood_report(results_file, results_ef_file, dataset_path, ontology_path, output_file):
    results = load_jsonl(results_file)
    results_ef = load_jsonl(results_ef_file)
    results_ef_by_id = {item.get("id", ""): item for item in results_ef}
    dataset_index = load_dataset_samples(dataset_path)

    report_rows = []   

    for answer in results:
        answer_id = answer.get("id", "")
        if answer_id not in results_ef_by_id :
            if answer.get("task_type") == "decoder_synthesis":
                print(f"[WARN] Sample {answer_id}: no corresponding execution feedback evaluation found.")
                continue
            continue
        answer_ef = results_ef_by_id[answer_id]
        
        difficulty = answer.get("metadata", {}).get("difficulty", "unknown")
        task = answer.get("task_type", "")
        eval_report = {}
        eval_report_ef = {}
        
        if answer["task_type"] == "decoder_synthesis" and difficulty != "hard_negative_control":
            task_id =answer.get("metadata", {}).get("task_id", "")
            samples = dataset_index.get(task_id, [])
            if not samples:
                print(f"[WARN] Sample ({task_id}): no JSON reference found for decoder synthesis evaluation.")
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
                    ontology_path=ontology_path
                )
                eval_report_ef = mark_decoder_synthesis(
                    answer_ef,
                    generalization_samples=gen_samples or None,
                    blob_hex=blob_hex,
                    expected_output=expected_output,
                    ontology_path=ontology_path
                )
                value_accuracy = eval_report.get("field_scores", {}).get("field_value_accuracy", "") if eval_report.get("field_scores") else 0.0
                repaired_value_accuracy = eval_report_ef.get("field_scores", {}).get("field_value_accuracy", "") if eval_report_ef.get("field_scores") else 0.0
                
                report_rows.append({"sample_id": answer_id, "encoding_family": answer.get("metadata", {}).get("encoding_family", ""), 
                                    "initial_decoder_parse_success": eval_report.get("decoder_python_parse_success", ""), "repaired_decoder_parse_success": eval_report_ef.get("decoder_python_parse_success", ""),
                                    "initial_runtime_success": eval_report.get("decoder_runtime_success", ""), "repaired_runtime_success": eval_report_ef.get("decoder_runtime_success", ""),
                                    "initial_json_parse_success": eval_report.get("valid_json_output", ""), "repaired_json_parse_success": eval_report_ef.get("valid_json_output", ""),
                                    "initial_value_accuracy": value_accuracy, "repaired_value_accuracy": repaired_value_accuracy,
                                    "initial_manual_score": 0, "repaired_manual_score": 0, "initial_automatic_score": eval_report.get("automatic_score", ""), 
                                    "repaired_automatic_score": eval_report_ef.get("automatic_score", "")})

        # if answer["task_type"] == "decoder_synthesis" and difficulty == "hard_negative_control":
        #     eval_report = mark_decoder_synthesis_undecodable(answer)
        #     report_rows.append({})
            
    save_ood_report_csv(report_rows, output_file)
            
