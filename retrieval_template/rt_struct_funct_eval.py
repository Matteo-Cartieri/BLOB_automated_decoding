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

def rt_mark_decoder_synthesis_undecodable(answer):
    predicted = answer.get("predicted_json") or {}
    expected  = answer.get("expected") or {}

    eval_report = {
        "encoding_classification_accuracy":    0,
        "expected_fields":      0.0,
        "errors":               [],
        "automatic_score":                 0,
    }
    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0, errors=["Parsing failed for the answer"])
    
    # checks on metadata and classification
    eval_report["encoding_classification_accuracy"] = (predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower())
    eval_report["expected_fields"] = lists_similarity(predicted.get("expected_fields", []), expected.get("expected_fields", []))
    
    eval_report["decodable_ok"] = (predicted.get("is_decodable") == expected.get("is_decodable"))

    code = predicted.get("decoder_code", "")
    if expected.get("is_decodable") is False:
        # if the expected sample is not decodable, check if the model identified it as non-decodable and predicted the expected fields
        if predicted.get("is_decodable") is False and is_empty_code(code):
            meta_ok = (eval_report["encoding_classification_accuracy"] + eval_report["expected_fields"]) / 2
            if meta_ok >= 0.85:
                return make_report(eval_report, mark=5)
            elif meta_ok >= 0.65:
                return make_report(eval_report, mark=4)
            elif meta_ok >= 0.45:
                return make_report(eval_report, mark=3)
            elif meta_ok >= 0.25:
                return make_report(eval_report, mark=2)
            else:
                return make_report(eval_report, mark=1)
        elif predicted.get("is_decodable") is False and not is_empty_code(code):
            return make_report(eval_report, mark=1, errors=["Correct refusal but decoder_code is not empty"])
        else:
            return make_report(eval_report, mark=0, errors=["Model claimed sample is decodable but it is not"])
    else:
        return make_report(eval_report, mark=0, errors=["Expected sample is decodable but this function is for non-decodable cases only"])

def rt_mark_decoder_synthesis(answer, generalization_samples, blob_hex, expected_output, ontology_path):
    """
    Mark synthesis of BLOB to JSON decoder.

    Marking scale:
        0  no code generated or non-decodable
        1  execution failed (code generated but crashes)
        2  valid JSON output but incorrect fields
        3  partially correct 
        4  almost correct
        5  correct, reusable decoder
    """
    predicted = answer.get("predicted_json") or {}
    expected  = answer.get("expected") or {}

    eval_report = {
        "encoding_classification_accuracy": 0,
        "decoder_python_parse_success": 0,
        "decoder_runtime_success": 0,
        "expected_fields":      0.0,
        "field_scores":         None,
        "generalization_score": 0.0,
        "errors":               [],
        "manual_score": 0,
        "automatic_score": 0,
    }
    
    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0, errors=["Parsing failed for the answer"])
    
    # checks on metadata and classification
    eval_report["encoding_classification_accuracy"] = 1 if predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower() else 0
    eval_report["expected_fields"] = lists_similarity(predicted.get("expected_fields", []), expected.get("expected_fields", []))
    
    eval_report["decodable_ok"] = 1 if predicted.get("is_decodable") == expected.get("is_decodable") else 0

    code = predicted.get("decoder_code", "")
    if expected.get("is_decodable") is False:
        # if the expected sample is not decodable, check if the model identified it as non-decodable and predicted the expected fields
        if predicted.get("is_decodable") is False and is_empty_code(code):
            meta_ok = (eval_report["encoding_classification_accuracy"] + eval_report["expected_fields"]) / 2
            if meta_ok >= 0.85:
                return make_report(eval_report, mark=5)
            elif meta_ok >= 0.65:
                return make_report(eval_report, mark=4)
            elif meta_ok >= 0.45:
                return make_report(eval_report, mark=3)
            elif meta_ok >= 0.25:
                return make_report(eval_report, mark=2)
            else:
                return make_report(eval_report, mark=1)
        elif predicted.get("is_decodable") is True:
            return make_report(eval_report, mark=0, errors=["Model claimed sample is decodable but it is not"])

    # if code is absent
    if is_empty_code(code):
        return make_report(eval_report, mark=0, errors=["The decoder_code field is empty"])
    
    # if code not syntactically valid
    code = code.strip()
    if not is_syntactically_valid(code):
        return make_report(eval_report, mark=0, errors=["SyntaxError in generated code"])

    eval_report["decoder_python_parse_success"] = 1

    # check for hardcoding patterns in the code
    eval_report["hardcoded"] = is_hardcoded(code)
    if eval_report["hardcoded"]:
        return make_report(eval_report, mark=0, errors=["The code contains hardcoded values"])

    # run the code and evaluate execution and output
    run_result = run_decoder(code, blob_hex)
    eval_report["decoder_runtime_success"] = 1 if run_result["success"] else 0
    eval_report["valid_json_output"] = 1 if run_result["success"] and isinstance(run_result["output"], dict) else 0

    # if code not executable
    if not run_result["success"]:
        return make_report(eval_report, mark=0, errors=[run_result["error"]])
    
    # if code executable but output not valid JSON
    if run_result["success"] and not eval_report["valid_json_output"]:
        return make_report(eval_report, mark=1, errors=["Invalid JSON output"])

    # evaluate field-level scores on decoder output
    ontology = load_ontology(ontology_path)
    reverse_index = build_reverse_index(ontology)
    eval_report["field_scores"] = field_level_score(run_result["output"], expected_output, reverse_index)
 
    # generalization score
    gen_score = 0.0
    if generalization_samples:
        gen_results = []
        for sample in generalization_samples:
            r = run_decoder(code, sample["blob_hex"])
            if r["success"] and isinstance(r["output"], dict):
                fs = field_level_score(r["output"], sample["expected_output"], reverse_index)
                gen_results.append(0.2 * fs["field_name_f1"] + 0.8 * fs["field_value_accuracy"])
            else:
                gen_results.append(0.0)
        gen_score = sum(gen_results) / len(gen_results)
        eval_report["generalization_score"] = round(gen_score, 4)

    # final score
    f1 = eval_report["field_scores"].get("field_name_f1", 0.0)
    val_acc = eval_report["field_scores"].get("field_value_accuracy", 0.0)
    hallucinated = eval_report["field_scores"].get("hallucination_penalty", 0.0)
    meta_ok = (eval_report["encoding_classification_accuracy"] + eval_report["expected_fields"]) / 2
    composite = (0.05 * f1 + 0.8 * val_acc + 0.05 * gen_score + 0.05 * meta_ok + 0.05 * (1 - hallucinated))
    if composite >= 0.85:
        return make_report(eval_report, mark=5)
    elif composite >= 0.65:
        return make_report(eval_report, mark=4)
    elif composite >= 0.45:
        return make_report(eval_report, mark=3)
    else:
        return make_report(eval_report, mark=2)

def save_general_report_csv(rows, filename):
    fields = ["sample_id", "encoding_family", "encoding_classification_accuracy", "decoder_python_parse_success", "decoder_runtime_success", "json_parse_success", "field_name_f1", "field_value_accuracy", "hallucinated_field_rate", "manual_score", "automatic_score", "ood_success"]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(serialized)
    return
    
def save_ood_report_csv(rows, filename):
    fields = ["sample_id", "encoding_family", "seen_or_ood", "retrieved_template_id", "template_rank", "encoding_prediction_without_template","encoding_prediction_with_template", "decoder_runs_without_template", "decoder_runs_with_template", "json_valid_without_template", "json_valid_with_template", "value_accuracy_without_template", "value_accuracy_with_template", "manual_score_without_template", "manual_score_with_template", "automatic_score_without_template", "automatic_score_with_template"]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(serialized)
    return

def create_general_report(results_file, dataset_path, ontology_path, output_file):
    results = load_jsonl(results_file)
    dataset_index = load_dataset_samples(dataset_path)

    report_rows = []   

    for answer in results:
        answer_id = answer.get("id", "")
        difficulty = answer.get("metadata", {}).get("difficulty", "unknown")
        task = answer.get("task_type", "")
        eval_report = {}
        
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
                
                value_accuracy = eval_report.get("field_scores", {}).get("field_value_accuracy", "") if eval_report.get("field_scores") else 0.0
                report_rows.append({"sample_id": answer_id, "encoding_family": answer.get("metadata", {}).get("encoding_family", ""), 
                                    "encoding_classification_accuracy": eval_report.get("encoding_classification_accuracy", ""), 
                                    "decoder_python_parse_success": eval_report.get("decoder_python_parse_success", ""), 
                                    "decoder_runtime_success": eval_report.get("decoder_runtime_success", ""), 
                                    "json_parse_success": eval_report.get("valid_json_output", ""), 
                                    "field_name_f1": eval_report.get("field_scores", {}).get("field_name_f1", "") if eval_report.get("field_scores") else 0.0, 
                                    "field_value_accuracy": value_accuracy, 
                                    "hallucinated_field_rate": eval_report.get("field_scores", {}).get("hallucination_penalty", "") if eval_report.get("field_scores") else 0.0, 
                                    "manual_score": 0, "automatic_score": eval_report.get("automatic_score", ""),
                                    "ood_success": 1 if eval_report.get("decoder_runtime_success", "") and eval_report.get("decoder_python_parse_success", "") and value_accuracy > 0.8 else 0
                                    })
    save_general_report_csv(report_rows, output_file)

def create_ood_report(results_file, results_rt_file, dataset_path, ontology_path, output_file):
    results = load_jsonl(results_file)
    results_rt = load_jsonl(results_rt_file)
    results_rt_by_id = {item.get("id", ""): item for item in results_rt}
    dataset_index = load_dataset_samples(dataset_path)

    report_rows = []   

    for answer in results:
        answer_id = answer.get("id", "")
        if answer_id not in results_rt_by_id :
            if answer.get("task_type") == "decoder_synthesis":
                print(f"[WARN] Sample {answer_id}: no corresponding retrieval template evaluation found.")
                continue
            continue
        answer_rt = results_rt_by_id[answer_id]
        
        difficulty = answer.get("metadata", {}).get("difficulty", "unknown")
        task = answer.get("task_type", "")
        eval_report = {}
        eval_report_rt = {}
        
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
                eval_report_rt = rt_mark_decoder_synthesis(
                    answer_rt,
                    generalization_samples=gen_samples or None,
                    blob_hex=blob_hex,
                    expected_output=expected_output,
                    ontology_path=ontology_path
                )
                value_accuracy_without_template = eval_report.get("field_scores", {}).get("field_value_accuracy", "") if eval_report.get("field_scores") else 0.0
                value_accuracy_with_template = eval_report_rt.get("field_scores", {}).get("field_value_accuracy", "") if eval_report_rt.get("field_scores") else 0.0
                decoder_runs_without_template = eval_report.get("decoder_runtime_success", "") if eval_report.get("decoder_runtime_success") else 0
                decoder_runs_with_template = eval_report_rt.get("decoder_runtime_success", "") if eval_report_rt.get("decoder_runtime_success") else 0
                encoding_prediction_without_template = answer.get("predicted_json", {}).get("classification", "") if answer.get("predicted_json") else "unknown"
                encoding_prediction_with_template = answer_rt.get("predicted_json", {}).get("classification", "") if answer_rt.get("predicted_json") else "unknown"
                ood_encodings=["lz4_json", "msgpack_blob", "nskeyedarchiver_plist", ]
                seen_or_ood = "ood_encoding" if answer.get("metadata", {}).get("encoding_family", "") in ood_encodings else "seen"
                
                report_rows.append({"sample_id": answer_id, "encoding_family": answer.get("metadata", {}).get("encoding_family", ""), 
                                    "seen_or_ood": seen_or_ood, "retrieved_template_id": answer.get("metadata", {}).get("encoding_family", ""), 
                                    "template_rank": 1, "encoding_prediction_without_template": encoding_prediction_without_template,
                                    "encoding_prediction_with_template": encoding_prediction_with_template, "decoder_runs_without_template": decoder_runs_without_template, 
                                    "decoder_runs_with_template": decoder_runs_with_template, "json_valid_without_template": eval_report.get("valid_json_output", ""), 
                                    "json_valid_with_template": eval_report_rt.get("valid_json_output", ""), "value_accuracy_without_template": value_accuracy_without_template, 
                                    "value_accuracy_with_template": value_accuracy_with_template, "manual_score_without_template": 0, 
                                    "manual_score_with_template": 0, "automatic_score_without_template": eval_report.get("automatic_score", ""), 
                                    "automatic_score_with_template": eval_report_rt.get("automatic_score", "")})

        # if answer["task_type"] == "decoder_synthesis" and difficulty == "hard_negative_control":
        #     eval_report = mark_decoder_synthesis_undecodable(answer)
        #     report_rows.append({})
            
    save_ood_report_csv(report_rows, output_file)
            
def average_general_report(report_file):
    rows = load_csv(report_file)
    if not rows:
        print(f"[WARN] No rows found in {report_file}")
        return
    avg_row = defaultdict(float)
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                avg_row[k] += v
    num_rows = len(rows)
    for k in avg_row:
        avg_row[k] /= num_rows
    avg_row["sample_id"] = "average"
    avg_row["encoding_family"] = "average"
    for k, v in avg_row.items():
        print(f"{k}: {v}")