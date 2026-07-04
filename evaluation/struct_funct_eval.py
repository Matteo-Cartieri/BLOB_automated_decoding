import math
import ast
import subprocess
import tempfile
from collections import defaultdict
import textwrap
import os
import json
import csv

def save_report_csv(rows, filename):
    fields = ["task", "difficulty", "syntax_ok", "execution_ok", "valid_json_output", "hardcoded", "classification_ok", "expected_fields", "field_scores", "generalization_score", "errors", "mark", "id"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(serialized)
    return

def load_ontology(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_reverse_index(ontology):
    """
    Builds a reverse index mapping each synonym to its canonical field name and weight.
    """
    index = {}
    for canonical, entry in ontology.items():
        index[canonical] = (canonical, 1.0)
        for synonym, weight in zip(entry["synonyms"], entry["weights"]):
            if synonym not in index or index[synonym][1] < weight:
                index[synonym] = (canonical, weight)
    return index

def is_empty_code(value) -> bool:
    NO_CODE_STRINGS = {"", "none", "null", "n/a", "na", "not applicable", "no code", "no decoder",
                       "not decodable", "unknown - likely encrypted or a custom format.",
                       "none (likely encrypted or proprietary format)"}
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NO_CODE_STRINGS

def mark_decoder_synthesis_undecodable(answer):
    predicted = answer.get("predicted_json") or {}
    expected  = answer.get("expected") or {}

    eval_report = {
        "classification_ok":    False,
        "expected_fields":      0.0,
        "errors":               [],
        "mark":                 0,
    }
    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0, errors=["Parsing failed for the answer"])
    
    # checks on metadata and classification
    eval_report["classification_ok"] = (predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower())
    eval_report["expected_fields"] = lists_similarity(predicted.get("expected_fields", []), expected.get("expected_fields", []))
    
    eval_report["decodable_ok"] = (predicted.get("is_decodable") == expected.get("is_decodable"))

    code = predicted.get("decoder_code", "")
    if expected.get("is_decodable") is False:
        # if the expected sample is not decodable, check if the model identified it as non-decodable and predicted the expected fields
        if predicted.get("is_decodable") is False and is_empty_code(code):
            meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields"]) / 2
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

def mark_decoder_synthesis(answer, generalization_samples, blob_hex, expected_output, ontology_path):
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
        "syntax_ok":            False,
        "execution_ok":         False,
        "valid_json_output":    False,
        "hardcoded":            False,
        "classification_ok":    False,
        "expected_fields":      0.0,
        "field_scores":         None,
        "generalization_score": 0.0,
        "errors":               [],
        "mark":                 0,
    }
    
    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0, errors=["Parsing failed for the answer"])
    
    # checks on metadata and classification
    eval_report["classification_ok"] = (predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower())
    eval_report["expected_fields"] = lists_similarity(predicted.get("expected_fields", []), expected.get("expected_fields", []))
    
    eval_report["decodable_ok"] = (predicted.get("is_decodable") == expected.get("is_decodable"))

    code = predicted.get("decoder_code", "")
    if expected.get("is_decodable") is False:
        # if the expected sample is not decodable, check if the model identified it as non-decodable and predicted the expected fields
        if predicted.get("is_decodable") is False and is_empty_code(code):
            meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields"]) / 2
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

    eval_report["syntax_ok"] = True

    # check for hardcoding patterns in the code
    eval_report["hardcoded"] = is_hardcoded(code)
    if eval_report["hardcoded"]:
        return make_report(eval_report, mark=0, errors=["The code contains hardcoded values"])

    # run the code and evaluate execution and output
    run_result = run_decoder(code, blob_hex)
    eval_report["execution_ok"] = run_result["success"]
    eval_report["valid_json_output"] = run_result["success"] and isinstance(run_result["output"], dict)

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
                gen_results.append(0.2 * fs["field_f1"] + 0.8 * fs["value_accuracy"])
            else:
                gen_results.append(0.0)
        gen_score = sum(gen_results) / len(gen_results)
        eval_report["generalization_score"] = round(gen_score, 4)

    # final score
    f1 = eval_report["field_scores"].get("field_f1", 0.0)
    val_acc = eval_report["field_scores"].get("value_accuracy", 0.0)
    hallucinated = eval_report["field_scores"].get("hallucination_penalty", 0.0)
    meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields"]) / 2
    composite = (0.05 * f1 + 0.8 * val_acc + 0.05 * gen_score + 0.05 * meta_ok + 0.05 * (1 - hallucinated))
    if composite >= 0.85:
        return make_report(eval_report, mark=5)
    elif composite >= 0.65:
        return make_report(eval_report, mark=4)
    elif composite >= 0.45:
        return make_report(eval_report, mark=3)
    else:
        return make_report(eval_report, mark=2)
    
def old_mark_decoder_synthesis(answer, generalization_samples, blob_hex, expected_output, ontology_path):
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
        "syntax_ok":            False,
        "execution_ok":         False,
        "valid_json_output":    False,
        "hardcoded":            False,
        "classification_ok":    False,
        "expected_fields":      0.0,
        "field_scores":         None,
        "generalization_score": 0.0,
        "errors":               [],
        "mark":                 0,
    }
    
    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0)
    
    # checks on metadata and classification
    eval_report["classification_ok"] = (predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower())
    eval_report["expected_fields"] = lists_similarity(predicted.get("expected_fields", []), expected.get("expected_fields", []))
    
    eval_report["decodable_ok"] = (predicted.get("is_decodable") == expected.get("is_decodable"))

    code = predicted.get("decoder_code", "")
    if expected.get("is_decodable") is False:
        # if the expected sample is not decodable, check if the model identified it as non-decodable and predicted the expected fields
        if predicted.get("is_decodable") is False and not code:
            meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields"]) / 2
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
        elif predicted.get("is_decodable") is False and code:
            return make_report(eval_report, mark=0, errors=["Model claimed sample is not decodable but decoder_code is not empty"])
        
    # if code is absent
    if not code:
        return make_report(eval_report, mark=0, errors=["The decoder_code field is empty"])
    
    # if code not syntactically valid
    code = code.strip()
    if not is_syntactically_valid(code):
        return make_report(eval_report, mark=0, errors=["SyntaxError in generated code"])

    eval_report["syntax_ok"] = True

    # check for hardcoding patterns in the code
    eval_report["hardcoded"] = is_hardcoded(code)
    if eval_report["hardcoded"]:
        return make_report(eval_report, mark=0, errors=["The code contains hardcoded values"])

    # run the code and evaluate execution and output
    run_result = run_decoder(code, blob_hex)
    eval_report["execution_ok"] = run_result["success"]
    eval_report["valid_json_output"] = run_result["success"] and isinstance(run_result["output"], dict)

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
                gen_results.append(0.2 * fs["field_f1"] + 0.8 * fs["value_accuracy"])
            else:
                gen_results.append(0.0)
        gen_score = sum(gen_results) / len(gen_results)
        eval_report["generalization_score"] = round(gen_score, 4)

    # final score
    f1 = eval_report["field_scores"].get("field_f1", 0.0)
    val_acc = eval_report["field_scores"].get("value_accuracy", 0.0)
    hallucinated = eval_report["field_scores"].get("hallucination_penalty", 0.0)
    meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields"]) / 2
    composite = (0.4 * f1 + 0.45 * val_acc + 0.05 * gen_score + 0.05 * meta_ok + 0.05 * (1 - hallucinated))

    if composite >= 0.85:
        return make_report(eval_report, mark=5)
    elif composite >= 0.65:
        return make_report(eval_report, mark=4)
    elif composite >= 0.45:
        return make_report(eval_report, mark=3)
    else:
        return make_report(eval_report, mark=2)

def make_report(eval_report, mark, errors=None):
    eval_report["mark"] = mark
    if errors:
        eval_report["errors"].extend(errors)
    return eval_report

def lists_similarity(a, b):
    """Return the percentage of common elements between two lists."""
    a = a or []
    b = b or []
    if not a and not b:
        return 1.0
    common = set(a) & set(b)
    total = set(a) | set(b)
    return len(common) / len(total) if total else 0.0

def is_syntactically_valid(code):
    """Checks if the Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def is_hardcoded(code: str) -> bool:
    """
    Detect if the code contains hardcoded values.
    Suspicious patterns include:
      - return { ... } with only constant values and no reference to input parameters
      - no reference to function arguments in the code
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # looks for return statements that return a dict literal with only constant values
        for child in ast.walk(node):
            if not isinstance(child, ast.Return):
                continue
            if child.value is None:
                continue
            if isinstance(child.value, ast.Dict):
                all_constants = all( isinstance(v, ast.Constant) for v in child.value.values if v is not None)
                if all_constants:
                    return True
    return False

def run_decoder(code, blob_hex, timeout=10):
    """
    Executes the generated decoder code in a subprocess with the given blob_hex as input.
    """
    runner_logic = textwrap.dedent(f"""
import json, sys

blob = bytes.fromhex("{blob_hex}")

fn = None
for fn_name in ("decode_payload", "decode_blob", "decode", "attempt_decode"):
    if fn_name in globals():
        fn = globals()[fn_name]
        break

if fn is None:
    print(json.dumps({{"error": "No decode function found"}}))
    sys.exit(1)

try:
    result = fn(blob)
    if isinstance(result, dict):
        print(json.dumps(result))
    else:
        print(json.dumps({{"error": f"Decoder returned non-dict type: {{type(result)}}"}}))
except Exception as e:
    print(json.dumps({{"error": f"Runtime error during decode: {{str(e)}}"}}))
""")

    full_code = code + "\n" + runner_logic

    # write the full code to a temporary file and execute it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        proc = subprocess.run( 
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # check for execution errors
        if proc.returncode != 0:
            error_msg = proc.stderr.strip() or proc.stdout.strip()
            return {"success": False, "output": None, "error": error_msg}
        try:
            # check if output is valid JSON
            output = json.loads(proc.stdout.strip())
            if isinstance(output, dict) and "__error__" in output or "error" in output:
                return {"success": False, "output": None, "error": output.get("__error__") or output.get("error")}
            return {"success": True, "output": output, "error": None}
        except json.JSONDecodeError:
            return {"success": False, "output": None, "error": f"Invalid JSON output: {proc.stdout[:200]}"}
    # if timeout expires, kill the process and return error
    except subprocess.TimeoutExpired:
        return {"success": False, "output": None, "error": "Timeout expired during code execution"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def match_field(predicted_key, expected_key, reverse_index):
    """
    Returns a score between 0 and 1 indicating how well the predicted field matches the expected field, using the reverse index for synonyms and weights.
    """
    # exact match
    if predicted_key == expected_key:
        return 1.0

    # find canonical forms and weights 
    pred_canonical, pred_weight = reverse_index.get(predicted_key, (predicted_key, 0.0))
    exp_canonical, exp_weight   = reverse_index.get(expected_key,  (expected_key,  0.0))

    if pred_canonical == exp_canonical:
        return round(min(pred_weight, exp_weight), 4)

    return 0.0

def find_best_match(predicted_key, expected_keys, reverse_index, used):
    """
    Finds the best matching expected key for a given predicted key, using the reverse index and avoiding already used expected keys. Returns the best matching expected key and its score.
    """
    best_key, best_score = None, 0.0
    for ek in expected_keys:
        if ek in used:
            continue
        score = match_field(predicted_key, ek, reverse_index)
        if score > best_score:
            best_score = score
            best_key = ek
    return best_key, best_score

# def field_level_score(predicted_output, expected_output):
#     """
#     Returns precision, recall, f1, value_accuracy of the predicted output compared to the expected output.
#     """
#     pred_keys = set(predicted_output.keys())
#     exp_keys = set(expected_output.keys())
#     common = pred_keys & exp_keys

#     precision = len(common) / len(pred_keys) if pred_keys else 0.0
#     recall = len(common) / len(exp_keys)  if exp_keys  else 0.0
#     f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
#     # percentage of correctly predicted values among the expected keys, comparing values with semantic tolerance
#     value_acc = (sum(1 for k in common if compare_json_values(predicted_output[k], expected_output[k])) / len(exp_keys) if exp_keys else 0.0)

#     extra_keys = pred_keys - exp_keys
#     hallucination_penalty = len(extra_keys) / (len(exp_keys) + len(extra_keys)) if (exp_keys or extra_keys) else 0.0

#     return {
#         "field_precision": round(precision, 4),
#         "field_recall": round(recall, 4),
#         "field_f1": round(f1, 4),
#         "value_accuracy": round(value_acc, 4),
#         "hallucination_penalty": round(hallucination_penalty, 4),
#         "missing_keys": sorted(exp_keys - pred_keys),
#         "extra_keys": sorted(pred_keys - exp_keys),
#     }

def field_level_score(predicted_output, expected_output, reverse_index):
    pred_keys = list(predicted_output.keys())
    exp_keys  = list(expected_output.keys())
    exp_set   = set(exp_keys)

    if not exp_keys:
        return {
            "field_precision": 0.0, "field_recall": 0.0,
            "field_f1": 0.0, "value_accuracy": 0.0,
            "hallucination_penalty": 0.0,
            "missing_keys": [], "extra_keys": pred_keys,
            "match_detail": [],
        }

    used_exp = set()
    matches  = []

    for pk in pred_keys:
        best_ek, name_score = find_best_match(pk, exp_set, reverse_index, used_exp)
        if name_score == 0.0:
            matches.append((pk, None, 0.0, False))
            continue

        used_exp.add(best_ek)
        value_ok = compare_json_values(predicted_output[pk], expected_output[best_ek])
        matches.append((pk, best_ek, name_score, value_ok))


    matched_exp = {ek for _, ek, score, _ in matches if ek and score > 0.0}
    missing = sorted(set(exp_keys) - matched_exp)
    extra = sorted(pk for pk, ek, score, _ in matches if ek is None or score == 0.0)

    # fraction of predicted keys that matched expected keys (weighted by name score)
    precision = (sum(s for _, _, s, _ in matches) / len(pred_keys) if pred_keys else 0.0)
    # fraction of expected keys that were matched by predicted keys (weighted by name score)
    recall = (sum(s for _, ek, s, _ in matches if ek is not None) / len(exp_keys)  if exp_keys else 0.0)
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    # fraction of expected keys that were matched and had correct values
    value_accuracy = (sum(s for _, ek, s, v in matches if ek and v) / len(exp_keys) if exp_keys else 0.0)
    # fraction of predicted keys that were not in expected keys
    hallucination_penalty = (len(extra) / (len(exp_keys) + len(extra)) if (exp_keys or extra) else 0.0)

    exact_matches   = sum(1 for _, ek, s, _ in matches if ek and s == 1.0)
    partial_matches = sum(1 for _, ek, s, _ in matches if ek and 0.0 < s < 1.0)
    return {
        "field_precision":       round(precision, 4),
        "field_recall":          round(recall, 4),
        "field_f1":              round(f1, 4),
        "value_accuracy":        round(value_accuracy, 4),
        "hallucination_penalty": round(hallucination_penalty, 4),
        "exact_matches":         exact_matches,
        "partial_matches":       partial_matches,
        "missing_keys":          missing,
        "extra_keys":            extra,
    }

def mark_blob_to_json(answer, ontology_path, float_tol=1e-4):
    eval_report = {
        "field_scores": None,
        "mark": 0,
    }

    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0)

    predicted = answer["predicted_json"] or {}
    expected  = answer["expected"] or {}

    ontology = load_ontology(ontology_path)
    reverse_index = build_reverse_index(ontology)
    eval_report["field_scores"] = field_level_score(predicted, expected, reverse_index)
    fs = eval_report["field_scores"]

    composite = (0.8 * fs["value_accuracy"] + 0.1 * fs["field_f1"] + 0.1 * (1 - fs["hallucination_penalty"]))

    if composite >= 0.85:
        return make_report(eval_report, mark=5)
    elif composite >= 0.65:
        return make_report(eval_report, mark=4)
    elif composite >= 0.45:
        return make_report(eval_report, mark=3)
    elif composite >= 0.25:
        return make_report(eval_report, mark=2)
    else:
        return make_report(eval_report, mark=1)
    
def old_mark_blob_to_json(answer, ontology_path, float_tol=1e-4):
    eval_report = {
        "field_scores": None,
        "mark": 0,
    }

    if not answer.get("parse_ok", False):
        return make_report(eval_report, mark=0)

    predicted = answer["predicted_json"] or {}
    expected  = answer["expected"] or {}

    ontology = load_ontology(ontology_path)
    reverse_index = build_reverse_index(ontology)
    eval_report["field_scores"] = field_level_score(predicted, expected, reverse_index)
    fs = eval_report["field_scores"]

    composite = (0.5 * fs["value_accuracy"] + 0.4 * fs["field_f1"] + 0.1 * (1 - fs["hallucination_penalty"]))

    if composite >= 0.85:
        return make_report(eval_report, mark=5)
    elif composite >= 0.65:
        return make_report(eval_report, mark=4)
    elif composite >= 0.45:
        return make_report(eval_report, mark=3)
    elif composite >= 0.25:
        return make_report(eval_report, mark=2)
    else:
        return make_report(eval_report, mark=1)

def compare_json_values(v1, v2, float_tol=1e-4):
    """Semantic comparison between two values with tolerance for floats and normalization."""
    if v1 == v2:
        return True

    # if both are numbers, compare with tolerance
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return math.isclose(float(v1), float(v2), rel_tol=float_tol)

    # if both are strings: light normalization (strip, case-insensitive for timestamps)
    if isinstance(v1, str) and isinstance(v2, str):
        norm = lambda s: s.strip().replace(" ", "T").lower()
        return norm(v1) == norm(v2)

    # if both are lists: compare element by element
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False
        return all(compare_json_values(a, b, float_tol) for a, b in zip(v1, v2))

    # if both are dictionaries: recursive comparison field by field
    if isinstance(v1, dict) and isinstance(v2, dict):
        if set(v1.keys()) != set(v2.keys()):
            return False
        return all(compare_json_values(v1[k], v2[k], float_tol) for k in v1)

    return False