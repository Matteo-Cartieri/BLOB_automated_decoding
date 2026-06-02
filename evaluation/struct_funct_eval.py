import math
import ast
import subprocess
import tempfile
from collections import defaultdict
import textwrap
import os
import json

def mark_decoder_synthesis(answer, generalization_samples, blob_hex, expected_output):
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
        "expected_fields_ok":   False,
        "field_scores":         None,
        "composite_score":      0.0,
        "generalization_score": None,
        "errors":               [],
    }
    # Checks on metadata and classification
    eval_report["classification_ok"] = (predicted.get("classification", "").strip().lower() == expected.get("classification", "").strip().lower())
    eval_report["expected_fields_ok"] = lists_are_equal(predicted.get("expected_fields"), expected.get("expected_fields"))
    eval_report["decodable_ok"] = (predicted.get("is_decodable") == expected.get("is_decodable"))

    code = predicted.get("decoder_code", "")
    if expected.get("is_decodable") is False:
        # if the sample is not decodable, check if the model correctly identified it as non-decodable and predicted the expected fields
        if predicted.get("is_decodable") is False and not code:
            meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields_ok"]) / 2
            eval_report["composite_score"] = round(meta_ok, 4)
            if eval_report["composite_score"] == 0.0:
                return 1
            elif eval_report["composite_score"] < 0.3:
                return 2
            elif eval_report["composite_score"] < 0.6:
                return 3
            elif eval_report["composite_score"] < 0.9:
                return 4
            else:
                return 5
        elif predicted.get("is_decodable") is True:
            eval_report["errors"].append("False Positive: Model claimed sample is decodable but it is not.")
            return 0

    if not code:
        eval_report["errors"].append("The decoder_code field is empty")
        return 0

    code = code.strip()
    if not is_syntactically_valid(code):
        eval_report["errors"].append("SyntaxError in generated code")
        return 0

    eval_report["syntax_ok"] = True

    eval_report["hardcoded"] = is_hardcoded(code)

    # run the code and evaluate execution and output
    run_result = run_decoder(code, blob_hex)
    eval_report["execution_ok"] = run_result["success"]
    eval_report["valid_json_output"] = run_result["success"] and isinstance(run_result["output"], dict)

    if not run_result["success"]:
        eval_report["errors"].append(run_result["error"])
        return 1  

    # evaluate field-level scores on decoder output
    eval_report["field_scores"] = field_level_score(run_result["output"], expected_output)
 
    # generalization score
    gen_score = None
    if generalization_samples:
        gen_results = []
        for sample in generalization_samples:
            r = run_decoder(code, sample["blob_hex"])
            if r["success"] and isinstance(r["output"], dict):
                fs = field_level_score(r["output"], sample["expected_output"])
                gen_results.append(0.5 * fs["field_f1"] + 0.5 * fs["value_accuracy"])
            else:
                gen_results.append(0.0)
        gen_score = sum(gen_results) / len(gen_results)
        eval_report["generalization_score"] = round(gen_score, 4)

    # final score
    f1 = eval_report["field_scores"].get("field_f1", 0.0)
    val_acc = eval_report["field_scores"].get("value_accuracy", 0.0)
    meta_ok = (eval_report["classification_ok"] + eval_report["expected_fields_ok"]) / 2
    hardcode_penalty = 0.3 if eval_report["hardcoded"] else 0.0

    composite = (0.4*val_acc + 0.3*f1 + 0.15*gen_score + 0.15*meta_ok) * (1 - hardcode_penalty)
    
    eval_report["composite_score"] = round(composite, 4)
    # mapping on the 0-5 mark scale
    if composite == 0.0:
        mark = 1   
    elif composite < 0.3:
        mark = 2   
    elif composite < 0.6:
        mark = 3   
    elif composite < 0.9:
        mark = 4   
    else:
        mark = 5   

    return mark

def lists_are_equal(a, b):
    """Compare two lists for equality."""
    if isinstance(a, list) and isinstance(b, list):
        return set(a) == set(b)
    return a == b

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

# Esecuzione del codice del decoder nel contesto globale
blob = bytes.fromhex("{blob_hex}")

# Cerca la funzione usando globals() invece di dir()
fn = None
for fn_name in ("decode_payload", "decode_blob", "decode"):
    if fn_name in globals():
        fn = globals()[fn_name]
        break

if fn is None:
    print(json.dumps({{"error": "no decode function found"}}))
    sys.exit(1)

try:
    result = fn(blob)
    if isinstance(result, dict):
        print(json.dumps(result))
    else:
        # Se non è un dizionario, lo stampiamo come errore o stringa vuota
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
            if isinstance(output, dict) and "__error__" in output:
                return {"success": False, "output": None, "error": output["__error__"]}
            return {"success": True, "output": output, "error": None}
        except json.JSONDecodeError:
            return {
                "success": False,
                "output": None,
                "error": f"invalid JSON output: {proc.stdout[:200]}",
            }
    # if timeout expires, kill the process and return error
    except subprocess.TimeoutExpired:
        return {"success": False, "output": None, "error": "timeout"}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def field_level_score(predicted_output, expected_output):
    """
    Returns precision, recall, f1, value_accuracy of the predicted output compared to the expected output.
    """
    pred_keys = set(predicted_output.keys())
    exp_keys = set(expected_output.keys())
    common = pred_keys & exp_keys

    precision = len(common) / len(pred_keys) if pred_keys else 0.0
    recall = len(common) / len(exp_keys)  if exp_keys  else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    # percentage of correctly predicted values among the expected keys, comparing values with semantic tolerance
    value_acc = (sum(1 for k in common if compare_json_values(predicted_output[k], expected_output[k])) / len(exp_keys) if exp_keys else 0.0)

    return {
        "field_precision": round(precision, 4),
        "field_recall":    round(recall, 4),
        "field_f1":        round(f1, 4),
        "value_accuracy":  round(value_acc, 4),
        "missing_keys":    sorted(exp_keys - pred_keys),
        "extra_keys":      sorted(pred_keys - exp_keys),
    }


def mark_blob_to_json(answer, acc_threshold = 0.7, float_tol=1e-4):
    if not answer.get("parse_ok", False):
        return 0

    predicted = answer["predicted_json"] or {}
    expected = answer["expected"] or {}

    pred_keys = set(answer["predicted_json"].keys())
    exp_keys  = set(answer["expected"].keys())
    mark = 1.0
    common_keys = pred_keys & exp_keys
    missing_keys = exp_keys - pred_keys   
    extra_keys = pred_keys - exp_keys

    field_precision = len(pred_keys & exp_keys) / len(pred_keys) if pred_keys else 0.0
    field_recall = len(pred_keys & exp_keys) / len(exp_keys) if exp_keys else 0.0
    field_f1 = 2 * (field_precision * field_recall) / (field_precision + field_recall) if (field_precision + field_recall) > 0 else 0.0
    # percentage of correctly predicted values among the expected keys, comparing values with semantic tolerance
    correct_values = sum(1 for k in common_keys if compare_json_values(predicted[k], expected[k], float_tol))
    value_accuracy = correct_values / len(exp_keys) if exp_keys else 0.0

    hallucination_penalty = len(extra_keys) / (len(exp_keys) + len(extra_keys)) if (exp_keys or extra_keys) else 0.0

    composite_score = (0.4 * value_accuracy + 0.25 * field_recall+ 0.25 * field_precision+ 0.1 * (1 - hallucination_penalty))

    if composite_score == 0.0:
        mark = 1   
    elif composite_score < 0.3:
        mark = 2   
    elif composite_score < 0.6:
        mark = 3   
    elif composite_score < 0.9:
        mark = 4   
    else:
        mark = 5
    return mark

def compare_json_values(v1, v2, float_tol=1e-4):
    """Semantic comparison between two values with tolerance for floats and normalization."""
    if v1 == v2:
        return True

    # if both are numbers, compare with tolerance
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return math.isclose(float(v1), float(v2), rel_tol=float_tol)

    # if both are strings: light normalization (strip, case-insensitive for timestamps)
    if isinstance(v1, str) and isinstance(v2, str):
        norm = lambda s: s.strip().replace(" ", "T").replace("T", "T", 1)
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