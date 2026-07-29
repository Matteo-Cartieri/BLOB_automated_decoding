def generate_failure_feedback(first_answer, report_row, last_repairs):
    """
    Generate feedback for a failed execution.
    """
    error_type = get_error_type(report_row,report_row.get("errors", {}))
    error_message_short = report_row.get("errors", "N/A")
    decoder_stage = get_decoder_stage(error_type)
    observed_output = report_row.get("output", None)
    expected_fields = first_answer.get("expected", {}).get("expected_fields", [])
    failed_checks = ""
    if len(last_repairs) == 0:
        failed_checks += f"No previous repair attempts.\n"
    else:
        for i, repair in enumerate(last_repairs):
            error_type = get_error_type(repair,repair.get("errors", {}))
            failed_checks += f"Error after repair {i + 1}: {error_type}\n"

    feedback_message = f"""error_type:
        {error_type}

    error_message_short:
        {error_message_short}

    decoder_stage:
        {decoder_stage}

    observed_output:
        {observed_output if observed_output else "(None)"}

    expected_fields:
        {", ".join(expected_fields) or "(None)"}
    
    failed_checks:
        {failed_checks}
    """
    return feedback_message

def get_error_type(report_row, errors):
    """Determine the type of error based on the report row and error messages."""
    errors_str = " ".join(str(e.lower()) for e in errors)
    if "syntaxerror" in errors:
        return "syntax_error"
    elif "timeout" in errors:
        return "timeout"
    elif "non-dict type" in errors:
        return "wrong_output_type"
    elif "invalid json" in errors:
        return "json_parse_error"
    elif "modulenotfounderror" in errors:
        return "import_error"
    elif report_row.get("field_scores", {}) and len(report_row.get("field_scores", {}).get("missing_keys", [])) > 0:
        return "missing_fields"
    elif report_row.get("field_scores", {}) and len(report_row.get("field_scores", {}).get("extra_keys", [])) > 0:
        return "hallucinated_fields"
    elif report_row.get("field_value_accuracy", 0.0) < 1.0:
        return "wrong_values"
    else:
        return "runtime_exception"
    
def get_decoder_stage(error_type):
    """Determine the stage of the decoder based on the error type."""
    if error_type == "syntax_error":
        return "parse"
    elif error_type == "import_error":
        return "import"
    elif error_type in ["runtime_exception", "timeout"]:
        return "execution"
    elif error_type in ["wrong_output_type", "json_parse_error"]:
        return "output_parsing"
    elif error_type in ["missing_fields", "hallucinated_fields", "wrong_values"]:
        return "structural_evaluation"
    else:
        return "unknown"
    
def generate_repair_prompt(decoder_code, feedback_message):
    """
    Generate a repair prompt for the model based on the feedback message.

    Returns:
        str: Repair prompt for the model.
    """
    repair_prompt = f"""The following Python decoder was generated for a binary BLOB, but it failed during execution or validation.

            Decoder:
            {decoder_code}
            
            Failure report:
            {feedback_message}
            
            Please produce a corrected Python decoder. Keep the same function signature. Do not invent fields. Do not use unsafe operations. Return only the corrected decoder code.
        """
    return repair_prompt

def repair_prompt(question, answer, report, last_repairs=[]):
    """Augment the question prompt for the model with the feedback message."""
    feedback_message = generate_failure_feedback(answer, report, last_repairs)
    repaired_prompt = generate_repair_prompt(answer.get("predicted_json", {}).get("decoder_code", ""), feedback_message)
    
    original_prompt = question.get("prompt", "") 

    repaired = dict(question)  
    messages = [dict(m) for m in question["messages"]]

    original_content = messages[1]["content"]
    messages[1] = {
        "role": "user",
        "content": f"Task\n{original_content}\n\nExecution feedback:\n{repaired_prompt}\n",
    }

    repaired["messages"] = messages
    return repaired