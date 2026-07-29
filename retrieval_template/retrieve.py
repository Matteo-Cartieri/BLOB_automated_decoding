import yaml
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from execution_feedback_repair.exec_feedback_repair import generate_repair_prompt
import re



def load_templates(templates_dir, embedder):
    """
    Loads all .yaml files and indexes them by encoding_family and computes embeddings 
    for the description of each template for semantic similarity retrieval.
    """
    templates = {}
    descriptions = []
    keys = []

    for path in Path(templates_dir).glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            tpl = yaml.safe_load(f)
        # use encoding_family (or template_id) as the key 
        key = tpl.get("encoding_family") or tpl.get("template_id")
        templates[key] = tpl
        # create a combined description for embedding
        desc_text = f"{tpl.get('encoding_family', '')}: {tpl.get('short_description', '')}"
        descriptions.append(desc_text)
        keys.append(key)

    # compute embeddings for all descriptions 
    if descriptions:
        embeddings = embedder.encode(descriptions)
        for key, emb in zip(keys, embeddings):
            templates[key]["_embedding"] = emb

    return templates

def extract_prompt_signals(question):
    """Extract magic bytes, detected types and parts of the text from the user's prompt."""
    messages = question.get("messages", [])
    content = messages[1].get("content", "") + "\n"

    magic_hits = re.findall(r'"magic_byte_hits":\s*\[?\s*"([^"]+)"', content, re.IGNORECASE)
    
    hex_prefixes = []
    blob_heads = re.findall(r'"blob_hex_head":\s*"([0-9a-fA-F]+)"', content, re.IGNORECASE)
    hex_prefixes.extend(blob_heads)
    common_prefixes_blocks = re.findall(r'"common_prefixes_hex":\s*\[\s*(.*?)\s*\]\s*,\s*"', content, re.DOTALL | re.IGNORECASE)
    for block in common_prefixes_blocks:
        found_hex = re.findall(r'\[\s*"([0-9a-fA-F]+)"', block)
        hex_prefixes.extend(found_hex)

    detected_keywords = set()
    keywords_to_check = [
        "binary_plist",
        "nskeyedarchiver",
        "zlib",
        "protobuf",
        "gzip",
        "xor_compressed",
        "base64",
        "lz4",
        "msgpack",
    ]
    for kw in keywords_to_check:
        if kw in content.lower():
            detected_keywords.add(kw)

    return {
        "full_text": content,
        "magic_hits": [m.lower() for m in magic_hits],
        "hex_prefixes": [p.upper() for p in hex_prefixes],
        "keywords": detected_keywords,
    }

def match_by_keyword(question, templates):
    """Match the question to a template based on keywords in the prompt or metadata."""
    signals = extract_prompt_signals(question)

    candidate_family = question.get("metadata", {}).get("encoding_family", "").lower()
    prompt_keywords = signals["keywords"]

    for tpl in templates.values():
        family = tpl.get("encoding_family", "").lower()
        template_id = tpl.get("template_id", "").lower()

        if candidate_family and (candidate_family == family or candidate_family == template_id):
            return tpl

        if family in prompt_keywords or template_id in prompt_keywords:
            return tpl

    return None

def match_by_magic_bytes(question, templates):
    """Match the question to a template based on magic bytes or hex prefixes found in the prompt."""
    signals = extract_prompt_signals(question)

    sample_magic_hits = signals["magic_hits"]
    sample_hex = signals["hex_prefixes"]

    for tpl in templates.values():
        signatures = tpl.get("typical_magic_bytes_or_signatures", [])
        for sig in signatures:
            clean_sig = (
                sig.replace("0x", "")
                .replace(" ", "")
                .replace(":", "")
                .upper()
            )

            for hex_head in sample_hex:
                if clean_sig in hex_head or hex_head.startswith(clean_sig):
                    return tpl

            for hit in sample_magic_hits:
                if (hit in tpl.get("template_id", "").lower() or hit in tpl.get("encoding_family", "").lower()):
                    return tpl

    return None

def match_by_semantic_similarity(question, templates, embedder, top_k=1):
    """Match the question to templates based on semantic similarity of the prompt to template descriptions."""
    signals = extract_prompt_signals(question)
    prompt_text = signals["full_text"]

    if not prompt_text or not templates:
        return []

    query_emb = embedder.encode([prompt_text]) # prompt_text[:1000]

    scored_templates = []
    for tpl in templates.values():
        if "_embedding" not in tpl:
            continue
        tpl_emb = tpl["_embedding"].reshape(1, -1)
        score = float(cosine_similarity(query_emb, tpl_emb)[0][0])

        scored_templates.append((score, tpl))

    scored_templates.sort(key=lambda x: x[0], reverse=True)
    return [tpl for score, tpl in scored_templates[:top_k]]

def format_template_context(template):
    """Format the template information into a string to include in the prompt."""
    if template is None:
        return None
    magic = template.get("typical_magic_bytes_or_signatures", [])
    magic_str = ", ".join(magic) if isinstance(magic, list) else str(magic)

    libs = template.get("required_python_libraries")
    libs_str = ", ".join(libs) if isinstance(libs, list) else str(libs)

    failures = template.get("common_failure_conditions", [])
    failures_str = "\n".join(f"- {c}" for c in failures)

    snippet = template["minimal_decoder_snippet"].strip() if template.get("minimal_decoder_snippet") else "null"

    return (
        f"Template ID: {template['template_id']}\n"
        f"Encoding family: {template['encoding_family']}\n"
        f"Description: {template['short_description']}\n"
        f"Typical magic bytes/signatures: {magic_str}\n"
        f"Required Python libraries: {libs_str}\n"
        f"Minimal decoder snippet:\n```python\n{snippet}\n```\n"
        f"Expected output type: {template.get('expected_output_type')}\n"
        f"Common failure conditions:\n{failures_str}\n"
    )

def augment_prompt(question, templates, method, embedder=None, top_k=1):
    """Augment the question prompt with template information based on the specified retrieval method."""
    if method == "keyword":
        tpl = format_template_context(match_by_keyword(question, templates))
    elif method == "magic_bytes":
        tpl = format_template_context(match_by_magic_bytes(question, templates))
    elif method == "semantic_similarity":
        top_templates = match_by_semantic_similarity(question, templates, embedder, top_k)
        tpl = "\n\n".join(format_template_context(t) for t in top_templates) if top_templates else None
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    if tpl is None:
        return question  

    original_prompt = question.get("prompt", "") 

    augmented = dict(question)  
    messages = [dict(m) for m in question["messages"]]

    original_content = messages[1]["content"]
    messages[1] = {
        "role": "user",
        "content": f"{tpl}\n### Task\n{original_content}\n",
    }

    augmented["messages"] = messages
    return augmented

def augment_repair_prompt(question, answer, report, feedback, templates, method, embedder=None, top_k=1):
    repair_question = generate_repair_prompt(question, answer, report)
    augmented_question = augment_prompt(repair_question, templates, method, embedder, top_k)
    
    return augmented_question