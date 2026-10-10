"""Source-backed claims and response actions shared by grounding, writing and review."""
import json
import re


ACTIONS = ("answer_factual", "answer_behavioural", "decline_demonstration", "conversational")
STATUSES = ("documented", "not_documented")

CLAIM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        **{name: {"type": "STRING"} for name in (
            "source_quote", "statement", "answer_part",
        )},
        "status": {"type": "STRING", "enum": list(STATUSES)},
    },
    "required": ["source_quote", "statement", "status", "answer_part"],
}
PART_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "request": {"type": "STRING"},
        "action": {"type": "STRING", "enum": list(ACTIONS)},
    },
    "required": ["request", "action"],
}


def evidence_sources(context: dict) -> list[dict]:
    """Label already-selected material; never retrieve or call a model here.

    Topic headings are metadata, not quotable evidence. Conversation/summary and
    interviewer questions resolve references but cannot establish candidate facts.
    """
    sources = []
    doc = context.get("doc_reference_section") or ""
    topics = context.get("doc_topic_list") or []
    if topics:
        markers = re.compile(r"(?m)^(?:" + "|".join(re.escape(t) for t in topics) + r"):\n")
        matches = list(markers.finditer(doc))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
            content = doc[match.end():end].strip()
            if content:
                sources.append({"id": f"doc-{len(sources) + 1}", "kind": "candidate",
                                "label": match.group().strip()[:-1], "content": content})
    elif doc.strip() and doc != "No document reference available.":
        sources.append({"id": "doc-1", "kind": "candidate", "content": doc.strip()})
    for i, example in enumerate(context.get("similar_examples") or []):
        if isinstance(example.get("answer"), str) and example["answer"].strip():
            sources.append({"id": f"example-{i + 1}", "kind": "candidate",
                            "content": example["answer"]})
    identity = context.get("candidate_identity")
    if isinstance(identity, str) and identity.strip():
        sources.append({"id": "identity", "kind": "candidate", "content": identity})
    role = context.get("job_context")
    if isinstance(role, str) and role.strip():
        sources.append({"id": "offered-role", "kind": "offered_role", "content": role.strip()})
    return sources


class GroundingValidationError(ValueError):
    """Fixed diagnostic code and message; never includes model or source content."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def validate_grounding(value: object) -> dict:
    """Check response structure, not textual or semantic agreement.

    Grounding is prompted to preserve source wording. The AI response reviewer
    checks the reply against grounding; Python does not compare excerpts.
    """
    if not isinstance(value, dict):
        raise GroundingValidationError("not_object", "Grounding must be an object")
    if value.get("question_type") not in ("factual", "behavioural", "conversational"):
        raise GroundingValidationError("question_type", "Invalid question type")
    if value.get("coverage") not in ("full", "partial", "none") or not isinstance(value.get("missing"), str):
        raise GroundingValidationError("coverage", "Invalid coverage")
    parts = value.get("response_parts")
    if not isinstance(parts, list) or not parts or any(
        not isinstance(p, dict) or not isinstance(p.get("request"), str) or not p["request"].strip()
        or not isinstance(p.get("action"), str) or p["action"] not in ACTIONS for p in parts
    ):
        raise GroundingValidationError("response_actions", "Invalid response actions")
    claims = value.get("claims")
    if not isinstance(claims, list):
        raise GroundingValidationError("claims_not_list", "Claims must be a list")
    checked = []
    for claim in claims:
        if not isinstance(claim, dict) or any(not isinstance(claim.get(k), str) for k in CLAIM_SCHEMA["required"]):
            raise GroundingValidationError("claim_shape", "Malformed claim")
        if not claim["source_quote"].strip():
            raise GroundingValidationError("empty_source_quote", "Claim has no supporting excerpt")
        if not claim["statement"].strip():
            raise GroundingValidationError("empty_statement", "Claim has no interpretation")
        if not claim["answer_part"].strip():
            raise GroundingValidationError("claim_answer_part", "Claim has no answer-part description")
        if claim["status"] not in STATUSES:
            raise GroundingValidationError("claim_status", "Invalid evidence status")
        checked.append({key: claim[key] for key in CLAIM_SCHEMA["required"]})
    # Compatibility projection for existing orchestration; writer/reviewer use
    # the richer claims, never this lossy view when a contract is present.
    return {"question_type": value["question_type"], "coverage": value["coverage"],
            "missing": value["missing"], "response_parts": parts, "claims": checked,
            "facts": [c["source_quote"] for c in checked]}


def render_contract(grounding: dict) -> str:
    return json.dumps({key: grounding.get(key) for key in (
        "response_parts", "claims", "coverage", "missing",
    )}, ensure_ascii=False)
