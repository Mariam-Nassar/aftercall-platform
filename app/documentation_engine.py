"""
app/documentation_engine.py

Documentation generation module for the AI-Powered After-Call
Automation Platform.

Responsibility (and ONLY responsibility):
    Generate ONE structured CRM documentation record (a
    DocumentationRecord) from the outputs of the intake, customer
    lookup, and handbook search modules, using Google Gemini as a
    strictly grounded generator.

This module does NOT:
    - search the handbook (consumes already-retrieved context)
    - look up customers (consumes an already-loaded Customer)
    - read transcript files (consumes an already-loaded TranscriptInput)
    - decide whether the record should be saved
    - decide or perform escalation (only recommends, via a boolean flag)
    - review, approve, or reject records
    - persist, route, or display anything

Grounding policy:
    The handbook context takes priority over the model's own knowledge.
    The transcript takes priority over assumptions. Customer data takes
    priority over guessing. Any field the model cannot support with
    evidence from the transcript, customer record, or handbook context
    must be returned as "Not stated" rather than invented.

Runtime requirements:
    - Environment variable GOOGLE_API_KEY (or an explicit api_key
      argument) must be available to authenticate with Gemini.
    - Package: google-generativeai.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai

from app.customer_lookup import Customer
from app.intake import TranscriptInput

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "gemini-3.6-flash"
DEFAULT_TEMPERATURE = 0.0

_STRING_FIELDS = (
    "summary",
    "issue",
    "root_cause",
    "resolution",
    "category",
    "subcategory",
    "priority",
    "disposition",
    "sentiment",
)
_LIST_FIELDS = ("pending_actions", "keywords", "tags")
_NOT_STATED = "Not stated"

_REQUIRED_FIELDS = (
    *_STRING_FIELDS,
    *_LIST_FIELDS,
    "escalation_recommended",
    "confidence",
    "grounding",
)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class LLMResponseError(Exception):
    """Raised when the LLM call fails or returns no usable content."""


class InvalidJSONError(Exception):
    """Raised when the LLM response cannot be parsed as JSON."""


class ValidationError(Exception):
    """Raised when the parsed output cannot be validated into a DocumentationRecord."""


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentationRecord:
    """
    A single, grounded CRM documentation draft.

    Attributes:
        summary: Short summary of the call, at most three sentences.
        issue: The customer's actual problem, as stated in the
            transcript.
        root_cause: The identified root cause, or "Not stated" if not
            supported by the transcript.
        resolution: What actually happened on the call, or
            "Not stated" if unresolved.
        pending_actions: Follow-up actions still pending; empty list
            if none.
        category: Top-level category, drawn from handbook retrieval.
        subcategory: Subcategory, drawn from handbook retrieval.
        priority: Priority level, drawn from handbook retrieval.
        disposition: Call disposition, drawn from handbook retrieval.
        sentiment: Customer sentiment, drawn from handbook retrieval.
        keywords: Important words/phrases extracted from the
            transcript.
        tags: Free-form labels useful for downstream filtering.
        escalation_recommended: Whether escalation is recommended.
            This is a recommendation only; final escalation decisions
            belong to business_rules.py.
        confidence: Model's confidence in this record, between 0 and 1.
        grounding: Mapping of field name to the list of evidence
            references (e.g. transcript lines, handbook rules) that
            support it.
    """

    summary: str
    issue: str
    root_cause: str
    resolution: str
    pending_actions: list[str]
    category: str
    subcategory: str
    priority: str
    disposition: str
    sentiment: str
    keywords: list[str]
    tags: list[str]
    escalation_recommended: bool
    confidence: float
    grounding: dict[str, list[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def _format_handbook_context(handbook_context: list[dict[str, Any]]) -> str:
    """
    Render retrieved handbook chunks into a readable, labeled block.

    Args:
        handbook_context: List of handbook retrieval results, each
            expected to have "source" and "content" keys.

    Returns:
        A single string listing each chunk with its source, or a
        placeholder line if no context was retrieved.
    """
    if not handbook_context:
        return "No handbook context was retrieved."

    lines: list[str] = []
    for index, chunk in enumerate(handbook_context, start=1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        lines.append(f"[Handbook Rule {index} | source: {source}]\n{content}")
    return "\n\n".join(lines)


def _format_customer(customer: Customer) -> str:
    """
    Render a Customer record into a readable, labeled block.

    Args:
        customer: The customer associated with this call.

    Returns:
        A single string describing the known customer fields.
    """
    fields = {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "plan": customer.plan,
    }
    lines = [f"{key}: {value}" for key, value in fields.items() if value is not None]
    return "\n".join(lines) if lines else "No customer fields available."


def _numbered_transcript(transcript_text: str) -> str:
    """
    Prefix each non-empty transcript line with a stable line number.

    Numbering gives the model a concrete evidence reference (e.g.
    "Transcript Line 4") to cite in the grounding output instead of
    quoting or paraphrasing loosely.

    Args:
        transcript_text: Raw transcript text.

    Returns:
        The transcript with each line prefixed by "Line N: ".
    """
    numbered_lines = []
    for index, line in enumerate(transcript_text.splitlines(), start=1):
        numbered_lines.append(f"Line {index}: {line}")
    return "\n".join(numbered_lines)


def _build_system_instruction() -> str:
    """
    Build the system-level instruction establishing grounding rules.

    Returns:
        A system instruction string enforcing strict grounding and
        JSON-only output.
    """
    return (
        "You are a documentation assistant generating CRM after-call "
        "documentation for a customer service platform. You are "
        "forbidden from inventing information. You must ground every "
        "field you produce in the evidence provided to you.\n\n"
        "Priority of evidence, from highest to lowest:\n"
        "1. The documentation handbook context provided to you takes "
        "priority over your own general knowledge.\n"
        "2. The call transcript takes priority over any assumption you "
        "might otherwise make.\n"
        "3. The customer record takes priority over guessing.\n\n"
        "Two different kinds of fields, two different rules:\n\n"
        "FACTUAL fields (root_cause, resolution, pending_actions) describe "
        "what actually happened. If the transcript does not support one of "
        f'these, return the exact string "{_NOT_STATED}" for it. Never '
        "fabricate order numbers, refunds, promises, resolutions, "
        "follow-up dates, agent actions, ticket numbers, or customer "
        "information.\n\n"
        "CLASSIFICATION fields (category, subcategory, priority, "
        "disposition, sentiment) are judgments you make about the call, "
        "not facts you extract from it. If the handbook context contains "
        "any relevant taxonomy, priority scale, disposition list, or "
        "sentiment rubric, you MUST pick the single best-matching value "
        "from that handbook context — copy the exact label text as it "
        "appears there, do not paraphrase or shorten it. Do not return "
        f'"{_NOT_STATED}" for a classification field just because the '
        "call is ambiguous, partial, or unresolved: an unresolved or "
        "identity-unverified call still has a category, a priority, and "
        f'a disposition. Only use "{_NOT_STATED}" for a classification '
        "field when the retrieved handbook context truly contains no "
        "applicable value for it.\n\n"
        "You must respond with ONLY valid JSON. No markdown code "
        "fences, no explanations, no text outside the JSON object."
    )


def build_prompt(
    transcript: TranscriptInput,
    customer: Customer,
    handbook_context: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Build the system instruction and user prompt for documentation generation.

    Args:
        transcript: The loaded call transcript.
        customer: The customer associated with the call.
        handbook_context: Handbook chunks retrieved for this call.

    Returns:
        A tuple of (system_instruction, user_prompt).
    """
    numbered_transcript = _numbered_transcript(transcript.transcript_text)
    customer_block = _format_customer(customer)
    handbook_block = _format_handbook_context(handbook_context)

    schema_description = (
        "Return a JSON object with exactly these keys:\n"
        '"summary": string, at most three sentences.\n'
        '"issue": string, the customer\'s actual problem.\n'
        f'"root_cause": string, or "{_NOT_STATED}" if not supported.\n'
        f'"resolution": string, or "{_NOT_STATED}" if unresolved.\n'
        '"pending_actions": list of strings, empty list if none.\n'
        '"category": string, the exact label copied from the handbook '
        "taxonomy in the context above — not a paraphrase.\n"
        '"subcategory": string, the exact label copied from the handbook '
        "taxonomy in the context above — not a paraphrase.\n"
        '"priority": string, the exact priority level name used in the '
        "handbook context above.\n"
        '"disposition": string, the exact disposition name used in the '
        "handbook context above.\n"
        '"sentiment": string, the exact sentiment label used in the '
        "handbook context above.\n"
        '"keywords": list of important words/phrases from the transcript.\n'
        '"tags": list of useful free-form labels.\n'
        '"escalation_recommended": boolean, a recommendation only.\n'
        '"confidence": float between 0 and 1.\n'
        '"grounding": an object mapping each of the field names above '
        "to a list of evidence references (e.g. \"Transcript Line 4\", "
        "\"Handbook Rule 2\") that support the value you produced for "
        "that field."
    )

    user_prompt = (
        f"CALL TRANSCRIPT:\n{numbered_transcript}\n\n"
        f"CUSTOMER RECORD:\n{customer_block}\n\n"
        f"HANDBOOK CONTEXT:\n{handbook_block}\n\n"
        f"{schema_description}"
    )

    logger.info(
        "Prompt built for call_id=%s customer_id=%s (%d handbook chunk(s))",
        transcript.call_id,
        customer.customer_id,
        len(handbook_context),
    )
    return _build_system_instruction(), user_prompt


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------


def call_llm(
    system_instruction: str,
    user_prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Call Google Gemini and return its raw text response.

    Args:
        system_instruction: System-level grounding instructions.
        user_prompt: The user-role prompt containing transcript,
            customer, and handbook context.
        model_name: Name of the Gemini model to use.
        api_key: Optional explicit API key. If not provided, the
            google.generativeai client must already be configured
            (e.g. via the GOOGLE_API_KEY environment variable).
        temperature: Sampling temperature; kept low for deterministic
            documentation output.

    Returns:
        The raw text of the model's response (expected to be a JSON
        string).

    Raises:
        LLMResponseError: If the API call fails or returns no content.
    """
    try:
        # genai.configure() must always be called at least once: when
        # api_key is None, it performs its own GEMINI_API_KEY /
        # GOOGLE_API_KEY environment-variable lookup internally. If we
        # skip calling it entirely, the client falls back to
        # Application Default Credentials and fails even when the
        # environment variable is set.
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )

        logger.info("Sending documentation request to model=%s", model_name)
        response = model.generate_content(user_prompt)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise LLMResponseError(f"LLM request failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise LLMResponseError("LLM returned an empty response.")

    logger.info("Received LLM response (%d characters)", len(text))
    return text


# --------------------------------------------------------------------------
# JSON parsing
# --------------------------------------------------------------------------


def parse_json(raw_text: str) -> dict[str, Any]:
    """
    Parse the LLM's raw text response into a JSON object.

    Defensively strips Markdown code fences in case the model wraps
    its JSON despite instructions not to.

    Args:
        raw_text: Raw text returned by call_llm().

    Returns:
        The parsed JSON object as a dictionary.

    Raises:
        InvalidJSONError: If the text cannot be parsed as a JSON
            object.
    """
    cleaned = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidJSONError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise InvalidJSONError("LLM response JSON must be an object.")

    logger.info("Parsed LLM JSON response with %d top-level key(s)", len(data))
    return data


# --------------------------------------------------------------------------
# Output validation
# --------------------------------------------------------------------------


def _coerce_string_field(data: dict[str, Any], field_name: str) -> str:
    """
    Coerce a single string field, falling back to "Not stated" when unsupported.

    Args:
        data: The raw parsed JSON output.
        field_name: Name of the field to coerce.

    Returns:
        A string value: the original if valid and non-empty, otherwise
        "Not stated".
    """
    value = data.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()

    logger.warning("Field '%s' missing or invalid; defaulting to '%s'", field_name, _NOT_STATED)
    return _NOT_STATED


def _coerce_list_field(data: dict[str, Any], field_name: str) -> list[str]:
    """
    Coerce a single list-of-strings field, defaulting to an empty list.

    Args:
        data: The raw parsed JSON output.
        field_name: Name of the field to coerce.

    Returns:
        A list of strings, filtering out any non-string items.
    """
    value = data.get(field_name)
    if not isinstance(value, list):
        if value is not None:
            logger.warning("Field '%s' is not a list; defaulting to []", field_name)
        return []

    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _coerce_grounding(data: dict[str, Any]) -> dict[str, list[str]]:
    """
    Coerce the grounding field into a mapping of field name to evidence list.

    Args:
        data: The raw parsed JSON output.

    Returns:
        A dictionary mapping field names to lists of evidence strings.
        Entries with an invalid shape are dropped rather than causing
        validation to fail outright.
    """
    raw_grounding = data.get("grounding")
    if not isinstance(raw_grounding, dict):
        logger.warning("Field 'grounding' missing or invalid; defaulting to {}")
        return {}

    grounding: dict[str, list[str]] = {}
    for key, value in raw_grounding.items():
        if isinstance(value, list):
            grounding[key] = [str(item) for item in value if isinstance(item, str)]
        elif isinstance(value, str) and value.strip():
            grounding[key] = [value.strip()]
        else:
            logger.warning("Grounding entry for '%s' has an invalid shape; skipping", key)

    return grounding


def _coerce_escalation_recommended(data: dict[str, Any]) -> bool:
    """
    Coerce the escalation_recommended field to a strict boolean.

    Args:
        data: The raw parsed JSON output.

    Returns:
        The boolean value.

    Raises:
        ValidationError: If the field is missing or not a genuine
            boolean (or an unambiguous boolean-like string).
    """
    value = data.get("escalation_recommended")
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"

    raise ValidationError(
        "Field 'escalation_recommended' must be a boolean, got: "
        f"{value!r}"
    )


def _coerce_confidence(data: dict[str, Any]) -> float:
    """
    Coerce the confidence field to a float clamped to [0, 1].

    Args:
        data: The raw parsed JSON output.

    Returns:
        The confidence value, clamped to the valid range.

    Raises:
        ValidationError: If the field is missing or not numeric.
    """
    value = data.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Field 'confidence' must be a number, got: {value!r}")

    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        logger.warning("Confidence %.3f out of range; clamping to [0, 1]", confidence)
        confidence = max(0.0, min(1.0, confidence))
    return confidence


def validate_output(data: dict[str, Any]) -> DocumentationRecord:
    """
    Validate and normalize parsed LLM output into a DocumentationRecord.

    Unsupported or malformed textual fields are replaced with
    "Not stated" rather than causing validation to fail, in keeping
    with the strict grounding policy. Structural failures on
    unambiguous fields (escalation_recommended, confidence) raise
    ValidationError since there is no safe default to invent.

    Args:
        data: Parsed JSON object, typically the output of parse_json().

    Returns:
        A fully validated, immutable DocumentationRecord.

    Raises:
        ValidationError: If required structural fields cannot be
            reasonably coerced.
    """
    if not isinstance(data, dict):
        raise ValidationError("Validated input must be a JSON object.")

    string_values = {name: _coerce_string_field(data, name) for name in _STRING_FIELDS}
    list_values = {name: _coerce_list_field(data, name) for name in _LIST_FIELDS}
    escalation_recommended = _coerce_escalation_recommended(data)
    confidence = _coerce_confidence(data)
    grounding = _coerce_grounding(data)

    record = DocumentationRecord(
        summary=string_values["summary"],
        issue=string_values["issue"],
        root_cause=string_values["root_cause"],
        resolution=string_values["resolution"],
        pending_actions=list_values["pending_actions"],
        category=string_values["category"],
        subcategory=string_values["subcategory"],
        priority=string_values["priority"],
        disposition=string_values["disposition"],
        sentiment=string_values["sentiment"],
        keywords=list_values["keywords"],
        tags=list_values["tags"],
        escalation_recommended=escalation_recommended,
        confidence=confidence,
        grounding=grounding,
    )

    logger.info(
        "Validated documentation record: category=%s priority=%s escalation_recommended=%s",
        record.category,
        record.priority,
        record.escalation_recommended,
    )
    return record


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def create_documentation(
    transcript: TranscriptInput,
    customer: Customer,
    handbook_context: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> DocumentationRecord:
    """
    Generate a single grounded CRM documentation record.

    Orchestrates prompt construction, the Gemini call, JSON parsing,
    and validation. Performs no persistence, routing, or review; the
    returned record is the complete output of this module.

    Args:
        transcript: The loaded call transcript.
        customer: The customer associated with the call.
        handbook_context: Handbook chunks retrieved for this call
            (e.g. from handbook_search.retrieve_rules()).
        model_name: Name of the Gemini model to use.
        api_key: Optional explicit API key; falls back to the
            environment configuration used by google.generativeai.

    Returns:
        A fully validated DocumentationRecord.

    Raises:
        LLMResponseError: If the Gemini call fails or returns no
            content.
        InvalidJSONError: If the response cannot be parsed as JSON.
        ValidationError: If the parsed output cannot be validated.
    """
    system_instruction, user_prompt = build_prompt(transcript, customer, handbook_context)
    raw_response = call_llm(system_instruction, user_prompt, model_name=model_name, api_key=api_key)
    parsed = parse_json(raw_response)
    record = validate_output(parsed)

    logger.info(
        "Documentation created for call_id=%s customer_id=%s",
        transcript.call_id,
        customer.customer_id,
    )
    return record
