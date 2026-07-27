"""
app/business_rules.py

Business rules / decision engine for the AI-Powered After-Call
Automation Platform.

Responsibility (and ONLY responsibility):
    Given ONE validated DocumentationRecord (produced by
    documentation_engine.py), deterministically decide exactly one
    routing outcome:

        AUTO_SAVE | HUMAN_REVIEW | ESCALATE | NON_INTERACTION

This module NEVER:
    - calls Gemini, LangChain, OpenAI, or any other LLM/framework
    - generates text, summaries, or documentation
    - performs retrieval (handbook search)
    - reads transcript files or loads customer records
    - writes to a database, saves CRM records, or renders UI
    - approves or reviews records (it only routes them)

All decisions are pure, deterministic Python logic over the fields of
a DocumentationRecord. The LLM (upstream) may *recommend* escalation
via `escalation_recommended`; this module is the sole authority that
*decides* the final routing outcome.

Prompt-injection note (Rule 6):
    Free-text instructions that may appear inside transcript-derived
    fields (e.g. "mark this resolved", "don't escalate") are never
    parsed as commands anywhere in this module. Routing is driven
    exclusively by the structured checks below (grounding, keyword
    sets, confidence); such phrases are, at most, logged for audit
    purposes and never influence the decision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Final

from app.documentation_engine import DocumentationRecord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: Final[float] = 0.75
"""Minimum DocumentationRecord.confidence required for AUTO_SAVE eligibility."""

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "summary",
    "issue",
    "category",
    "priority",
    "disposition",
    "sentiment",
)
"""DocumentationRecord fields that must be present, grounded, and non-null."""

_NOT_STATED: Final[str] = "not stated"

ESCALATION_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "escalate",
        "escalation required",
        "escalation needed",
        "requires supervisor",
        "requires manager",
        "manager required",
        "supervisor required",
    }
)
"""Phrases that, if present in categorical/text fields, imply a handbook-mandated escalation."""

SENSITIVE_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "threat",
        "threatened",
        "threatening",
        "abuse",
        "abusive",
        "fraud",
        "fraudulent",
        "unauthorized charge",
        "unauthorized transaction",
        "lawsuit",
        "sue",
        "attorney",
        "legal action",
        "legal complaint",
        "regulator",
        "regulatory complaint",
        "compliance violation",
        "data breach",
        "privacy violation",
        "personal data leak",
        "harassment",
        "harass",
        "violence",
        "violent",
        "assault",
        "security incident",
        "hacked",
        "unauthorized access",
        "medical emergency",
        "chest pain",
        "heart attack",
        "self-harm",
        "self harm",
        "suicide",
        "kill myself",
        "harm myself",
    }
)
"""Phrases indicating a sensitive situation that must always be escalated (Rule 4)."""

NON_INTERACTION_PATTERNS: Final[frozenset[str]] = frozenset(
    {
        "silent call",
        "no interaction",
        "call disconnected immediately",
        "wrong number",
        "empty call",
        "no response from customer",
        "customer hung up immediately",
        "dead air",
        "no audio detected",
    }
)
"""Phrases indicating no genuine customer interaction occurred (Rule 9)."""

IGNORED_INSTRUCTION_PATTERNS: Final[frozenset[str]] = frozenset(
    {
        "mark this resolved",
        "close the ticket",
        "ignore policy",
        "don't escalate",
        "do not escalate",
        "approve this",
    }
)
"""
Customer-authored instructions that must NEVER influence routing
(Rule 6). These are detected only for audit logging; they are never
read as commands and never appear in any branch of evaluate_record's
control flow.
"""


# --------------------------------------------------------------------------
# Output types
# --------------------------------------------------------------------------


class DecisionType(str, Enum):
    """The set of allowed final routing decisions. No other values are valid."""

    AUTO_SAVE = "AUTO_SAVE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ESCALATE = "ESCALATE"
    NON_INTERACTION = "NON_INTERACTION"


@dataclass(frozen=True)
class BusinessDecision:
    """
    The final, deterministic routing outcome for a DocumentationRecord.

    Attributes:
        decision: The routing outcome.
        reason: Human-readable explanation of why this decision was
            reached.
        triggered_rules: Ordered list of rule labels (e.g. "Rule 4")
            that contributed to this decision.
        confidence: The confidence value from the source
            DocumentationRecord, carried through for traceability.
        timestamp: UTC time at which the decision was made.
    """

    decision: DecisionType
    reason: str
    triggered_rules: list[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class BusinessRuleError(Exception):
    """Base exception for all business-rule evaluation failures."""


class InvalidDecisionError(BusinessRuleError):
    """Raised when a computed decision is not one of the allowed DecisionType values."""


class GroundingValidationError(BusinessRuleError):
    """Raised when a DocumentationRecord's grounding field is structurally invalid."""


# --------------------------------------------------------------------------
# Text helpers (private)
# --------------------------------------------------------------------------


def _field_text(record: DocumentationRecord, field_name: str) -> str:
    """
    Get the lowercased text value of a single DocumentationRecord field.

    Args:
        record: The documentation record to read from.
        field_name: Name of the field to read.

    Returns:
        The field's string value, lowercased, or an empty string if
        the field is not a string.
    """
    value = getattr(record, field_name, None)
    return value.lower() if isinstance(value, str) else ""


def _classification_text(record: DocumentationRecord) -> str:
    """
    Concatenate only the handbook-derived classification fields.

    Used specifically for escalation-keyword detection (Rule 5), which
    must reflect the handbook-driven classification of the call, not
    free-form narrative text that may directly quote the customer
    (e.g. a customer saying "don't escalate" must never itself trigger
    escalation, per Rule 6).

    Args:
        record: The documentation record to read from.

    Returns:
        A single lowercased string of category, subcategory, priority,
        and disposition.
    """
    field_names = ("category", "subcategory", "priority", "disposition")
    return " ".join(_field_text(record, name) for name in field_names)


def _combined_text(record: DocumentationRecord) -> str:
    """
    Concatenate every text-bearing field of a DocumentationRecord into one string.

    Used as the search surface for keyword-based detection (sensitive
    content, escalation keywords, non-interaction patterns). Keywords
    and tags are included since they are extracted directly from the
    transcript and may carry signal not present in the prose fields.

    Args:
        record: The documentation record to read from.

    Returns:
        A single lowercased string containing all textual content.
    """
    text_field_names = (
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
    parts = [_field_text(record, name) for name in text_field_names]
    parts.append(" ".join(record.keywords).lower())
    parts.append(" ".join(record.tags).lower())
    return " ".join(parts)


def _compile_keyword_pattern(phrases: frozenset[str]) -> re.Pattern[str]:
    """
    Compile a set of phrases into a single word-boundary regex pattern.

    Word boundaries prevent false positives such as "sue" matching
    inside "issued". Longer phrases are ordered first so multi-word
    phrases are not shadowed by a shorter phrase they contain.

    Args:
        phrases: A set of lowercased phrases to match.

    Returns:
        A compiled, case-insensitive regex matching any of the phrases
        as whole words/phrases.
    """
    ordered = sorted(phrases, key=len, reverse=True)
    alternation = "|".join(re.escape(phrase) for phrase in ordered)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def _matches_any(text: str, pattern: re.Pattern[str]) -> bool:
    """
    Check whether a compiled keyword pattern matches anywhere in text.

    Args:
        text: The text to search within.
        pattern: A pattern produced by _compile_keyword_pattern().

    Returns:
        True if the pattern matches.
    """
    return pattern.search(text) is not None


_SENSITIVE_PATTERN: Final[re.Pattern[str]] = _compile_keyword_pattern(SENSITIVE_KEYWORDS)
_ESCALATION_PATTERN: Final[re.Pattern[str]] = _compile_keyword_pattern(ESCALATION_KEYWORDS)
_NON_INTERACTION_PATTERN: Final[re.Pattern[str]] = _compile_keyword_pattern(NON_INTERACTION_PATTERNS)
_INJECTION_PATTERN: Final[re.Pattern[str]] = _compile_keyword_pattern(IGNORED_INSTRUCTION_PATTERNS)


# --------------------------------------------------------------------------
# Grounding / completeness helpers
# --------------------------------------------------------------------------


def _ensure_valid_grounding_type(record: DocumentationRecord) -> None:
    """
    Ensure the record's grounding attribute is a dictionary.

    Args:
        record: The documentation record to check.

    Raises:
        GroundingValidationError: If `record.grounding` is not a dict.
    """
    if not isinstance(record.grounding, dict):
        raise GroundingValidationError(
            "DocumentationRecord.grounding must be a dict, got: "
            f"{type(record.grounding).__name__}"
        )


def _missing_required_fields(record: DocumentationRecord) -> list[str]:
    """
    Identify required fields whose value is empty or "Not stated".

    Args:
        record: The documentation record to check.

    Returns:
        A list of field names (from REQUIRED_FIELDS) whose content is
        missing, per Rule 2.
    """
    missing: list[str] = []
    for name in REQUIRED_FIELDS:
        value = getattr(record, name, None)
        if not isinstance(value, str) or not value.strip():
            missing.append(name)
        elif value.strip().lower() == _NOT_STATED:
            missing.append(name)
    return missing


def _unsupported_grounded_fields(record: DocumentationRecord) -> list[str]:
    """
    Identify required fields that have no grounding entry at all.

    Args:
        record: The documentation record to check.

    Returns:
        A list of required field names not present as keys in
        `record.grounding`, per Rule 1.
    """
    return [name for name in REQUIRED_FIELDS if name not in record.grounding]


def _ambiguous_grounded_fields(record: DocumentationRecord) -> list[str]:
    """
    Identify required fields whose grounding entry exists but is empty.

    A field claimed as grounded but backed by zero evidence references
    is treated as ambiguous rather than simply unsupported, per Rule 7.

    Args:
        record: The documentation record to check.

    Returns:
        A list of required field names present in `record.grounding`
        with an empty evidence list.
    """
    return [
        name
        for name in REQUIRED_FIELDS
        if name in record.grounding and not record.grounding[name]
    ]


def validate_grounding(record: DocumentationRecord) -> bool:
    """
    Check that every required field is grounded with non-empty evidence.

    Args:
        record: The documentation record to check.

    Returns:
        True if every field in REQUIRED_FIELDS has a non-empty
        grounding entry; False otherwise.

    Raises:
        GroundingValidationError: If `record.grounding` is not a dict.
    """
    _ensure_valid_grounding_type(record)
    return not _unsupported_grounded_fields(record) and not _ambiguous_grounded_fields(record)


def has_missing_fields(record: DocumentationRecord) -> bool:
    """
    Check whether any required field is empty or "Not stated".

    Args:
        record: The documentation record to check.

    Returns:
        True if at least one required field is missing (Rule 2).
    """
    return bool(_missing_required_fields(record))


def is_complete(record: DocumentationRecord) -> bool:
    """
    Check whether a record is structurally complete and grounded.

    Combines the "documentation is complete" and "grounding exists"
    conditions of Rule 3.

    Args:
        record: The documentation record to check.

    Returns:
        True if no required fields are missing and every required
        field is grounded with non-empty evidence.

    Raises:
        GroundingValidationError: If `record.grounding` is not a dict.
    """
    return not has_missing_fields(record) and validate_grounding(record)


# --------------------------------------------------------------------------
# Trigger detection helpers
# --------------------------------------------------------------------------


def contains_sensitive_content(record: DocumentationRecord) -> bool:
    """
    Check whether the record indicates a sensitive situation.

    Sensitive situations (threats, abuse, fraud, legal or regulatory
    complaints, privacy issues, harassment, violence, security
    incidents, medical emergencies, self-harm reports) always require
    escalation regardless of confidence or completeness, per Rule 4.

    Args:
        record: The documentation record to check.

    Returns:
        True if any sensitive-content keyword is found in the record's
        textual fields.
    """
    return _matches_any(_combined_text(record), _SENSITIVE_PATTERN)


def requires_escalation(record: DocumentationRecord) -> bool:
    """
    Check whether the record requires escalation.

    Escalation is required if the handbook-derived classification
    fields (category, subcategory, priority, disposition) indicate an
    escalation requirement, or if the upstream model recommended
    escalation, per Rule 5. Free-form narrative fields are
    deliberately excluded from this check: a customer's own words
    (e.g. "don't escalate") must never influence the decision, per
    Rule 6. This function does not itself decide escalation for
    sensitive content; see contains_sensitive_content() for Rule 4.

    Args:
        record: The documentation record to check.

    Returns:
        True if escalation is required.
    """
    if record.escalation_recommended:
        return True
    return _matches_any(_classification_text(record), _ESCALATION_PATTERN)


def detect_non_interaction(record: DocumentationRecord) -> bool:
    """
    Check whether the record indicates no genuine customer interaction.

    Covers empty calls, silent calls, immediate disconnects, and wrong
    numbers, per Rule 9.

    Args:
        record: The documentation record to check.

    Returns:
        True if a non-interaction pattern is detected.
    """
    return _matches_any(_combined_text(record), _NON_INTERACTION_PATTERN)


def _detect_prompt_injection_attempt(record: DocumentationRecord) -> bool:
    """
    Detect customer-authored phrases attempting to influence routing.

    This check exists purely for audit logging (Rule 6). Its result is
    NEVER used to affect the routing decision.

    Args:
        record: The documentation record to check.

    Returns:
        True if an instruction-like phrase is present in the record's
        textual fields.
    """
    return _matches_any(_combined_text(record), _INJECTION_PATTERN)


# --------------------------------------------------------------------------
# Decision assembly helpers
# --------------------------------------------------------------------------


def _validate_decision_type(decision: DecisionType) -> DecisionType:
    """
    Validate that a computed decision is one of the allowed DecisionType values.

    Args:
        decision: The decision to validate.

    Returns:
        The same decision, unchanged.

    Raises:
        InvalidDecisionError: If `decision` is not a DecisionType member.
    """
    if not isinstance(decision, DecisionType):
        raise InvalidDecisionError(f"Computed decision is not a valid DecisionType: {decision!r}")
    return decision


def _finalize(
    decision: DecisionType,
    reason: str,
    triggered_rules: list[str],
    record: DocumentationRecord,
) -> BusinessDecision:
    """
    Assemble and log the final BusinessDecision.

    Args:
        decision: The computed routing decision.
        reason: Human-readable explanation for the decision.
        triggered_rules: Ordered list of rule labels that applied.
        record: The source documentation record (used for confidence).

    Returns:
        A fully populated, immutable BusinessDecision.

    Raises:
        InvalidDecisionError: If `decision` is not a valid DecisionType.
    """
    _validate_decision_type(decision)

    business_decision = BusinessDecision(
        decision=decision,
        reason=reason,
        triggered_rules=list(triggered_rules),
        confidence=record.confidence,
        timestamp=datetime.now(timezone.utc),
    )

    logger.info(
        "Final decision=%s triggered_rules=%s confidence=%.2f",
        business_decision.decision.value,
        business_decision.triggered_rules,
        business_decision.confidence,
    )
    return business_decision


def _build_human_review_reason(
    missing_fields: list[str],
    unsupported_fields: list[str],
    ambiguous_fields: list[str],
) -> str:
    """
    Build a human-readable HUMAN_REVIEW reason from the failing checks.

    Args:
        missing_fields: Required fields that are empty or "Not stated".
        unsupported_fields: Required fields with no grounding entry.
        ambiguous_fields: Required fields with an empty grounding entry.

    Returns:
        A reason string listing the specific failing fields.
    """
    parts: list[str] = []
    if unsupported_fields:
        parts.append(f"Missing grounding for: {', '.join(unsupported_fields)}.")
    if missing_fields:
        parts.append(f"Missing required content for: {', '.join(missing_fields)}.")
    if ambiguous_fields:
        parts.append(f"Ambiguous (zero-evidence) grounding for: {', '.join(ambiguous_fields)}.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def evaluate_record(
    record: DocumentationRecord,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> BusinessDecision:
    """
    Deterministically evaluate a DocumentationRecord and return a routing decision.

    Evaluation order:
        1. Non-interaction (Rule 9) - checked first since it overrides
           all other classification.
        2. Sensitive content (Rule 4) and handbook/model-recommended
           escalation (Rule 5) - always escalate, regardless of
           completeness or confidence.
        3. Grounding and missing-field checks (Rules 1, 2, 7) -> HUMAN_REVIEW.
        4. Confidence threshold (Rule 8) -> HUMAN_REVIEW.
        5. Otherwise, all AUTO_SAVE conditions are satisfied (Rules 3, 10).

    Args:
        record: A validated DocumentationRecord produced by
            documentation_engine.create_documentation().
        confidence_threshold: Minimum confidence required for
            AUTO_SAVE eligibility. Defaults to CONFIDENCE_THRESHOLD.

    Returns:
        A BusinessDecision describing the final routing outcome.

    Raises:
        BusinessRuleError: If `record` is not a DocumentationRecord.
        GroundingValidationError: If `record.grounding` is not a dict.
        InvalidDecisionError: If an internal computation produces an
            invalid decision value (defensive; should not occur in
            normal operation).
    """
    if not isinstance(record, DocumentationRecord):
        raise BusinessRuleError(
            f"evaluate_record expects a DocumentationRecord, got: {type(record).__name__}"
        )

    logger.info(
        "Evaluating record: category=%s priority=%s disposition=%s confidence=%.2f",
        record.category,
        record.priority,
        record.disposition,
        record.confidence,
    )

    if _detect_prompt_injection_attempt(record):
        logger.warning(
            "Instruction-like phrasing detected in record content; "
            "ignored for routing purposes (Rule 6)."
        )

    triggered_rules: list[str] = []

    # Rule 9: non-interaction takes precedence over all other checks.
    if detect_non_interaction(record):
        triggered_rules.append("Rule 9")
        return _finalize(
            DecisionType.NON_INTERACTION,
            "No genuine customer interaction was detected in this call.",
            triggered_rules,
            record,
        )

    # Rules 4 & 5: sensitive content and escalation triggers are mandatory
    # and override completeness/confidence checks entirely.
    sensitive = contains_sensitive_content(record)
    escalation_needed = requires_escalation(record)

    if sensitive:
        triggered_rules.append("Rule 4")
    if escalation_needed:
        triggered_rules.append("Rule 5")

    if sensitive or escalation_needed:
        reasons = []
        if sensitive:
            reasons.append("sensitive content was detected")
        if escalation_needed:
            reasons.append("escalation is required by the handbook or model recommendation")
        reason = f"Escalating because {', and '.join(reasons)}."
        return _finalize(DecisionType.ESCALATE, reason, triggered_rules, record)

    # Rules 1, 2, 7: structural completeness and grounding.
    missing_fields = _missing_required_fields(record)
    unsupported_fields = _unsupported_grounded_fields(record)
    ambiguous_fields = _ambiguous_grounded_fields(record)

    if missing_fields or unsupported_fields or ambiguous_fields:
        if unsupported_fields:
            triggered_rules.append("Rule 1")
        if missing_fields:
            triggered_rules.append("Rule 2")
        if ambiguous_fields:
            triggered_rules.append("Rule 7")

        reason = _build_human_review_reason(missing_fields, unsupported_fields, ambiguous_fields)
        return _finalize(DecisionType.HUMAN_REVIEW, reason, triggered_rules, record)

    # Rule 8: confidence threshold.
    if record.confidence < confidence_threshold:
        triggered_rules.append("Rule 8")
        reason = (
            f"Confidence {record.confidence:.2f} is below the required "
            f"threshold of {confidence_threshold:.2f}."
        )
        return _finalize(DecisionType.HUMAN_REVIEW, reason, triggered_rules, record)

    # Rules 3 & 10: every AUTO_SAVE condition has been satisfied.
    triggered_rules.extend(["Rule 1", "Rule 3", "Rule 10"])
    reason = "Documentation is complete, grounded, unambiguous, and free of escalation or sensitive-content triggers."
    return _finalize(DecisionType.AUTO_SAVE, reason, triggered_rules, record)