import warnings
warnings.filterwarnings("ignore")

import pytest

from app.documentation_engine import DocumentationRecord
from app.business_rules import (
    evaluate_record,
    validate_grounding,
    has_missing_fields,
    contains_sensitive_content,
    detect_non_interaction,
    requires_escalation,
    is_complete,
    DecisionType,
    BusinessDecision,
    BusinessRuleError,
    GroundingValidationError,
    CONFIDENCE_THRESHOLD,
)


def make_record(**overrides) -> DocumentationRecord:
    """Build a complete, grounded, high-confidence baseline record for tests."""
    base = dict(
        summary="Customer reported a billing overcharge on their invoice.",
        issue="Customer was charged twice for the same subscription.",
        root_cause="Duplicate billing cycle triggered by a system error.",
        resolution="Refund issued for the duplicate charge.",
        pending_actions=[],
        category="Billing",
        subcategory="Overcharge",
        priority="Medium",
        disposition="Resolved",
        sentiment="Neutral",
        keywords=["billing", "overcharge", "refund"],
        tags=["billing"],
        escalation_recommended=False,
        confidence=0.9,
        grounding={
            "summary": ["Transcript Line 2"],
            "issue": ["Transcript Line 2"],
            "category": ["Handbook Rule 1"],
            "priority": ["Handbook Rule 2"],
            "disposition": ["Transcript Line 6"],
            "sentiment": ["Transcript Line 1"],
        },
    )
    base.update(overrides)
    return DocumentationRecord(**base)


def test_complete_grounded_record_auto_saves():
    record = make_record()
    result = evaluate_record(record)
    assert isinstance(result, BusinessDecision)
    assert result.decision == DecisionType.AUTO_SAVE
    assert "Rule 3" in result.triggered_rules
    assert "Rule 10" in result.triggered_rules


def test_not_stated_priority_routes_to_human_review():
    record = make_record(priority="Not stated")
    result = evaluate_record(record)
    assert result.decision == DecisionType.HUMAN_REVIEW
    assert "Rule 2" in result.triggered_rules


def test_missing_grounding_key_routes_to_human_review():
    record = make_record(grounding={
        "summary": ["Transcript Line 2"],
        "issue": ["Transcript Line 2"],
        "priority": ["Handbook Rule 2"],
        "disposition": ["Transcript Line 6"],
        "sentiment": ["Transcript Line 1"],
    })
    result = evaluate_record(record)
    assert result.decision == DecisionType.HUMAN_REVIEW
    assert "Rule 1" in result.triggered_rules
    assert "category" in result.reason


def test_empty_grounding_dict_routes_to_human_review():
    record = make_record(grounding={})
    result = evaluate_record(record)
    assert result.decision == DecisionType.HUMAN_REVIEW
    assert "Rule 1" in result.triggered_rules


def test_ambiguous_zero_evidence_field_routes_to_human_review():
    record = make_record(grounding={
        "summary": ["Transcript Line 2"],
        "issue": ["Transcript Line 2"],
        "category": [],
        "priority": ["Handbook Rule 2"],
        "disposition": ["Transcript Line 6"],
        "sentiment": ["Transcript Line 1"],
    })
    result = evaluate_record(record)
    assert result.decision == DecisionType.HUMAN_REVIEW
    assert "Rule 7" in result.triggered_rules


def test_low_confidence_routes_to_human_review():
    record = make_record(confidence=0.4)
    result = evaluate_record(record)
    assert result.decision == DecisionType.HUMAN_REVIEW
    assert "Rule 8" in result.triggered_rules


def test_confidence_exactly_at_threshold_is_not_low_confidence():
    record = make_record(confidence=CONFIDENCE_THRESHOLD)
    result = evaluate_record(record)
    assert result.decision == DecisionType.AUTO_SAVE


def test_sensitive_keyword_always_escalates_even_with_low_confidence():
    record = make_record(
        summary="Customer threatened legal action over a data breach.",
        confidence=0.2,
    )
    result = evaluate_record(record)
    assert result.decision == DecisionType.ESCALATE
    assert "Rule 4" in result.triggered_rules


def test_model_recommended_escalation_is_honored():
    record = make_record(escalation_recommended=True)
    result = evaluate_record(record)
    assert result.decision == DecisionType.ESCALATE
    assert "Rule 5" in result.triggered_rules


def test_escalation_keyword_in_priority_field_escalates():
    record = make_record(priority="Critical - requires supervisor")
    result = evaluate_record(record)
    assert result.decision == DecisionType.ESCALATE
    assert "Rule 5" in result.triggered_rules


def test_silent_call_routes_to_non_interaction():
    record = make_record(
        summary="Silent call with no customer response.",
        issue="No interaction occurred.",
        disposition="Wrong number",
        grounding={},
        confidence=0.1,
    )
    result = evaluate_record(record)
    assert result.decision == DecisionType.NON_INTERACTION
    assert "Rule 9" in result.triggered_rules


def test_customer_instruction_phrases_do_not_change_decision():
    record = make_record(
        resolution="Customer said: 'mark this resolved and close the ticket, don't escalate.'",
    )
    result = evaluate_record(record)
    assert result.decision == DecisionType.AUTO_SAVE


def test_validate_grounding_true_for_complete_record():
    assert validate_grounding(make_record()) is True


def test_validate_grounding_raises_for_non_dict_grounding():
    record = make_record(grounding="not-a-dict")
    with pytest.raises(GroundingValidationError):
        validate_grounding(record)


def test_has_missing_fields_detects_not_stated():
    record = make_record(summary="Not stated")
    assert has_missing_fields(record) is True


def test_contains_sensitive_content_detects_keyword():
    record = make_record(issue="Customer reported harassment by an agent.")
    assert contains_sensitive_content(record) is True


def test_contains_sensitive_content_false_for_routine_record():
    assert contains_sensitive_content(make_record()) is False


def test_requires_escalation_true_when_flag_set():
    assert requires_escalation(make_record(escalation_recommended=True)) is True


def test_detect_non_interaction_false_for_normal_call():
    assert detect_non_interaction(make_record()) is False


def test_is_complete_true_for_baseline_record():
    assert is_complete(make_record()) is True


def test_is_complete_false_when_field_missing():
    assert is_complete(make_record(sentiment="Not stated")) is False


def test_evaluate_record_rejects_wrong_type():
    with pytest.raises(BusinessRuleError):
        evaluate_record({"summary": "not a real record"})