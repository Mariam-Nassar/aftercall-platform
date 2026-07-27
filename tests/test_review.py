import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone

import pytest

from app.documentation_engine import DocumentationRecord
from app.business_rules import BusinessDecision, DecisionType
from app.review import (
    route_record,
    prepare_auto_save,
    prepare_review,
    prepare_escalation,
    prepare_archive,
    create_review_result,
    ReviewResult,
    ReviewStatus,
    ReviewQueue,
    ReviewError,
    RoutingError,
    UnknownDecisionError,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def documentation() -> DocumentationRecord:
    return DocumentationRecord(
        summary="Customer reported a billing overcharge.",
        issue="Charged twice for the same subscription.",
        root_cause="Not stated",
        resolution="Not stated",
        pending_actions=["Issue refund"],
        category="Billing",
        subcategory="Overcharge",
        priority="Medium",
        disposition="Unresolved",
        sentiment="Neutral",
        keywords=["billing", "overcharge"],
        tags=["billing"],
        escalation_recommended=False,
        confidence=0.9,
        grounding={
            "summary": ["Transcript Line 2"],
            "issue": ["Transcript Line 2"],
            "category": ["Handbook Rule 1"],
            "priority": ["Handbook Rule 2"],
            "disposition": ["Transcript Line 4"],
            "sentiment": ["Transcript Line 1"],
        },
    )


def make_decision(decision_type: DecisionType, reason: str = "Because reasons.") -> BusinessDecision:
    return BusinessDecision(
        decision=decision_type,
        reason=reason,
        triggered_rules=["Rule X"],
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------
# route_record - one test per decision type
# --------------------------------------------------------------------------


def test_auto_save_routes_to_save_queue(documentation):
    decision = make_decision(DecisionType.AUTO_SAVE)
    result = route_record(documentation, decision)

    assert isinstance(result, ReviewResult)
    assert result.status == ReviewStatus.READY_TO_SAVE
    assert result.queue == ReviewQueue.SAVE_QUEUE
    assert result.review_required is False
    assert result.assigned_to is None


def test_human_review_routes_to_review_queue_with_reason(documentation):
    decision = make_decision(DecisionType.HUMAN_REVIEW, reason="Missing grounded priority.")
    result = route_record(documentation, decision)

    assert result.status == ReviewStatus.PENDING_REVIEW
    assert result.queue == ReviewQueue.REVIEW_QUEUE
    assert result.review_required is True
    assert result.assigned_to is None
    assert result.review_reason == "Missing grounded priority."


def test_escalate_routes_to_escalation_queue_and_team(documentation):
    decision = make_decision(DecisionType.ESCALATE, reason="Sensitive legal complaint detected.")
    result = route_record(documentation, decision)

    assert result.status == ReviewStatus.ESCALATED
    assert result.queue == ReviewQueue.ESCALATION_QUEUE
    assert result.review_required is True
    assert result.assigned_to == "Escalation Team"


def test_non_interaction_routes_to_archive_queue(documentation):
    decision = make_decision(DecisionType.NON_INTERACTION, reason="Silent call.")
    result = route_record(documentation, decision)

    assert result.status == ReviewStatus.ARCHIVED
    assert result.queue == ReviewQueue.ARCHIVE_QUEUE
    assert result.review_required is False
    assert result.assigned_to is None


# --------------------------------------------------------------------------
# ReviewResult shape
# --------------------------------------------------------------------------


def test_review_result_has_unique_id_and_timestamps(documentation):
    decision = make_decision(DecisionType.AUTO_SAVE)
    result_a = route_record(documentation, decision)
    result_b = route_record(documentation, decision)

    assert result_a.review_id != result_b.review_id
    assert result_a.created_at == result_a.updated_at
    assert result_a.documentation is documentation
    assert result_a.decision is decision


# --------------------------------------------------------------------------
# prepare_*() functions directly
# --------------------------------------------------------------------------


def test_prepare_auto_save_plan():
    decision = make_decision(DecisionType.AUTO_SAVE)
    plan = prepare_auto_save(decision)
    assert plan.status == ReviewStatus.READY_TO_SAVE
    assert plan.queue == ReviewQueue.SAVE_QUEUE
    assert plan.review_required is False
    assert plan.assigned_to is None


def test_prepare_review_plan_carries_reason():
    decision = make_decision(DecisionType.HUMAN_REVIEW, reason="Ambiguous disposition.")
    plan = prepare_review(decision)
    assert plan.review_required is True
    assert plan.review_reason == "Ambiguous disposition."
    assert plan.assigned_to is None


def test_prepare_escalation_plan_assigns_team():
    decision = make_decision(DecisionType.ESCALATE)
    plan = prepare_escalation(decision)
    assert plan.assigned_to == "Escalation Team"
    assert plan.review_required is True


def test_prepare_archive_plan_no_review():
    decision = make_decision(DecisionType.NON_INTERACTION)
    plan = prepare_archive(decision)
    assert plan.review_required is False
    assert plan.assigned_to is None


def test_create_review_result_from_plan(documentation):
    decision = make_decision(DecisionType.AUTO_SAVE)
    plan = prepare_auto_save(decision)
    result = create_review_result(documentation, decision, plan)

    assert isinstance(result, ReviewResult)
    assert result.status == plan.status
    assert result.queue == plan.queue


# --------------------------------------------------------------------------
# Validation / error handling
# --------------------------------------------------------------------------


def test_route_record_rejects_missing_documentation():
    decision = make_decision(DecisionType.AUTO_SAVE)
    with pytest.raises(ReviewError):
        route_record(None, decision)


def test_route_record_rejects_missing_decision(documentation):
    with pytest.raises(ReviewError):
        route_record(documentation, None)


def test_route_record_rejects_wrong_types():
    with pytest.raises(ReviewError):
        route_record("not-a-record", "not-a-decision")


def test_route_record_rejects_invalid_decision_enum(documentation):
    decision = make_decision(DecisionType.AUTO_SAVE)
    object.__setattr__(decision, "decision", "NOT_A_REAL_DECISION")

    with pytest.raises(UnknownDecisionError):
        route_record(documentation, decision)


def test_validate_routing_plan_detects_mismatch(documentation):
    from app.review import _validate_routing_plan, _RoutingPlan

    mismatched_plan = _RoutingPlan(
        status=ReviewStatus.ESCALATED,  # wrong status for AUTO_SAVE
        queue=ReviewQueue.SAVE_QUEUE,
        review_required=False,
        review_reason=None,
        assigned_to=None,
    )
    with pytest.raises(RoutingError):
        _validate_routing_plan(DecisionType.AUTO_SAVE, mismatched_plan)