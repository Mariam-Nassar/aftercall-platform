"""
app/review.py

Review workflow routing module for the AI-Powered After-Call
Automation Platform.

Responsibility (and ONLY responsibility):
    Given a DocumentationRecord and the BusinessDecision computed for
    it, deterministically route the record to the correct queue and
    produce a single ReviewResult describing that routing.

This module NEVER:
    - creates or modifies documentation content
    - calls Gemini or any other LLM
    - performs handbook retrieval
    - evaluates business rules (it only consumes an already-decided
      BusinessDecision)
    - saves records, writes to a database, or has any other side effect

Routing is pure, deterministic Python logic: DecisionType in, a fixed
(status, queue, review_required, assigned_to) tuple out. No AI, no
prompts, no generation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Final

from app.business_rules import BusinessDecision, DecisionType
from app.documentation_engine import DocumentationRecord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class ReviewStatus(str, Enum):
    """The final status assigned to a routed record."""

    READY_TO_SAVE = "READY_TO_SAVE"
    PENDING_REVIEW = "PENDING_REVIEW"
    ESCALATED = "ESCALATED"
    ARCHIVED = "ARCHIVED"


class ReviewQueue(str, Enum):
    """The queue a routed record is placed into."""

    SAVE_QUEUE = "SAVE_QUEUE"
    REVIEW_QUEUE = "REVIEW_QUEUE"
    ESCALATION_QUEUE = "ESCALATION_QUEUE"
    ARCHIVE_QUEUE = "ARCHIVE_QUEUE"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ReviewError(Exception):
    """Base exception for all review-routing failures."""


class RoutingError(ReviewError):
    """Raised when a computed routing plan is internally inconsistent."""


class UnknownDecisionError(ReviewError):
    """Raised when a BusinessDecision carries a decision value with no known route."""


# --------------------------------------------------------------------------
# Output structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewResult:
    """
    The complete outcome of routing one DocumentationRecord.

    Attributes:
        review_id: Unique identifier for this routing outcome.
        documentation: The documentation record being routed.
        decision: The business decision that drove this routing.
        status: The final status assigned to the record.
        queue: The queue the record was placed into.
        review_required: Whether a human must act on this record.
        review_reason: Explanation for why review is required, or None.
        assigned_to: The team/role assigned to handle this record, or
            None if no assignment is needed.
        created_at: UTC time this ReviewResult was created.
        updated_at: UTC time this ReviewResult was last updated (equal
            to created_at, since this module produces the result in a
            single step with no later mutation).
    """

    review_id: str
    documentation: DocumentationRecord
    decision: BusinessDecision
    status: ReviewStatus
    queue: ReviewQueue
    review_required: bool
    review_reason: str | None
    assigned_to: str | None
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------
# Internal routing plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _RoutingPlan:
    """
    The routing outcome for a single decision, before a review_id is assigned.

    Attributes:
        status: The ReviewStatus to assign.
        queue: The ReviewQueue to assign.
        review_required: Whether human review is required.
        review_reason: Reason for review, or None.
        assigned_to: Assignee, or None.
    """

    status: ReviewStatus
    queue: ReviewQueue
    review_required: bool
    review_reason: str | None
    assigned_to: str | None


_DECISION_ROUTING: Final[dict[DecisionType, tuple[ReviewStatus, ReviewQueue]]] = {
    DecisionType.AUTO_SAVE: (ReviewStatus.READY_TO_SAVE, ReviewQueue.SAVE_QUEUE),
    DecisionType.HUMAN_REVIEW: (ReviewStatus.PENDING_REVIEW, ReviewQueue.REVIEW_QUEUE),
    DecisionType.ESCALATE: (ReviewStatus.ESCALATED, ReviewQueue.ESCALATION_QUEUE),
    DecisionType.NON_INTERACTION: (ReviewStatus.ARCHIVED, ReviewQueue.ARCHIVE_QUEUE),
}
"""Canonical mapping from DecisionType to (status, queue), used both to
build and to validate every routing plan."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_inputs(documentation: DocumentationRecord, decision: BusinessDecision) -> None:
    """
    Validate that both inputs are present and well-typed.

    Args:
        documentation: The documentation record to validate.
        decision: The business decision to validate.

    Raises:
        ReviewError: If either argument is missing or of the wrong type.
        UnknownDecisionError: If `decision.decision` is not a valid
            DecisionType member.
    """
    if documentation is None or not isinstance(documentation, DocumentationRecord):
        raise ReviewError(
            f"route_record requires a DocumentationRecord, got: {type(documentation).__name__}"
        )
    if decision is None or not isinstance(decision, BusinessDecision):
        raise ReviewError(
            f"route_record requires a BusinessDecision, got: {type(decision).__name__}"
        )
    if not isinstance(decision.decision, DecisionType):
        raise UnknownDecisionError(
            f"BusinessDecision.decision is not a valid DecisionType: {decision.decision!r}"
        )


def _validate_routing_plan(decision_type: DecisionType, plan: _RoutingPlan) -> None:
    """
    Confirm a computed routing plan matches the canonical status/queue mapping.

    This is a defense-in-depth check: the prepare_*() functions are
    expected to always produce a plan consistent with
    _DECISION_ROUTING, but this guards against future edits breaking
    that invariant silently.

    Args:
        decision_type: The DecisionType the plan was built for.
        plan: The routing plan to validate.

    Raises:
        RoutingError: If the plan's status or queue does not match the
            canonical mapping for `decision_type`.
    """
    expected_status, expected_queue = _DECISION_ROUTING[decision_type]
    if plan.status is not expected_status or plan.queue is not expected_queue:
        raise RoutingError(
            f"Routing plan for {decision_type.value} is inconsistent: "
            f"expected status={expected_status.value} queue={expected_queue.value}, "
            f"got status={plan.status.value} queue={plan.queue.value}"
        )


# --------------------------------------------------------------------------
# Per-decision routing preparation
# --------------------------------------------------------------------------


def prepare_auto_save(decision: BusinessDecision) -> _RoutingPlan:
    """
    Build the routing plan for an AUTO_SAVE decision.

    The record is ready for persistence with no human involvement.

    Args:
        decision: The business decision (unused beyond its type, kept
            for a consistent function signature across all
            prepare_*() functions).

    Returns:
        A _RoutingPlan with status=READY_TO_SAVE, queue=SAVE_QUEUE,
        review_required=False, and no assignee.
    """
    status, queue = _DECISION_ROUTING[DecisionType.AUTO_SAVE]
    return _RoutingPlan(
        status=status,
        queue=queue,
        review_required=False,
        review_reason=None,
        assigned_to=None,
    )


def prepare_review(decision: BusinessDecision) -> _RoutingPlan:
    """
    Build the routing plan for a HUMAN_REVIEW decision.

    The record is sent to the manual review queue, carrying the
    business decision's reason forward so reviewers know why it was
    flagged.

    Args:
        decision: The business decision that triggered review.

    Returns:
        A _RoutingPlan with status=PENDING_REVIEW, queue=REVIEW_QUEUE,
        review_required=True, review_reason=decision.reason, and no
        assignee.
    """
    status, queue = _DECISION_ROUTING[DecisionType.HUMAN_REVIEW]
    return _RoutingPlan(
        status=status,
        queue=queue,
        review_required=True,
        review_reason=decision.reason,
        assigned_to=None,
    )


def prepare_escalation(decision: BusinessDecision) -> _RoutingPlan:
    """
    Build the routing plan for an ESCALATE decision.

    The record is sent to the escalation queue and assigned to the
    Escalation Team.

    Args:
        decision: The business decision that triggered escalation.

    Returns:
        A _RoutingPlan with status=ESCALATED, queue=ESCALATION_QUEUE,
        review_required=True, and assigned_to="Escalation Team".
    """
    status, queue = _DECISION_ROUTING[DecisionType.ESCALATE]
    return _RoutingPlan(
        status=status,
        queue=queue,
        review_required=True,
        review_reason=decision.reason,
        assigned_to="Escalation Team",
    )


def prepare_archive(decision: BusinessDecision) -> _RoutingPlan:
    """
    Build the routing plan for a NON_INTERACTION decision.

    The record is archived without entering any documentation
    workflow; no human involvement is required.

    Args:
        decision: The business decision (unused beyond its type, kept
            for a consistent function signature across all
            prepare_*() functions).

    Returns:
        A _RoutingPlan with status=ARCHIVED, queue=ARCHIVE_QUEUE,
        review_required=False, and no assignee.
    """
    status, queue = _DECISION_ROUTING[DecisionType.NON_INTERACTION]
    return _RoutingPlan(
        status=status,
        queue=queue,
        review_required=False,
        review_reason=None,
        assigned_to=None,
    )


_ROUTING_HANDLERS: Final[dict[DecisionType, Callable[[BusinessDecision], _RoutingPlan]]] = {
    DecisionType.AUTO_SAVE: prepare_auto_save,
    DecisionType.HUMAN_REVIEW: prepare_review,
    DecisionType.ESCALATE: prepare_escalation,
    DecisionType.NON_INTERACTION: prepare_archive,
}


# --------------------------------------------------------------------------
# Result assembly
# --------------------------------------------------------------------------


def create_review_result(
    documentation: DocumentationRecord,
    decision: BusinessDecision,
    plan: _RoutingPlan,
) -> ReviewResult:
    """
    Assemble the final, immutable ReviewResult from a routing plan.

    Args:
        documentation: The documentation record being routed.
        decision: The business decision that drove routing.
        plan: The routing plan computed by one of the prepare_*()
            functions.

    Returns:
        A fully populated ReviewResult with a freshly generated
        review_id and UTC timestamps.
    """
    now = datetime.now(timezone.utc)
    return ReviewResult(
        review_id=str(uuid.uuid4()),
        documentation=documentation,
        decision=decision,
        status=plan.status,
        queue=plan.queue,
        review_required=plan.review_required,
        review_reason=plan.review_reason,
        assigned_to=plan.assigned_to,
        created_at=now,
        updated_at=now,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def route_record(documentation: DocumentationRecord, decision: BusinessDecision) -> ReviewResult:
    """
    Route a documentation record to its correct queue based on its business decision.

    This is the sole entry point for this module: it validates its
    inputs, dispatches to the appropriate prepare_*() function based
    on `decision.decision`, and returns a single ReviewResult. No
    persistence, generation, or rule evaluation happens here.

    Args:
        documentation: The documentation record to route.
        decision: The business decision computed for this record.

    Returns:
        A ReviewResult describing where the record was routed and
        whether human review is required.

    Raises:
        ReviewError: If `documentation` or `decision` is missing or of
            the wrong type.
        UnknownDecisionError: If `decision.decision` is not a
            recognized DecisionType.
        RoutingError: If the computed routing plan is internally
            inconsistent with the canonical decision-to-route mapping
            (defensive; should not occur in normal operation).
    """
    _validate_inputs(documentation, decision)

    logger.info(
        "Record received: category=%s priority=%s decision=%s",
        documentation.category,
        documentation.priority,
        decision.decision.value,
    )
    logger.info("Business decision: %s (confidence=%.2f)", decision.decision.value, decision.confidence)

    handler = _ROUTING_HANDLERS.get(decision.decision)
    if handler is None:
        raise UnknownDecisionError(f"No routing handler for decision: {decision.decision!r}")

    logger.info("Routing started for decision=%s", decision.decision.value)
    plan = handler(decision)
    _validate_routing_plan(decision.decision, plan)

    result = create_review_result(documentation, decision, plan)

    logger.info("Assigned queue: %s", result.queue.value)
    logger.info("Final review status: %s", result.status.value)
    logger.info("Routing completed: review_id=%s", result.review_id)

    return result