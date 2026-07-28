"""
app/human_review.py

Human-in-the-loop action layer for the AI-Powered After-Call
Automation Platform.

Responsibility (and ONLY responsibility):
    Turn a human reviewer's decision (approve / edit / override) on an
    already-routed, already-persisted ReviewResult into an updated
    ReviewResult, and persist that update through storage.py.

This module is what app/review.py's docstring calls "a person": the
review screen (app/review_screen.py) is the UI; this module is the
logic behind its three buttons. It NEVER:
    - calls Gemini or performs handbook retrieval
    - decides routing for a *new* call (that is business_rules.py's
      job, invoked once per call inside app/main.py)
    - talks to the filesystem directly except through storage.py's
      public functions

Why routing logic is reused, not duplicated:
    edit_record() and override_record() both call
    business_rules.evaluate_record() / review.route_record() again
    after the human's change, so the same deterministic rules that
    governed the automatic pass also govern the corrected one. A human
    can edit content or force a different outcome, but they cannot
    silently move a record into "saved" without that move being
    recorded as a deliberate action with a reason and an audit trail
    (Rule 6 of business_rules.py: instructions/edits never bypass the
    routing logic itself).
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.business_rules import DecisionType
from app.business_rules import BusinessDecision, DecisionType, evaluate_record
from app.documentation_engine import DocumentationRecord
from app.review import ReviewQueue, ReviewResult, ReviewStatus, route_record
from app.storage import (
    DEFAULT_BASE_DIR,
    delete_record,
    list_records_with_paths,
    load_record,
    save_record,
)

logger = logging.getLogger(__name__)

DEFAULT_REVIEWER: str = "human_reviewer"


class HumanReviewError(Exception):
    """Base exception for all human-review-action failures."""


class AlreadyFinalizedError(HumanReviewError):
    """Raised when an action targets a record that is already saved or archived."""


# --------------------------------------------------------------------------
# Reading the queue
# --------------------------------------------------------------------------


def list_pending(base_dir: Path = DEFAULT_BASE_DIR) -> list[tuple[Path, ReviewResult]]:
    """
    List every record still waiting on a human: HUMAN_REVIEW + ESCALATE queues.

    Args:
        base_dir: Root output directory.

    Returns:
        A list of (path, ReviewResult) tuples for every record whose
        status is PENDING_REVIEW or ESCALATED, in that order.
    """
    review_items = list_records_with_paths(status=ReviewStatus.PENDING_REVIEW, base_dir=base_dir)
    escalated_items = list_records_with_paths(status=ReviewStatus.ESCALATED, base_dir=base_dir)
    return [*escalated_items, *review_items]


def _require_pending(result: ReviewResult) -> None:
    """Raise if a record is not in a state a human action can still apply to."""
    if result.status in (ReviewStatus.READY_TO_SAVE, ReviewStatus.ARCHIVED):
        raise AlreadyFinalizedError(
            f"Record is already finalized with status={result.status.value}; "
            "no further human action is possible."
        )


def _resave(path: Path, new_result: ReviewResult, base_dir: Path) -> Path:
    """Persist new_result under its correct queue and remove the old file if it moved."""
    call_id = path.stem
    new_path = save_record(new_result, call_id=call_id, base_dir=base_dir)
    if new_path.resolve() != path.resolve():
        delete_record(path)
    logger.info("Human action moved record %s -> %s", path, new_path)
    return new_path


# --------------------------------------------------------------------------
# Approve
# --------------------------------------------------------------------------


def approve_record(
    path: Path,
    reviewer: str = DEFAULT_REVIEWER,
    note: str = "",
    base_dir: Path = DEFAULT_BASE_DIR,
) -> Path:
    """
    Approve a pending or escalated record exactly as drafted, and save it.

    The original BusinessDecision is preserved for the audit trail
    (so it is still visible that the record was, for example,
    originally escalated); only the ReviewResult's status/queue move
    to READY_TO_SAVE/SAVE_QUEUE, with review_required cleared.

    Args:
        path: Path to the stored record (under outputs/reviews or
            outputs/escalations).
        reviewer: Identifier of the person approving the record.
        note: Optional free-text note to attach to the approval.
        base_dir: Root output directory.

    Returns:
        The path of the now-saved record under outputs/records.

    Raises:
        AlreadyFinalizedError: If the record was already saved or
            archived.
    """
    result = load_record(path)
    _require_pending(result)

    reason = f"Approved by {reviewer} (originally: {result.decision.reason})"
    if note:
        reason += f" | Reviewer note: {note}"

    approved_decision = dataclasses.replace(
        result.decision,
        triggered_rules=[*result.decision.triggered_rules, "Human Approval"],
    )

    approved_result = dataclasses.replace(
        result,
        decision=approved_decision,
        status=ReviewStatus.READY_TO_SAVE,
        queue=ReviewQueue.SAVE_QUEUE,
        review_required=False,
        review_reason=reason,
        assigned_to=None,
        updated_at=datetime.now(timezone.utc),
    )

    return _resave(path, approved_result, base_dir)


# --------------------------------------------------------------------------
# Edit
# --------------------------------------------------------------------------


def edit_record(
    path: Path,
    field_updates: dict[str, Any],
    reviewer: str = DEFAULT_REVIEWER,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> Path:
    """
    Apply reviewer edits to a record's documentation fields, then re-route it.

    The edited DocumentationRecord is passed back through
    business_rules.evaluate_record() and review.route_record(), so the
    same deterministic rules apply to the corrected content. Editing a
    record therefore does NOT automatically save it: if the edit still
    doesn't clear the AUTO_SAVE bar, the record stays in a human queue
    for a follow-up approve_record() call.

    Args:
        path: Path to the stored record.
        field_updates: Mapping of DocumentationRecord field name to
            new value (e.g. {"root_cause": "Modem firmware bug",
            "priority": "High"}).
        reviewer: Identifier of the person editing the record.
        base_dir: Root output directory.

    Returns:
        The path of the re-saved record (its queue may have changed).

    Raises:
        AlreadyFinalizedError: If the record was already saved or
            archived.
        ValueError: If field_updates references a field that does not
            exist on DocumentationRecord.
    """
    result = load_record(path)
    _require_pending(result)

    valid_fields = {f.name for f in dataclasses.fields(DocumentationRecord)}
    unknown = set(field_updates) - valid_fields
    if unknown:
        raise ValueError(f"Unknown DocumentationRecord field(s): {sorted(unknown)}")

    updated_documentation = dataclasses.replace(result.documentation, **field_updates)
    new_decision = evaluate_record(updated_documentation)
    routed = route_record(updated_documentation, new_decision)

    reason_prefix = f"Edited by {reviewer} (fields changed: {', '.join(sorted(field_updates))}). "
    edited_result = dataclasses.replace(
        routed,
        review_id=result.review_id,
        created_at=result.created_at,
        review_reason=(reason_prefix + (routed.review_reason or "")).strip(),
        updated_at=datetime.now(timezone.utc),
    )

    return _resave(path, edited_result, base_dir)


# --------------------------------------------------------------------------
# Override
# --------------------------------------------------------------------------


def override_record(
    path: Path,
    new_decision: DecisionType,
    note: str,
    reviewer: str = DEFAULT_REVIEWER,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> Path:
    """
    Force a record to a different routing outcome than business_rules chose.

    A justification note is mandatory and is embedded in the stored
    reason, so an override is always auditable: who did it, what the
    original automated decision was, and why it was overridden.

    Args:
        path: Path to the stored record.
        new_decision: The DecisionType the reviewer wants instead.
        note: Mandatory justification for the override.
        reviewer: Identifier of the person overriding the record.
        base_dir: Root output directory.

    Returns:
        The path of the re-saved record under its new queue.

    Raises:
        AlreadyFinalizedError: If the record was already saved or
            archived.
        ValueError: If `note` is blank.
    """
    if not note.strip():
        raise ValueError("A justification note is required to override a routing decision.")

    result = load_record(path)
    _require_pending(result)

    overridden_decision = BusinessDecision(
        decision=new_decision,
        reason=(
            f"Human override by {reviewer}: {note.strip()} "
            f"(was {result.decision.decision.value}: {result.decision.reason})"
        ),
        triggered_rules=[*result.decision.triggered_rules, "Human Override"],
        confidence=result.decision.confidence,
        timestamp=datetime.now(timezone.utc),
    )

    routed = route_record(result.documentation, overridden_decision)
    overridden_result = dataclasses.replace(
        routed,
        review_id=result.review_id,
        created_at=result.created_at,
        updated_at=datetime.now(timezone.utc),
    )

    return _resave(path, overridden_result, base_dir)
