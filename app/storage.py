"""
app/storage.py

Persistence module for the AI-Powered After-Call Automation Platform.

Responsibility (and ONLY responsibility):
    Persist the final ReviewResult (and the DocumentationRecord /
    BusinessDecision it carries) to disk as JSON, and provide the
    matching read/list/delete operations.

This module NEVER:
    - calls Gemini, OpenAI, or LangChain
    - reads transcript files or searches the handbook
    - generates documentation
    - applies business rules or performs routing
    - modifies a ReviewResult, DocumentationRecord, or BusinessDecision

Design goal:
    This module is a thin, swappable persistence boundary. Every
    public function operates purely in terms of ReviewResult and
    plain paths/JSON; nothing here is coupled to the JSON-file
    implementation detail. A future developer can replace the bodies
    of save_record(), load_record(), list_records(), and
    delete_record() with calls to PostgreSQL, MongoDB, or a cloud
    object store without changing any other module in this project.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from app.business_rules import BusinessDecision, DecisionType
from app.documentation_engine import DocumentationRecord
from app.review import ReviewQueue, ReviewResult, ReviewStatus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_BASE_DIR: Final[Path] = Path("outputs")
PIPELINE_VERSION: Final[str] = "1.0"

_STATUS_DIRECTORY: Final[dict[ReviewStatus, str]] = {
    ReviewStatus.READY_TO_SAVE: "records",
    ReviewStatus.PENDING_REVIEW: "reviews",
    ReviewStatus.ESCALATED: "escalations",
    ReviewStatus.ARCHIVED: "archive",
}
"""Maps each ReviewStatus to the subdirectory its record is stored under."""

_LOGS_SUBDIRECTORY: Final[str] = "logs"
"""Reserved subdirectory for future log output; always created, never
written to by this module."""

_ALL_SUBDIRECTORIES: Final[tuple[str, ...]] = (
    *_STATUS_DIRECTORY.values(),
    _LOGS_SUBDIRECTORY,
)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class StorageError(Exception):
    """Base exception for all persistence failures."""


class SerializationError(StorageError):
    """Raised when a record cannot be safely converted to or from JSON."""


class RecordNotFoundError(StorageError):
    """Raised when a requested stored record does not exist."""


class InvalidStoragePathError(StorageError):
    """Raised when a destination path is structurally invalid for storage."""


# --------------------------------------------------------------------------
# Directory management
# --------------------------------------------------------------------------


def ensure_directories(base_dir: Path = DEFAULT_BASE_DIR) -> None:
    """
    Create every required output subdirectory if it does not already exist.

    Never assumes any directory already exists; safe to call before
    every operation.

    Args:
        base_dir: Root output directory under which all subdirectories
            are created.

    Raises:
        InvalidStoragePathError: If a required path exists but is not
            a directory.
    """
    for subdirectory in _ALL_SUBDIRECTORIES:
        directory = base_dir / subdirectory
        if directory.exists() and not directory.is_dir():
            raise InvalidStoragePathError(
                f"Expected a directory but found a file at: {directory}"
            )
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info("Directory created: %s", directory)


def initialize_storage(base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    """
    Prepare the storage layer for use.

    Args:
        base_dir: Root output directory to initialize.

    Returns:
        The resolved base directory, ready for use.
    """
    ensure_directories(base_dir)
    logger.info("Storage initialized: %s", base_dir)
    return base_dir


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_review_result(review_result: ReviewResult) -> None:
    """
    Validate that a ReviewResult and its nested objects are well-formed.

    Args:
        review_result: The review result to validate.

    Raises:
        StorageError: If `review_result` or any of its required nested
            objects (documentation, decision) is missing or of the
            wrong type.
    """
    if not isinstance(review_result, ReviewResult):
        raise StorageError(f"Expected a ReviewResult, got: {type(review_result).__name__}")
    if not isinstance(review_result.documentation, DocumentationRecord):
        raise StorageError(
            f"ReviewResult.documentation must be a DocumentationRecord, got: "
            f"{type(review_result.documentation).__name__}"
        )
    if not isinstance(review_result.decision, BusinessDecision):
        raise StorageError(
            f"ReviewResult.decision must be a BusinessDecision, got: "
            f"{type(review_result.decision).__name__}"
        )
    if not isinstance(review_result.status, ReviewStatus):
        raise StorageError(
            f"ReviewResult.status must be a ReviewStatus, got: {type(review_result.status).__name__}"
        )


def _validate_destination(path: Path) -> None:
    """
    Validate that a file's parent directory exists and is a directory.

    Args:
        path: The file path that will be written to or read from.

    Raises:
        InvalidStoragePathError: If the parent directory does not
            exist or is not a directory.
    """
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise InvalidStoragePathError(f"Destination directory does not exist: {parent}")


# --------------------------------------------------------------------------
# Identifier / path resolution
# --------------------------------------------------------------------------


def _resolve_identifier(review_result: ReviewResult, call_id: str | None) -> str:
    """
    Resolve the identifier used to name a record's JSON file.

    Uses `call_id` if provided, otherwise falls back to the
    ReviewResult's own review_id (already a UUID), and as a last
    defensive resort generates a fresh UUID.

    Args:
        review_result: The review result being persisted.
        call_id: An explicit call identifier, if available.

    Returns:
        A non-empty string identifier suitable for use as a filename.
    """
    if call_id:
        return call_id
    if review_result.review_id:
        return review_result.review_id
    return str(uuid.uuid4())


def build_output_path(
    review_result: ReviewResult,
    call_id: str | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> Path:
    """
    Compute the destination file path for a ReviewResult.

    The subdirectory is chosen by `review_result.status`; the filename
    is the resolved identifier with a ".json" extension.

    Args:
        review_result: The review result to build a path for.
        call_id: An explicit call identifier, if available.
        base_dir: Root output directory.

    Returns:
        The full path the record should be written to or read from.

    Raises:
        InvalidStoragePathError: If `review_result.status` has no
            known storage directory.
    """
    subdirectory = _STATUS_DIRECTORY.get(review_result.status)
    if subdirectory is None:
        raise InvalidStoragePathError(
            f"No storage directory mapped for status: {review_result.status!r}"
        )

    identifier = _resolve_identifier(review_result, call_id)
    return base_dir / subdirectory / f"{identifier}.json"


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def _to_json_safe(value: Any) -> Any:
    """
    Recursively convert a value into plain, JSON-serializable Python types.

    Handles dataclass instances, Enum members, datetime objects,
    Path objects, dicts, lists/tuples, and JSON-native scalars.

    Args:
        value: Any value reachable from a ReviewResult.

    Returns:
        An equivalent value built only from dict, list, str, int,
        float, bool, and None.

    Raises:
        SerializationError: If a value of an unsupported type is
            encountered.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SerializationError(f"Cannot serialize value of type: {type(value).__name__}")


def serialize_record(review_result: ReviewResult, call_id: str | None = None) -> dict[str, Any]:
    """
    Convert a ReviewResult into the JSON-safe payload structure used for storage.

    The resulting dict has exactly four top-level keys: "metadata",
    "documentation", "decision", and "review".

    Args:
        review_result: The review result to serialize.
        call_id: An explicit call identifier, if available.

    Returns:
        A dict containing only JSON-native types.

    Raises:
        StorageError: If `review_result` fails validation.
        SerializationError: If the record cannot be fully converted to
            JSON-safe types.
    """
    _validate_review_result(review_result)

    full_payload = _to_json_safe(review_result)
    documentation_payload = full_payload.pop("documentation")
    decision_payload = full_payload.pop("decision")
    review_payload = full_payload  # remaining keys: review_id, status, queue, etc.

    identifier = _resolve_identifier(review_result, call_id)
    metadata = {
        "call_id": identifier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }

    payload = {
        "metadata": metadata,
        "documentation": documentation_payload,
        "decision": decision_payload,
        "review": review_payload,
    }

    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"Record is not fully JSON-serializable: {exc}") from exc

    logger.info("Record serialized: call_id=%s status=%s", identifier, review_result.status.value)
    return payload


# --------------------------------------------------------------------------
# Deserialization (used by load_record / list_records)
# --------------------------------------------------------------------------


def _documentation_from_dict(data: dict[str, Any]) -> DocumentationRecord:
    """
    Reconstruct a DocumentationRecord from its stored JSON representation.

    Args:
        data: The "documentation" section of a stored record.

    Returns:
        A reconstructed DocumentationRecord.

    Raises:
        SerializationError: If required fields are missing or malformed.
    """
    try:
        return DocumentationRecord(**data)
    except TypeError as exc:
        raise SerializationError(f"Malformed documentation payload: {exc}") from exc


def _decision_from_dict(data: dict[str, Any]) -> BusinessDecision:
    """
    Reconstruct a BusinessDecision from its stored JSON representation.

    Args:
        data: The "decision" section of a stored record.

    Returns:
        A reconstructed BusinessDecision.

    Raises:
        SerializationError: If required fields are missing or malformed.
    """
    try:
        return BusinessDecision(
            decision=DecisionType(data["decision"]),
            reason=data["reason"],
            triggered_rules=list(data["triggered_rules"]),
            confidence=float(data["confidence"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
    except (KeyError, ValueError) as exc:
        raise SerializationError(f"Malformed decision payload: {exc}") from exc


def _review_from_dict(
    data: dict[str, Any],
    documentation: DocumentationRecord,
    decision: BusinessDecision,
) -> ReviewResult:
    """
    Reconstruct a ReviewResult from its stored JSON representation.

    Args:
        data: The "review" section of a stored record.
        documentation: The already-reconstructed DocumentationRecord.
        decision: The already-reconstructed BusinessDecision.

    Returns:
        A reconstructed ReviewResult.

    Raises:
        SerializationError: If required fields are missing or malformed.
    """
    try:
        return ReviewResult(
            review_id=data["review_id"],
            documentation=documentation,
            decision=decision,
            status=ReviewStatus(data["status"]),
            queue=ReviewQueue(data["queue"]),
            review_required=data["review_required"],
            review_reason=data["review_reason"],
            assigned_to=data["assigned_to"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
    except (KeyError, ValueError) as exc:
        raise SerializationError(f"Malformed review payload: {exc}") from exc


def _deserialize_record(payload: dict[str, Any]) -> ReviewResult:
    """
    Reconstruct a full ReviewResult from a stored JSON payload.

    Args:
        payload: The complete stored record (metadata/documentation/
            decision/review).

    Returns:
        A reconstructed ReviewResult.

    Raises:
        SerializationError: If the payload is missing required
            top-level sections or their contents are malformed.
    """
    try:
        documentation = _documentation_from_dict(payload["documentation"])
        decision = _decision_from_dict(payload["decision"])
        return _review_from_dict(payload["review"], documentation, decision)
    except KeyError as exc:
        raise SerializationError(f"Stored record is missing a required section: {exc}") from exc


# --------------------------------------------------------------------------
# Public persistence API
# --------------------------------------------------------------------------


def save_record(
    review_result: ReviewResult,
    call_id: str | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> Path:
    """
    Persist a ReviewResult to disk as a single pretty-printed JSON file.

    The destination subdirectory is chosen automatically from
    `review_result.status`. Missing directories are created
    automatically; this function never assumes they already exist.

    Args:
        review_result: The review result to persist.
        call_id: An explicit call identifier, used as the filename
            when available. Falls back to review_result.review_id.
        base_dir: Root output directory.

    Returns:
        The path of the saved JSON file.

    Raises:
        StorageError: If `review_result` fails validation or the file
            cannot be written.
        SerializationError: If the record cannot be converted to JSON.
        InvalidStoragePathError: If the destination path is invalid.
    """
    _validate_review_result(review_result)
    ensure_directories(base_dir)

    payload = serialize_record(review_result, call_id)
    output_path = build_output_path(review_result, call_id, base_dir)
    _validate_destination(output_path)

    try:
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise StorageError(f"Failed to write record to {output_path}: {exc}") from exc

    logger.info("Record saved: %s", output_path)
    return output_path


def load_record(path: Path) -> ReviewResult:
    """
    Load and reconstruct a previously stored ReviewResult from disk.

    Args:
        path: Path to the stored JSON record.

    Returns:
        The reconstructed ReviewResult, with its DocumentationRecord
        and BusinessDecision fully rebuilt.

    Raises:
        RecordNotFoundError: If the file does not exist.
        StorageError: If the file cannot be read or parsed as JSON.
        SerializationError: If the parsed JSON is missing required
            fields or contains malformed values.
    """
    if not path.exists() or not path.is_file():
        raise RecordNotFoundError(f"Record not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except OSError as exc:
        raise StorageError(f"Failed to read record from {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Record at {path} is not valid JSON: {exc}") from exc

    review_result = _deserialize_record(payload)
    logger.info("Record loaded: %s", path)
    return review_result


def list_records(
    status: ReviewStatus | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> list[ReviewResult]:
    """
    List and load every stored record, optionally filtered by status.

    Args:
        status: If provided, only records stored under this status's
            subdirectory are returned. If None, records from every
            status subdirectory are returned.
        base_dir: Root output directory.

    Returns:
        A list of reconstructed ReviewResult objects, ordered by
        filename within each subdirectory.

    Raises:
        StorageError: If any stored file cannot be read or parsed.
        SerializationError: If any stored file contains malformed data.
    """
    ensure_directories(base_dir)

    if status is not None:
        subdirectories = [_STATUS_DIRECTORY[status]]
    else:
        subdirectories = list(_STATUS_DIRECTORY.values())

    records: list[ReviewResult] = []
    for subdirectory in subdirectories:
        directory = base_dir / subdirectory
        for file_path in sorted(directory.glob("*.json")):
            records.append(load_record(file_path))

    logger.info("Listed %d record(s) (status=%s)", len(records), status.value if status else "ALL")
    return records


def delete_record(path: Path) -> bool:
    """
    Delete a previously stored record from disk.

    Args:
        path: Path to the stored JSON record to delete.

    Returns:
        True if the file was deleted successfully.

    Raises:
        RecordNotFoundError: If the file does not exist.
        StorageError: If the file exists but cannot be deleted.
    """
    if not path.exists() or not path.is_file():
        raise RecordNotFoundError(f"Record not found: {path}")

    try:
        path.unlink()
    except OSError as exc:
        raise StorageError(f"Failed to delete record at {path}: {exc}") from exc

    logger.info("Record deleted: %s", path)
    return True