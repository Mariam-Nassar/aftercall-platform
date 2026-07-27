import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.documentation_engine import DocumentationRecord
from app.business_rules import BusinessDecision, DecisionType
from app.review import route_record, ReviewResult, ReviewStatus, ReviewQueue
from app.storage import (
    initialize_storage,
    ensure_directories,
    build_output_path,
    serialize_record,
    save_record,
    load_record,
    list_records,
    delete_record,
    StorageError,
    SerializationError,
    RecordNotFoundError,
    InvalidStoragePathError,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


@pytest.fixture
def documentation() -> DocumentationRecord:
    return DocumentationRecord(
        summary="Billing overcharge reported.",
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


@pytest.fixture
def auto_save_review(documentation) -> ReviewResult:
    return route_record(documentation, make_decision(DecisionType.AUTO_SAVE))


# --------------------------------------------------------------------------
# Directory management
# --------------------------------------------------------------------------


def test_ensure_directories_creates_all_five(base_dir):
    ensure_directories(base_dir)
    for subdirectory in ("records", "reviews", "escalations", "archive", "logs"):
        assert (base_dir / subdirectory).is_dir()


def test_initialize_storage_returns_base_dir(base_dir):
    result = initialize_storage(base_dir)
    assert result == base_dir
    assert (base_dir / "records").is_dir()


def test_ensure_directories_is_idempotent(base_dir):
    ensure_directories(base_dir)
    ensure_directories(base_dir)  # second call must not raise
    assert (base_dir / "records").is_dir()


def test_ensure_directories_raises_if_path_is_a_file(tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    (base / "records").write_text("not a directory")

    with pytest.raises(InvalidStoragePathError):
        ensure_directories(base)


# --------------------------------------------------------------------------
# build_output_path
# --------------------------------------------------------------------------


def test_build_output_path_uses_call_id_when_given(auto_save_review, base_dir):
    path = build_output_path(auto_save_review, call_id="K-001", base_dir=base_dir)
    assert path == base_dir / "records" / "K-001.json"


def test_build_output_path_falls_back_to_review_id(auto_save_review, base_dir):
    path = build_output_path(auto_save_review, call_id=None, base_dir=base_dir)
    assert path == base_dir / "records" / f"{auto_save_review.review_id}.json"


@pytest.mark.parametrize(
    "decision_type,expected_dir",
    [
        (DecisionType.AUTO_SAVE, "records"),
        (DecisionType.HUMAN_REVIEW, "reviews"),
        (DecisionType.ESCALATE, "escalations"),
        (DecisionType.NON_INTERACTION, "archive"),
    ],
)
def test_build_output_path_routes_by_status(documentation, base_dir, decision_type, expected_dir):
    review = route_record(documentation, make_decision(decision_type))
    path = build_output_path(review, call_id="K-999", base_dir=base_dir)
    assert path == base_dir / expected_dir / "K-999.json"


# --------------------------------------------------------------------------
# serialize_record
# --------------------------------------------------------------------------


def test_serialize_record_has_four_top_level_keys(auto_save_review):
    payload = serialize_record(auto_save_review, call_id="K-001")
    assert set(payload.keys()) == {"metadata", "documentation", "decision", "review"}


def test_serialize_record_is_json_native(auto_save_review):
    payload = serialize_record(auto_save_review, call_id="K-001")
    json.dumps(payload)  # must not raise


def test_serialize_record_metadata_contains_call_id_and_version(auto_save_review):
    payload = serialize_record(auto_save_review, call_id="K-001")
    assert payload["metadata"]["call_id"] == "K-001"
    assert payload["metadata"]["pipeline_version"] == "1.0"


def test_serialize_record_rejects_wrong_type():
    with pytest.raises(StorageError):
        serialize_record("not-a-review-result")


# --------------------------------------------------------------------------
# save_record / load_record round trip
# --------------------------------------------------------------------------


def test_save_and_load_round_trip(auto_save_review, base_dir):
    saved_path = save_record(auto_save_review, call_id="K-001", base_dir=base_dir)

    assert saved_path == base_dir / "records" / "K-001.json"
    assert saved_path.exists()

    loaded = load_record(saved_path)
    assert isinstance(loaded, ReviewResult)
    assert loaded.status == ReviewStatus.READY_TO_SAVE
    assert loaded.queue == ReviewQueue.SAVE_QUEUE
    assert loaded.documentation.category == "Billing"
    assert loaded.decision.decision == DecisionType.AUTO_SAVE


def test_save_record_creates_missing_directories_automatically(auto_save_review, base_dir):
    assert not base_dir.exists()
    save_record(auto_save_review, call_id="K-002", base_dir=base_dir)
    assert (base_dir / "records").is_dir()


def test_save_record_pretty_prints_json(auto_save_review, base_dir):
    path = save_record(auto_save_review, call_id="K-003", base_dir=base_dir)
    text = path.read_text(encoding="utf-8")
    assert "\n" in text  # indented, not a single line


def test_load_record_missing_file_raises(base_dir):
    with pytest.raises(RecordNotFoundError):
        load_record(base_dir / "records" / "does-not-exist.json")


def test_load_record_invalid_json_raises(base_dir):
    ensure_directories(base_dir)
    bad_file = base_dir / "records" / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SerializationError):
        load_record(bad_file)


# --------------------------------------------------------------------------
# list_records
# --------------------------------------------------------------------------


def test_list_records_returns_all_by_default(documentation, base_dir):
    save_record(route_record(documentation, make_decision(DecisionType.AUTO_SAVE)), call_id="K-001", base_dir=base_dir)
    save_record(route_record(documentation, make_decision(DecisionType.ESCALATE)), call_id="K-002", base_dir=base_dir)

    records = list_records(base_dir=base_dir)
    assert len(records) == 2


def test_list_records_filters_by_status(documentation, base_dir):
    save_record(route_record(documentation, make_decision(DecisionType.AUTO_SAVE)), call_id="K-001", base_dir=base_dir)
    save_record(route_record(documentation, make_decision(DecisionType.ESCALATE)), call_id="K-002", base_dir=base_dir)

    escalated_only = list_records(status=ReviewStatus.ESCALATED, base_dir=base_dir)
    assert len(escalated_only) == 1
    assert escalated_only[0].status == ReviewStatus.ESCALATED


def test_list_records_empty_when_nothing_saved(base_dir):
    assert list_records(base_dir=base_dir) == []


# --------------------------------------------------------------------------
# delete_record
# --------------------------------------------------------------------------


def test_delete_record_removes_file(auto_save_review, base_dir):
    path = save_record(auto_save_review, call_id="K-001", base_dir=base_dir)
    assert path.exists()

    result = delete_record(path)
    assert result is True
    assert not path.exists()


def test_delete_record_missing_file_raises(base_dir):
    with pytest.raises(RecordNotFoundError):
        delete_record(base_dir / "records" / "does-not-exist.json")