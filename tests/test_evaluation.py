"""
Unit and integration tests for app/evaluation.py

Run with:
    pytest tests/test_evaluation.py -v
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.evaluation import (
    AnswerKeyNotFoundError,
    GeneratedRecordNotFoundError,
    CallResult,
    load_answer_key,
    load_generated_records,
    match_records,
    compare_record,
    calculate_metrics,
    evaluate_dataset,
)
from app.documentation_engine import DocumentationRecord
from app.business_rules import BusinessDecision, DecisionType
from app.review import route_record
from app.storage import save_record


# ===========================================================
# UNIT TESTS - load_answer_key
# ===========================================================

def test_load_answer_key_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AnswerKeyNotFoundError):
        load_answer_key(tmp_path / "does_not_exist.json")


def test_load_answer_key_supports_dict_form(tmp_path: Path) -> None:
    path = tmp_path / "answer_key.json"
    path.write_text(json.dumps({"K-001": {"category": "Billing"}}))
    result = load_answer_key(path)
    assert result == {"K-001": {"category": "Billing"}}


def test_load_answer_key_supports_list_form(tmp_path: Path) -> None:
    path = tmp_path / "answer_key.json"
    path.write_text(json.dumps([{"call_id": "K-001", "category": "Billing"}]))
    result = load_answer_key(path)
    assert "K-001" in result
    assert result["K-001"]["category"] == "Billing"


def test_load_answer_key_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "answer_key.json"
    path.write_text("{ not valid json ")
    with pytest.raises(AnswerKeyNotFoundError):
        load_answer_key(path)


def test_load_answer_key_empty_dict_raises(tmp_path: Path) -> None:
    path = tmp_path / "answer_key.json"
    path.write_text(json.dumps({}))
    with pytest.raises(AnswerKeyNotFoundError):
        load_answer_key(path)


# ===========================================================
# UNIT TESTS - load_generated_records
# ===========================================================

def test_load_generated_records_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(GeneratedRecordNotFoundError):
        load_generated_records(tmp_path / "nope")


def test_load_generated_records_empty_directory_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "records"
    empty_dir.mkdir()
    with pytest.raises(GeneratedRecordNotFoundError):
        load_generated_records(empty_dir)


def test_load_generated_records_skips_bad_files(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "good.json").write_text(json.dumps({"call_id": "K-001", "category": "Billing"}))
    (records_dir / "bad.json").write_text("{ broken")
    (records_dir / "no_call_id.json").write_text(json.dumps({"category": "Billing"}))

    result = load_generated_records(records_dir)
    assert list(result.keys()) == ["K-001"]


def test_load_generated_records_scans_all_subfolders(tmp_path: Path) -> None:
    """Records routed into records/, reviews/, escalations/, archive/ must
    all be discovered from a single outputs/ root, not just outputs/records/."""
    outputs_dir = tmp_path / "outputs"
    for folder, call_id in [
        ("records", "K-001"),
        ("reviews", "K-002"),
        ("escalations", "K-003"),
        ("archive", "K-004"),
    ]:
        subdir = outputs_dir / folder
        subdir.mkdir(parents=True)
        (subdir / f"{call_id}.json").write_text(json.dumps({"call_id": call_id}))

    result = load_generated_records(outputs_dir)
    assert set(result.keys()) == {"K-001", "K-002", "K-003", "K-004"}


def test_load_generated_records_ignores_evaluation_and_dashboard_dirs(tmp_path: Path) -> None:
    """Must not treat its own reports (outputs/evaluation, outputs/dashboard)
    as pipeline records on a re-run."""
    outputs_dir = tmp_path / "outputs"
    records_dir = outputs_dir / "records"
    records_dir.mkdir(parents=True)
    (records_dir / "K-001.json").write_text(json.dumps({"call_id": "K-001"}))

    evaluation_dir = outputs_dir / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / "evaluation_report.json").write_text(json.dumps({"call_id": "SHOULD_NOT_APPEAR"}))

    dashboard_dir = outputs_dir / "dashboard"
    dashboard_dir.mkdir()
    (dashboard_dir / "dashboard_summary.json").write_text(json.dumps({"call_id": "SHOULD_NOT_APPEAR_EITHER"}))

    result = load_generated_records(outputs_dir)
    assert list(result.keys()) == ["K-001"]


# ===========================================================
# UNIT TESTS - match_records
# ===========================================================

def test_match_records_flags_missing_generated() -> None:
    answer_key = {"K-001": {}, "K-002": {}}
    generated = {"K-001": {}}
    pairs = match_records(answer_key, generated)
    by_id = {p.call_id: p for p in pairs}
    assert by_id["K-001"].predicted is not None
    assert by_id["K-002"].predicted is None


def test_match_records_ignores_unmatched_generated() -> None:
    answer_key = {"K-001": {}}
    generated = {"K-001": {}, "K-999": {}}
    pairs = match_records(answer_key, generated)
    assert len(pairs) == 1
    assert pairs[0].call_id == "K-001"


# ===========================================================
# UNIT TESTS - compare_record
# ===========================================================

def test_compare_record_perfect_match_is_correct() -> None:
    record = {
        "call_id": "K-001",
        "summary": "Same text",
        "category": "Billing",
        "subcategory": "Refund",
        "priority": "Low",
        "disposition": "Resolved",
        "sentiment": "Neutral",
        "escalation_recommended": False,
        "decision": "AUTO_SAVE",
    }
    result = compare_record("K-001", expected=record, predicted=dict(record))
    assert result.overall_result == CallResult.CORRECT
    assert result.mismatched_fields == []


def test_compare_record_missing_predicted_marks_missing() -> None:
    record = {"summary": "x", "category": "Billing"}
    result = compare_record("K-001", expected=record, predicted=None)
    assert result.overall_result == CallResult.MISSING_GENERATED
    assert set(result.mismatched_fields) == {
        "summary", "category", "subcategory", "priority",
        "disposition", "sentiment", "escalation_recommended", "decision",
    }


def test_compare_record_boolean_normalization() -> None:
    expected = {"escalation_recommended": True}
    predicted = {"escalation_recommended": "true"}
    result = compare_record("K-001", expected=expected, predicted=predicted)
    boolean_comparison = next(c for c in result.field_comparisons if c.field_name == "escalation_recommended")
    assert boolean_comparison.matched is True


def test_compare_record_categorical_case_insensitive() -> None:
    expected = {"category": "billing"}
    predicted = {"category": "BILLING"}
    result = compare_record("K-001", expected=expected, predicted=predicted)
    category_comparison = next(c for c in result.field_comparisons if c.field_name == "category")
    assert category_comparison.matched is True


def test_compare_record_text_below_threshold_mismatches() -> None:
    expected = {"summary": "The cat sat on the mat."}
    predicted = {"summary": "Completely unrelated sentence about rockets."}
    result = compare_record("K-001", expected=expected, predicted=predicted, similarity_threshold=0.75)
    summary_comparison = next(c for c in result.field_comparisons if c.field_name == "summary")
    assert summary_comparison.matched is False


def test_compare_record_nested_schema_extraction() -> None:
    expected = {"category": "Billing", "priority": "High"}
    predicted = {
        "call_id": "K-001",
        "documentation": {"category": "Billing", "priority": "High"},
    }
    result = compare_record("K-001", expected=expected, predicted=predicted)
    assert result.overall_result == CallResult.CORRECT


def test_compare_record_self_nested_decision_field_extracted_correctly() -> None:
    """Regression test: a stored record's top-level 'decision' key wraps a
    dict that itself has a 'decision' key ({"decision": {"decision": "AUTO_SAVE", ...}}).
    The wrapper must never be compared directly against the expected string."""
    expected = {"decision": "AUTO_SAVE"}
    predicted = {
        "decision": {
            "decision": "AUTO_SAVE",
            "reason": "Complete.",
            "triggered_rules": ["Rule 1"],
            "confidence": 0.9,
        }
    }
    result = compare_record("K-001", expected=expected, predicted=predicted)
    decision_comparison = next(c for c in result.field_comparisons if c.field_name == "decision")
    assert decision_comparison.predicted == "AUTO_SAVE"
    assert decision_comparison.matched is True


# ===========================================================
# UNIT TESTS - calculate_metrics
# ===========================================================

def test_calculate_metrics_on_empty_results() -> None:
    overall, field_metrics = calculate_metrics([])
    assert overall.total_calls == 0
    assert overall.overall_accuracy == 0.0
    for metrics in field_metrics.values():
        assert metrics.total == 0
        assert metrics.accuracy == 0.0


# ===========================================================
# INTEGRATION TEST - flat mock schema (fast, no real pipeline)
# ===========================================================

def test_evaluate_dataset_end_to_end(tmp_path: Path) -> None:
    answer_key_path = tmp_path / "answer_key.json"
    records_dir = tmp_path / "records"
    report_dir = tmp_path / "evaluation"
    records_dir.mkdir()

    answer_key_path.write_text(json.dumps({
        "K-001": {
            "summary": "Refund issued for duplicate charge.",
            "category": "Billing", "subcategory": "Refund", "priority": "Low",
            "disposition": "Resolved", "sentiment": "Neutral",
            "escalation_recommended": False, "decision": "AUTO_SAVE",
        }
    }))
    (records_dir / "K-001.json").write_text(json.dumps({
        "call_id": "K-001",
        "summary": "Refund issued for duplicate charge.",
        "category": "Billing", "subcategory": "Refund", "priority": "Low",
        "disposition": "Resolved", "sentiment": "Neutral",
        "escalation_recommended": False, "decision": "AUTO_SAVE",
        "confidence": 0.9,
    }))

    report = evaluate_dataset(
        answer_key_path=answer_key_path,
        generated_records_path=records_dir,
        report_directory=report_dir,
    )

    assert report.overall_metrics.total_calls == 1
    assert report.overall_metrics.overall_accuracy == 1.0
    assert (report_dir / "evaluation_report.json").exists()
    assert (report_dir / "evaluation_report.md").exists()


# ===========================================================
# REAL INTEGRATION TEST - actual pipeline modules end-to-end
# ===========================================================
#
# This exercises the REAL DocumentationRecord -> BusinessDecision ->
# ReviewResult -> storage.save_record() chain (no mocks), across all
# four decision types (AUTO_SAVE, HUMAN_REVIEW, ESCALATE,
# NON_INTERACTION), so records land in all four output subfolders
# (records/, reviews/, escalations/, archive/) exactly as the real
# pipeline would produce them. This is the test that originally caught
# both the "decision" self-nesting bug and the single-folder scanning
# bug — flat mock dicts could never have caught either.


def _make_documentation(**overrides) -> DocumentationRecord:
    base = dict(
        summary="Synthetic summary for the integration test.",
        issue="Synthetic issue.",
        root_cause="Not stated",
        resolution="Not stated",
        pending_actions=[],
        category="Billing",
        subcategory="Refund",
        priority="Low",
        disposition="Resolved",
        sentiment="Neutral",
        keywords=["billing"],
        tags=["billing"],
        escalation_recommended=False,
        confidence=0.88,
        grounding={
            "summary": ["Transcript Line 1"],
            "issue": ["Transcript Line 1"],
            "category": ["Handbook Rule 1"],
            "priority": ["Handbook Rule 1"],
            "disposition": ["Transcript Line 2"],
            "sentiment": ["Transcript Line 1"],
        },
    )
    base.update(overrides)
    return DocumentationRecord(**base)


def _persist_real_case(call_id: str, decision_type: DecisionType, base_dir: Path, **doc_overrides) -> dict:
    """Build and save one real ReviewResult, returning its matching answer-key entry."""
    documentation = _make_documentation(**doc_overrides)
    decision = BusinessDecision(
        decision=decision_type,
        reason="synthetic",
        triggered_rules=["Rule X"],
        confidence=documentation.confidence,
        timestamp=datetime.now(timezone.utc),
    )
    review_result = route_record(documentation, decision)
    save_record(review_result, call_id=call_id, base_dir=base_dir)

    return {
        "call_id": call_id,
        "summary": documentation.summary,
        "category": documentation.category,
        "subcategory": documentation.subcategory,
        "priority": documentation.priority,
        "disposition": documentation.disposition,
        "sentiment": documentation.sentiment,
        "escalation_recommended": documentation.escalation_recommended,
        "decision": decision_type.value,
    }


def test_evaluate_dataset_real_pipeline_all_decision_types(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    answer_key_path = tmp_path / "answer_key.json"
    report_dir = outputs_dir / "evaluation"

    answer_key = [
        _persist_real_case(
            "K-001", DecisionType.AUTO_SAVE, outputs_dir,
            category="Billing", subcategory="Refund", priority="Low",
            disposition="Resolved", sentiment="Neutral", escalation_recommended=False,
        ),
        _persist_real_case(
            "K-002", DecisionType.HUMAN_REVIEW, outputs_dir,
            category="Technical Support", subcategory="Login Issue", priority="Medium",
            disposition="Unresolved", sentiment="Negative", escalation_recommended=False,
        ),
        _persist_real_case(
            "K-003", DecisionType.ESCALATE, outputs_dir,
            category="Complaint", subcategory="Staff Conduct", priority="High",
            disposition="Escalated", sentiment="Negative", escalation_recommended=True,
        ),
        _persist_real_case(
            "K-004", DecisionType.NON_INTERACTION, outputs_dir,
            category="Non-interaction", subcategory="Wrong Number", priority="Low",
            disposition="Non-interaction", sentiment="Neutral", escalation_recommended=False,
        ),
    ]
    answer_key_path.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")

    # Sanity check: records really did land in four different subfolders.
    assert (outputs_dir / "records" / "K-001.json").exists()
    assert (outputs_dir / "reviews" / "K-002.json").exists()
    assert (outputs_dir / "escalations" / "K-003.json").exists()
    assert (outputs_dir / "archive" / "K-004.json").exists()

    report = evaluate_dataset(
        answer_key_path=answer_key_path,
        generated_records_path=outputs_dir,
        report_directory=report_dir,
    )

    assert report.overall_metrics.total_calls == 4
    assert report.overall_metrics.correct_predictions == 4
    assert report.overall_metrics.incorrect_predictions == 0
    assert report.overall_metrics.overall_accuracy == 1.0

    # The regression case: decision accuracy must be perfect, not zero.
    assert report.field_metrics["decision"].accuracy == 1.0
    assert report.field_metrics["category"].accuracy == 1.0

    for result in report.per_call_results:
        assert result.overall_result == CallResult.CORRECT
        assert result.mismatched_fields == []

    assert (report_dir / "evaluation_report.json").exists()
    assert (report_dir / "evaluation_report.md").exists()


def test_evaluate_dataset_detects_real_mismatch(tmp_path: Path) -> None:
    """A genuinely wrong prediction must still be caught as INCORRECT,
    even through the real pipeline's storage schema."""
    outputs_dir = tmp_path / "outputs"
    answer_key_path = tmp_path / "answer_key.json"
    report_dir = outputs_dir / "evaluation"

    entry = _persist_real_case(
        "K-001", DecisionType.AUTO_SAVE, outputs_dir,
        category="Billing", subcategory="Refund", priority="Low",
        disposition="Resolved", sentiment="Neutral", escalation_recommended=False,
    )
    # Corrupt the ground truth so it no longer matches what was saved.
    entry["category"] = "Technical Support"
    answer_key_path.write_text(json.dumps([entry], indent=2), encoding="utf-8")

    report = evaluate_dataset(
        answer_key_path=answer_key_path,
        generated_records_path=outputs_dir,
        report_directory=report_dir,
    )

    assert report.overall_metrics.overall_accuracy == 0.0
    assert report.per_call_results[0].overall_result == CallResult.INCORRECT
    assert "category" in report.per_call_results[0].mismatched_fields
