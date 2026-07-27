"""
Unit tests for app/evaluation.py

Run with:
    cd test_project && python3 -m pytest tests_evaluation.py -v
"""
import json
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


# ---------------------------------------------------------
# load_answer_key
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# load_generated_records
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# match_records
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# compare_record
# ---------------------------------------------------------

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
    predicted = {"escalation_recommended": "true"}  # string form
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


# ---------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------

def test_calculate_metrics_on_empty_results() -> None:
    overall, field_metrics = calculate_metrics([])
    assert overall.total_calls == 0
    assert overall.overall_accuracy == 0.0
    for metrics in field_metrics.values():
        assert metrics.total == 0
        assert metrics.accuracy == 0.0


# ---------------------------------------------------------
# evaluate_dataset (integration)
# ---------------------------------------------------------

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