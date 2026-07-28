"""
evaluation.py

Evaluation layer for the AI-Powered After-Call Automation Platform.

Responsibility
--------------
Compare pipeline-generated records against the official answer key
(ground truth), compute accuracy metrics per field and overall, and
produce JSON and Markdown evaluation reports.

This module is intentionally decoupled from the rest of the pipeline.
It does not call any LLM, perform retrieval, apply business rules, or
modify generated records or the answer key in any way. It only reads
JSON files from disk and writes evaluation reports.

A future developer can evaluate outputs from any model by pointing
`evaluate_dataset()` at a directory of generated JSON records and an
answer_key.json file with the same schema.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fallback path
    from difflib import SequenceMatcher

    _HAS_RAPIDFUZZ = False


logger = logging.getLogger(__name__)


# ===========================================================
# CONFIGURATION
# ===========================================================

SUMMARY_SIMILARITY_THRESHOLD: float = 0.75
REPORT_DIRECTORY: Path = Path("outputs/evaluation")
ANSWER_KEY_PATH: Path = Path("data/answer_key.json")
GENERATED_RECORDS_PATH: Path = Path("outputs")

JSON_REPORT_FILENAME: str = "evaluation_report.json"
MARKDOWN_REPORT_FILENAME: str = "evaluation_report.md"

PROJECT_TITLE: str = "Exology Pioneer Program — After-Call Automation Platform"


# ===========================================================
# ENUMS
# ===========================================================

class FieldType(str, Enum):
    """Classifies how a field must be compared."""

    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"


class CallResult(str, Enum):
    """Outcome of evaluating a single call."""

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    MISSING_GENERATED = "MISSING_GENERATED"
    MISSING_ANSWER_KEY = "MISSING_ANSWER_KEY"


# Field name -> comparison strategy. Order defines report ordering.
EVALUATED_FIELDS: dict[str, FieldType] = {
    "summary": FieldType.TEXT,
    "category": FieldType.CATEGORICAL,
    "subcategory": FieldType.CATEGORICAL,
    "priority": FieldType.CATEGORICAL,
    "disposition": FieldType.CATEGORICAL,
    "sentiment": FieldType.CATEGORICAL,
    "escalation_recommended": FieldType.BOOLEAN,
    "decision": FieldType.CATEGORICAL,
}


# ===========================================================
# CUSTOM EXCEPTIONS
# ===========================================================

class EvaluationError(Exception):
    """Base exception for all evaluation-related failures."""


class AnswerKeyNotFoundError(EvaluationError):
    """Raised when the answer key file cannot be located or parsed."""


class GeneratedRecordNotFoundError(EvaluationError):
    """Raised when no generated records can be located or parsed."""


class RecordMismatchError(EvaluationError):
    """Raised when a record cannot be safely compared to its counterpart."""


# ===========================================================
# DATACLASSES
# ===========================================================

@dataclass(frozen=True)
class FieldComparison:
    """Result of comparing a single field between expected and predicted."""

    field_name: str
    field_type: FieldType
    expected: Any
    predicted: Any
    matched: bool
    similarity_score: Optional[float] = None


@dataclass(frozen=True)
class CallEvaluationResult:
    """Evaluation outcome for a single call."""

    call_id: str
    expected: dict[str, Any]
    predicted: dict[str, Any]
    field_comparisons: list[FieldComparison]
    matched_fields: list[str]
    mismatched_fields: list[str]
    confidence: Optional[float]
    overall_result: CallResult


@dataclass(frozen=True)
class FieldMetrics:
    """Aggregate accuracy metrics for a single evaluated field."""

    field_name: str
    total: int
    correct: int
    incorrect: int
    accuracy: float


@dataclass(frozen=True)
class OverallMetrics:
    """Aggregate accuracy metrics across the entire dataset."""

    total_calls: int
    correct_predictions: int
    incorrect_predictions: int
    overall_accuracy: float
    summary_similarity_score: float
    average_confidence: Optional[float]


@dataclass(frozen=True)
class EvaluationReport:
    """Final output of the evaluation pipeline."""

    overall_metrics: OverallMetrics
    field_metrics: dict[str, FieldMetrics]
    per_call_results: list[CallEvaluationResult]
    report_paths: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass(frozen=True)
class MatchedPair:
    """A call_id paired with its expected/predicted records, if found."""

    call_id: str
    expected: Optional[dict[str, Any]]
    predicted: Optional[dict[str, Any]]


# ===========================================================
# INTERNAL HELPERS
# ===========================================================

def _extract_field(record: dict[str, Any], field_name: str) -> Any:
    """Look up a field in a record, tolerating nested record structures.

    Checks the top level first. If the top-level value is itself a
    dict that also contains `field_name` as a key (a "self-nesting"
    wrapper, e.g. a top-level "decision" key whose value is
    {"decision": "AUTO_SAVE", "reason": ..., ...}), the nested scalar
    is preferred over the wrapper dict. Otherwise falls back to
    searching one level into any nested dict values. Returns None if
    the field cannot be located. This keeps the module independent of
    any specific storage schema.
    """
    if field_name in record:
        top_value = record[field_name]
        if isinstance(top_value, dict) and field_name in top_value:
            return top_value[field_name]
        return top_value

    for value in record.values():
        if isinstance(value, dict) and field_name in value:
            return value[field_name]

    return None


def _extract_call_id(record: dict[str, Any]) -> Optional[str]:
    """Locate the call_id for a record, searching nested structures too."""
    for key in ("call_id", "callId", "id"):
        value = _extract_field(record, key)
        if value is not None:
            return str(value)
    return None


def _extract_confidence(record: dict[str, Any]) -> Optional[float]:
    """Locate a numeric confidence value for a record, if present."""
    value = _extract_field(record, "confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_similarity(expected: str, predicted: str) -> float:
    """Return a 0.0-1.0 similarity score between two text strings."""
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    if _HAS_RAPIDFUZZ:
        return _rapidfuzz_fuzz.token_sort_ratio(expected, predicted) / 100.0
    return SequenceMatcher(None, expected, predicted).ratio()


def _normalize_categorical(value: Any) -> str:
    """Normalize a categorical value for exact comparison."""
    return str(value).strip().lower() if value is not None else ""


def _normalize_boolean(value: Any) -> Optional[bool]:
    """Normalize a boolean-like value for exact comparison."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1"):
            return True
        if normalized in ("false", "no", "0"):
            return False
    if value is None:
        return None
    return bool(value)


# ===========================================================
# LOADING
# ===========================================================

def load_answer_key(path: Path = ANSWER_KEY_PATH) -> dict[str, dict[str, Any]]:
    """Load the ground-truth answer key, keyed by call_id.

    Supports either a dict keyed by call_id or a list of records each
    containing a call_id field.
    """
    if not path.exists():
        raise AnswerKeyNotFoundError(f"Answer key not found at: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AnswerKeyNotFoundError(f"Failed to read answer key at {path}: {exc}") from exc

    answer_key: dict[str, dict[str, Any]] = {}

    if isinstance(raw_data, dict):
        for key, value in raw_data.items():
            if isinstance(value, dict):
                call_id = _extract_call_id(value) or str(key)
                answer_key[call_id] = value
    elif isinstance(raw_data, list):
        for entry in raw_data:
            if not isinstance(entry, dict):
                continue
            call_id = _extract_call_id(entry)
            if call_id is None:
                logger.warning("Skipping answer key entry with no call_id: %s", entry)
                continue
            answer_key[call_id] = entry
    else:
        raise AnswerKeyNotFoundError(f"Unsupported answer key format in: {path}")

    if not answer_key:
        raise AnswerKeyNotFoundError(f"Answer key at {path} contained no usable records.")

    logger.info("Answer key loaded: %d records", len(answer_key))
    return answer_key


def load_generated_records(directory: Path = GENERATED_RECORDS_PATH) -> dict[str, dict[str, Any]]:
    """Load all generated pipeline records from a directory, keyed by call_id.

    Scans recursively so that records routed by review.py into any of
    its output subfolders (records/, reviews/, escalations/, archive/)
    are all picked up from a single `outputs/` root. Files under an
    "evaluation" or "dashboard" subfolder are skipped, since those hold
    this module's own generated reports rather than pipeline records.
    """
    if not directory.exists() or not directory.is_dir():
        raise GeneratedRecordNotFoundError(f"Generated records directory not found: {directory}")

    excluded_folder_names = {"evaluation", "dashboard"}
    json_files = [
        file_path
        for file_path in sorted(directory.glob("**/*.json"))
        if excluded_folder_names.isdisjoint(part for part in file_path.relative_to(directory).parts)
    ]
    if not json_files:
        raise GeneratedRecordNotFoundError(f"No generated JSON records found in: {directory}")

    generated: dict[str, dict[str, Any]] = {}

    for file_path in json_files:
        try:
            record = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable generated record %s: %s", file_path, exc)
            continue

        if not isinstance(record, dict):
            logger.warning("Skipping non-object generated record: %s", file_path)
            continue

        call_id = _extract_call_id(record)
        if call_id is None:
            logger.warning("Skipping generated record with no call_id: %s", file_path)
            continue

        generated[call_id] = record

    if not generated:
        raise GeneratedRecordNotFoundError(
            f"No valid generated records with a call_id were found in: {directory}"
        )

    logger.info("Generated records loaded: %d records", len(generated))
    return generated


# ===========================================================
# MATCHING
# ===========================================================

def match_records(
    answer_key: dict[str, dict[str, Any]],
    generated_records: dict[str, dict[str, Any]],
) -> list[MatchedPair]:
    """Pair every answer-key call_id with its generated record, if any.

    Generated records with no corresponding answer key entry are logged
    and excluded, since they cannot be scored against ground truth.
    """
    pairs: list[MatchedPair] = []

    for call_id, expected in answer_key.items():
        predicted = generated_records.get(call_id)
        if predicted is None:
            logger.warning("Missing generated record for call_id: %s", call_id)
        pairs.append(MatchedPair(call_id=call_id, expected=expected, predicted=predicted))

    unmatched_generated = set(generated_records.keys()) - set(answer_key.keys())
    for call_id in sorted(unmatched_generated):
        logger.warning("Generated record has no answer key entry, skipping: %s", call_id)

    logger.info("Record comparison completed: %d matched pairs", len(pairs))
    return pairs


# ===========================================================
# COMPARISON
# ===========================================================

def compare_record(
    call_id: str,
    expected: Optional[dict[str, Any]],
    predicted: Optional[dict[str, Any]],
    similarity_threshold: float = SUMMARY_SIMILARITY_THRESHOLD,
) -> CallEvaluationResult:
    """Compare a single expected/predicted record pair across all fields."""
    if expected is None:
        raise RecordMismatchError(f"Cannot compare call_id {call_id}: missing answer key record.")

    if predicted is None:
        comparisons = [
            FieldComparison(
                field_name=name,
                field_type=field_type,
                expected=_extract_field(expected, name),
                predicted=None,
                matched=False,
            )
            for name, field_type in EVALUATED_FIELDS.items()
        ]
        return CallEvaluationResult(
            call_id=call_id,
            expected=expected,
            predicted={},
            field_comparisons=comparisons,
            matched_fields=[],
            mismatched_fields=list(EVALUATED_FIELDS.keys()),
            confidence=None,
            overall_result=CallResult.MISSING_GENERATED,
        )

    comparisons: list[FieldComparison] = []
    matched_fields: list[str] = []
    mismatched_fields: list[str] = []

    for name, field_type in EVALUATED_FIELDS.items():
        expected_value = _extract_field(expected, name)
        predicted_value = _extract_field(predicted, name)
        similarity_score: Optional[float] = None

        if field_type is FieldType.CATEGORICAL:
            matched = _normalize_categorical(expected_value) == _normalize_categorical(predicted_value)
        elif field_type is FieldType.BOOLEAN:
            matched = _normalize_boolean(expected_value) == _normalize_boolean(predicted_value)
        else:  # FieldType.TEXT
            similarity_score = _text_similarity(str(expected_value or ""), str(predicted_value or ""))
            matched = similarity_score >= similarity_threshold

        comparisons.append(
            FieldComparison(
                field_name=name,
                field_type=field_type,
                expected=expected_value,
                predicted=predicted_value,
                matched=matched,
                similarity_score=similarity_score,
            )
        )

        if matched:
            matched_fields.append(name)
        else:
            mismatched_fields.append(name)

    overall_result = CallResult.CORRECT if not mismatched_fields else CallResult.INCORRECT

    return CallEvaluationResult(
        call_id=call_id,
        expected=expected,
        predicted=predicted,
        field_comparisons=comparisons,
        matched_fields=matched_fields,
        mismatched_fields=mismatched_fields,
        confidence=_extract_confidence(predicted),
        overall_result=overall_result,
    )


# ===========================================================
# METRICS
# ===========================================================

def calculate_metrics(
    per_call_results: list[CallEvaluationResult],
) -> tuple[OverallMetrics, dict[str, FieldMetrics]]:
    """Aggregate per-call comparisons into overall and per-field metrics."""
    total_calls = len(per_call_results)
    correct_predictions = sum(1 for r in per_call_results if r.overall_result == CallResult.CORRECT)
    incorrect_predictions = total_calls - correct_predictions
    overall_accuracy = correct_predictions / total_calls if total_calls else 0.0

    field_totals: dict[str, int] = {name: 0 for name in EVALUATED_FIELDS}
    field_correct: dict[str, int] = {name: 0 for name in EVALUATED_FIELDS}

    similarity_scores: list[float] = []
    confidence_values: list[float] = []

    for result in per_call_results:
        if result.confidence is not None:
            confidence_values.append(result.confidence)

        for comparison in result.field_comparisons:
            field_totals[comparison.field_name] += 1
            if comparison.matched:
                field_correct[comparison.field_name] += 1
            if comparison.field_type is FieldType.TEXT and comparison.similarity_score is not None:
                similarity_scores.append(comparison.similarity_score)

    field_metrics: dict[str, FieldMetrics] = {}
    for name in EVALUATED_FIELDS:
        total = field_totals[name]
        correct = field_correct[name]
        field_metrics[name] = FieldMetrics(
            field_name=name,
            total=total,
            correct=correct,
            incorrect=total - correct,
            accuracy=(correct / total) if total else 0.0,
        )

    overall_metrics = OverallMetrics(
        total_calls=total_calls,
        correct_predictions=correct_predictions,
        incorrect_predictions=incorrect_predictions,
        overall_accuracy=overall_accuracy,
        summary_similarity_score=statistics.fmean(similarity_scores) if similarity_scores else 0.0,
        average_confidence=statistics.fmean(confidence_values) if confidence_values else None,
    )

    logger.info(
        "Metrics calculated: overall_accuracy=%.4f total_calls=%d",
        overall_accuracy,
        total_calls,
    )
    return overall_metrics, field_metrics


# ===========================================================
# REPORT GENERATION
# ===========================================================

def _call_result_to_dict(result: CallEvaluationResult) -> dict[str, Any]:
    """Serialize a CallEvaluationResult into a JSON-friendly dict."""
    return {
        "call_id": result.call_id,
        "expected": {name: _extract_field(result.expected, name) for name in EVALUATED_FIELDS},
        "predicted": {name: _extract_field(result.predicted, name) for name in EVALUATED_FIELDS},
        "matched_fields": result.matched_fields,
        "mismatched_fields": result.mismatched_fields,
        "confidence": result.confidence,
        "overall_result": result.overall_result.value,
        "field_comparisons": [
            {
                "field_name": c.field_name,
                "field_type": c.field_type.value,
                "expected": c.expected,
                "predicted": c.predicted,
                "matched": c.matched,
                "similarity_score": c.similarity_score,
            }
            for c in result.field_comparisons
        ],
    }


def generate_json_report(
    overall_metrics: OverallMetrics,
    field_metrics: dict[str, FieldMetrics],
    per_call_results: list[CallEvaluationResult],
    metadata: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Build the JSON evaluation report as a plain dict."""
    return {
        "metadata": metadata,
        "timestamp": timestamp,
        "overall_metrics": asdict(overall_metrics),
        "field_metrics": {name: asdict(metrics) for name, metrics in field_metrics.items()},
        "per_call_results": [_call_result_to_dict(result) for result in per_call_results],
    }


def _generate_recommendations(
    overall_metrics: OverallMetrics, field_metrics: dict[str, FieldMetrics]
) -> list[str]:
    """Produce plain-language recommendations based on weak metrics."""
    recommendations: list[str] = []

    if overall_metrics.overall_accuracy < 0.80:
        recommendations.append(
            "Overall accuracy is below 80%. Review documentation prompts and handbook "
            "retrieval quality before promoting this pipeline configuration."
        )

    for name, metrics in field_metrics.items():
        if metrics.total and metrics.accuracy < 0.70:
            recommendations.append(
                f"Field '{name}' has low accuracy ({metrics.accuracy:.1%}). "
                f"Investigate prompt grounding and handbook context for this field."
            )

    if overall_metrics.summary_similarity_score < SUMMARY_SIMILARITY_THRESHOLD:
        recommendations.append(
            "Average summary similarity is below the configured threshold. "
            "Consider tightening summarization instructions to the LLM."
        )

    if overall_metrics.average_confidence is not None and overall_metrics.average_confidence < 0.6:
        recommendations.append(
            "Average model confidence is low. Consider routing more calls to human review."
        )

    if not recommendations:
        recommendations.append("No significant issues detected. Pipeline is performing within targets.")

    return recommendations


def generate_markdown_report(
    overall_metrics: OverallMetrics,
    field_metrics: dict[str, FieldMetrics],
    per_call_results: list[CallEvaluationResult],
    metadata: dict[str, Any],
    timestamp: str,
) -> str:
    """Build the human-readable Markdown evaluation report."""
    lines: list[str] = []

    lines.append(f"# {PROJECT_TITLE} — Evaluation Report")
    lines.append("")
    lines.append(f"**Evaluation Date:** {timestamp}")
    lines.append(f"**Dataset Size:** {overall_metrics.total_calls} calls")
    lines.append(f"**Overall Accuracy:** {overall_metrics.overall_accuracy:.2%}")
    lines.append("")

    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total Calls | {overall_metrics.total_calls} |")
    lines.append(f"| Correct Predictions | {overall_metrics.correct_predictions} |")
    lines.append(f"| Incorrect Predictions | {overall_metrics.incorrect_predictions} |")
    lines.append(f"| Overall Accuracy | {overall_metrics.overall_accuracy:.2%} |")
    lines.append(f"| Summary Similarity Score | {overall_metrics.summary_similarity_score:.2%} |")
    avg_conf = (
        f"{overall_metrics.average_confidence:.2%}"
        if overall_metrics.average_confidence is not None
        else "N/A"
    )
    lines.append(f"| Average Confidence | {avg_conf} |")
    lines.append("")

    lines.append("## Per-Field Accuracy")
    lines.append("")
    lines.append("| Field | Correct | Incorrect | Total | Accuracy |")
    lines.append("|---|---|---|---|---|")
    for name, metrics in field_metrics.items():
        lines.append(
            f"| {name} | {metrics.correct} | {metrics.incorrect} | "
            f"{metrics.total} | {metrics.accuracy:.2%} |"
        )
    lines.append("")

    failed_calls = [r for r in per_call_results if r.overall_result != CallResult.CORRECT]
    lines.append(f"## Failed Calls ({len(failed_calls)})")
    lines.append("")
    if failed_calls:
        lines.append("| Call ID | Result | Mismatched Fields |")
        lines.append("|---|---|---|")
        for result in failed_calls:
            mismatched = ", ".join(result.mismatched_fields) if result.mismatched_fields else "—"
            lines.append(f"| {result.call_id} | {result.overall_result.value} | {mismatched} |")
    else:
        lines.append("None. All calls matched the answer key.")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for recommendation in _generate_recommendations(overall_metrics, field_metrics):
        lines.append(f"- {recommendation}")
    lines.append("")

    lines.append("## Metadata")
    lines.append("")
    for key, value in metadata.items():
        lines.append(f"- **{key}**: {value}")

    return "\n".join(lines)


def save_reports(
    json_report: dict[str, Any],
    markdown_report: str,
    directory: Path = REPORT_DIRECTORY,
) -> dict[str, Path]:
    """Write the JSON and Markdown reports to disk, creating directories as needed."""
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / JSON_REPORT_FILENAME
    markdown_path = directory / MARKDOWN_REPORT_FILENAME

    json_path.write_text(json.dumps(json_report, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(markdown_report, encoding="utf-8")

    logger.info("Reports generated: %s, %s", json_path, markdown_path)

    return {"json_report": json_path, "markdown_report": markdown_path}


# ===========================================================
# ORCHESTRATION
# ===========================================================

def evaluate_dataset(
    answer_key_path: Path = ANSWER_KEY_PATH,
    generated_records_path: Path = GENERATED_RECORDS_PATH,
    report_directory: Path = REPORT_DIRECTORY,
    similarity_threshold: float = SUMMARY_SIMILARITY_THRESHOLD,
) -> EvaluationReport:
    """Run the full evaluation pipeline and return an EvaluationReport.

    Loads the answer key and generated records, compares every call,
    computes metrics, writes the JSON and Markdown reports, and returns
    a complete EvaluationReport describing the run.
    """
    logger.info("Evaluation started")

    answer_key = load_answer_key(answer_key_path)
    generated_records = load_generated_records(generated_records_path)

    pairs = match_records(answer_key, generated_records)

    per_call_results: list[CallEvaluationResult] = []
    for pair in pairs:
        try:
            result = compare_record(
                call_id=pair.call_id,
                expected=pair.expected,
                predicted=pair.predicted,
                similarity_threshold=similarity_threshold,
            )
        except RecordMismatchError as exc:
            logger.warning("Skipping call_id %s due to comparison error: %s", pair.call_id, exc)
            continue
        per_call_results.append(result)

    overall_metrics, field_metrics = calculate_metrics(per_call_results)

    timestamp = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "project": PROJECT_TITLE,
        "answer_key_path": str(answer_key_path),
        "generated_records_path": str(generated_records_path),
        "similarity_threshold": similarity_threshold,
        "evaluated_fields": list(EVALUATED_FIELDS.keys()),
        "similarity_backend": "rapidfuzz" if _HAS_RAPIDFUZZ else "difflib",
    }

    json_report = generate_json_report(
        overall_metrics=overall_metrics,
        field_metrics=field_metrics,
        per_call_results=per_call_results,
        metadata=metadata,
        timestamp=timestamp,
    )
    markdown_report = generate_markdown_report(
        overall_metrics=overall_metrics,
        field_metrics=field_metrics,
        per_call_results=per_call_results,
        metadata=metadata,
        timestamp=timestamp,
    )

    report_paths = save_reports(json_report, markdown_report, report_directory)

    logger.info("Evaluation completed")

    return EvaluationReport(
        overall_metrics=overall_metrics,
        field_metrics=field_metrics,
        per_call_results=per_call_results,
        report_paths=report_paths,
        metadata=metadata,
        timestamp=timestamp,
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate_dataset()
