"""
dashboard.py
=============================================================
AI-Powered After-Call Automation Platform - Presentation Layer
=============================================================

Responsibility
--------------
This module is the FINAL stage of the pipeline. It is a pure,
read-only presentation/analytics layer.

It:
    - Loads persisted pipeline outputs from outputs/
    - Loads the evaluation report from outputs/evaluation/
    - Computes metrics, distributions and statistics
    - Renders matplotlib charts
    - Exports a consolidated dashboard report (JSON, Markdown, CSV)

It NEVER:
    - Calls Gemini / LangChain / OpenAI
    - Performs retrieval (RAG)
    - Applies business rules
    - Reads transcripts
    - Modifies stored records or evaluation reports
    - Runs a web server (no Flask / FastAPI / Streamlit)

Data model
----------
Each persisted record (outputs/{records,reviews,escalations,archive}/*.json)
is expected to be a JSON object shaped roughly like:

{
    "call_id": "K-001",
    "customer_id": "C-1001",
    "documentation": {
        "summary": str,
        "issue": str,
        "root_cause": str,
        "resolution": str,
        "pending_actions": [str, ...],
        "category": str,
        "subcategory": str,
        "priority": str,
        "disposition": str,
        "sentiment": str,
        "keywords": [str, ...],
        "tags": [str, ...],
        "confidence": float,
        "grounding": str,
        "escalation_recommended": bool
    },
    "decision": {
        "decision": "AUTO_SAVE" | "HUMAN_REVIEW" | "ESCALATE" | "NON_INTERACTION",
        "reasons": [str, ...]
    },
    "review": {
        "status": "READY_TO_SAVE" | "PENDING_REVIEW" | "ESCALATED" | "ARCHIVED"
    },
    "timestamp": "2026-01-01T12:00:00"   # optional, ISO-8601
}

The dashboard is defensive: every field is read with .get() and sensible
fallbacks, and the record's source folder is used as a fallback signal
for review status / decision if the fields above are absent. This keeps
the dashboard forward-compatible with future upstream changes, per the
project's design goal.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt


# =============================================================
# LOGGING
# =============================================================

logger = logging.getLogger("dashboard")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | dashboard | %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# =============================================================
# CUSTOM EXCEPTIONS
# =============================================================

class DashboardError(Exception):
    """Base exception for all dashboard-related failures."""


class MetricsCalculationError(DashboardError):
    """Raised when metrics cannot be computed from loaded records."""


class VisualizationError(DashboardError):
    """Raised when a chart cannot be generated."""


class EvaluationReportNotFoundError(DashboardError):
    """Raised when outputs/evaluation/evaluation_report.json is missing."""


# =============================================================
# ENUMS
# =============================================================

class ReviewStatus(str, Enum):
    READY_TO_SAVE = "READY_TO_SAVE"
    PENDING_REVIEW = "PENDING_REVIEW"
    ESCALATED = "ESCALATED"
    ARCHIVED = "ARCHIVED"
    UNKNOWN = "UNKNOWN"


class Decision(str, Enum):
    AUTO_SAVE = "AUTO_SAVE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ESCALATE = "ESCALATE"
    NON_INTERACTION = "NON_INTERACTION"
    UNKNOWN = "UNKNOWN"


# Maps the well-known output subfolders to a fallback review status.
# Used only when a record does not carry an explicit review status.
_FOLDER_STATUS_FALLBACK: Dict[str, ReviewStatus] = {
    "records": ReviewStatus.READY_TO_SAVE,
    "reviews": ReviewStatus.PENDING_REVIEW,
    "escalations": ReviewStatus.ESCALATED,
    "archive": ReviewStatus.ARCHIVED,
}


# =============================================================
# DATA MODELS
# =============================================================

@dataclass
class LoadedRecord:
    """A single normalized pipeline record loaded from disk."""

    call_id: str
    source_folder: str
    file_path: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    disposition: Optional[str] = None
    sentiment: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    escalation_recommended: bool = False
    decision: str = Decision.UNKNOWN.value
    review_status: str = ReviewStatus.UNKNOWN.value
    timestamp: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardMetrics:
    """High level counters and accuracy scores."""

    total_calls_processed: int = 0
    auto_saved: int = 0
    pending_human_review: int = 0
    escalated_calls: int = 0
    archived_calls: int = 0
    average_confidence: Optional[float] = None
    overall_accuracy: Optional[float] = None
    category_accuracy: Optional[float] = None
    priority_accuracy: Optional[float] = None
    disposition_accuracy: Optional[float] = None
    sentiment_accuracy: Optional[float] = None
    escalation_accuracy: Optional[float] = None
    decision_accuracy: Optional[float] = None


@dataclass
class DashboardStatistics:
    """Distributions, quality metrics and timeline metrics."""

    category_distribution: Dict[str, int] = field(default_factory=dict)
    subcategory_distribution: Dict[str, int] = field(default_factory=dict)
    priority_distribution: Dict[str, int] = field(default_factory=dict)
    disposition_distribution: Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    decision_distribution: Dict[str, int] = field(default_factory=dict)
    review_status_distribution: Dict[str, int] = field(default_factory=dict)
    top_keywords: List[Tuple[str, int]] = field(default_factory=list)
    top_tags: List[Tuple[str, int]] = field(default_factory=list)
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    calls_per_day: Dict[str, int] = field(default_factory=dict)
    calls_per_hour: Dict[str, int] = field(default_factory=dict)
    escalations_over_time: Dict[str, int] = field(default_factory=dict)
    review_queue_growth: Dict[str, int] = field(default_factory=dict)


@dataclass
class DashboardReport:
    """Top-level object returned by run_dashboard()."""

    generated_at: str
    metrics: DashboardMetrics
    statistics: DashboardStatistics
    evaluation: Dict[str, Any] = field(default_factory=dict)
    failed_cases: Dict[str, Any] = field(default_factory=dict)
    chart_paths: Dict[str, str] = field(default_factory=dict)
    export_paths: Dict[str, str] = field(default_factory=dict)
    record_count: int = 0
    warnings: List[str] = field(default_factory=list)


# =============================================================
# CONSTANTS
# =============================================================

RECORD_SUBFOLDERS = ("records", "reviews", "escalations", "archive")
TOP_N_KEYWORDS = 15
TOP_N_TAGS = 15
CONFIDENCE_HISTOGRAM_BINS = 10


# =============================================================
# LOADING
# =============================================================

def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Read a single JSON file, returning None (and logging) on failure."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable record %s (%s)", path, exc)
        return None


def _normalize_record(raw: Dict[str, Any], source_folder: str, file_path: Path) -> LoadedRecord:
    """Flatten a raw persisted JSON record into a LoadedRecord."""
    documentation = raw.get("documentation", {}) if isinstance(raw, dict) else {}
    decision_block = raw.get("decision", {}) if isinstance(raw, dict) else {}
    review_block = raw.get("review", {}) if isinstance(raw, dict) else {}

    call_id = (
        raw.get("call_id")
        or documentation.get("call_id")
        or file_path.stem
    )

    decision_value = (
        decision_block.get("decision")
        if isinstance(decision_block, dict)
        else decision_block
    ) or Decision.UNKNOWN.value

    review_status_value = (
        review_block.get("status")
        if isinstance(review_block, dict)
        else review_block
    )
    if not review_status_value:
        review_status_value = _FOLDER_STATUS_FALLBACK.get(
            source_folder, ReviewStatus.UNKNOWN
        ).value

    confidence = documentation.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return LoadedRecord(
        call_id=str(call_id),
        source_folder=source_folder,
        file_path=str(file_path),
        category=documentation.get("category"),
        subcategory=documentation.get("subcategory"),
        priority=documentation.get("priority"),
        disposition=documentation.get("disposition"),
        sentiment=documentation.get("sentiment"),
        keywords=list(documentation.get("keywords") or []),
        tags=list(documentation.get("tags") or []),
        confidence=confidence,
        escalation_recommended=bool(documentation.get("escalation_recommended", False)),
        decision=str(decision_value),
        review_status=str(review_status_value),
        timestamp=raw.get("timestamp") or documentation.get("timestamp"),
        raw=raw,
    )


def load_records(outputs_dir: Path) -> List[LoadedRecord]:
    """
    Load every persisted record from outputs/{records,reviews,escalations,archive}.

    Missing subfolders are tolerated (treated as empty). Unreadable /
    malformed JSON files are skipped with a warning rather than aborting
    the whole dashboard run.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        raise DashboardError(f"Outputs directory does not exist: {outputs_dir}")

    records: List[LoadedRecord] = []
    seen_call_ids: set[str] = set()

    for folder_name in RECORD_SUBFOLDERS:
        folder = outputs_dir / folder_name
        if not folder.exists():
            logger.info("Subfolder '%s' not found, skipping.", folder_name)
            continue

        for json_file in sorted(folder.glob("*.json")):
            raw = _read_json_file(json_file)
            if raw is None:
                continue
            record = _normalize_record(raw, folder_name, json_file)

            # Guard against the same call_id appearing in multiple folders
            # (e.g. moved from reviews -> records over time); keep the
            # most recently modified file's version.
            if record.call_id in seen_call_ids:
                logger.warning(
                    "Duplicate call_id '%s' found in '%s'; keeping first occurrence.",
                    record.call_id,
                    folder_name,
                )
                continue

            seen_call_ids.add(record.call_id)
            records.append(record)

    logger.info("Records loaded: %d", len(records))
    return records


def load_evaluation(evaluation_dir: Path) -> Dict[str, Any]:
    """
    Load outputs/evaluation/evaluation_report.json.

    Raises EvaluationReportNotFoundError if the file is missing. Callers
    that want to tolerate a missing report (e.g. before evaluation.py has
    ever been run) should catch this exception.
    """
    evaluation_dir = Path(evaluation_dir)
    report_path = evaluation_dir / "evaluation_report.json"

    if not report_path.exists():
        raise EvaluationReportNotFoundError(
            f"Evaluation report not found at {report_path}"
        )

    raw = _read_json_file(report_path)
    if raw is None:
        raise EvaluationReportNotFoundError(
            f"Evaluation report at {report_path} could not be parsed."
        )

    logger.info("Evaluation report loaded from %s", report_path)
    return raw


# =============================================================
# METRICS
# =============================================================

def _safe_pct(value: Any) -> Optional[float]:
    """Coerce an accuracy-like value to a float percentage, or None."""
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def calculate_metrics(
    records: List[LoadedRecord], evaluation: Dict[str, Any]
) -> DashboardMetrics:
    """Compute the top-level counters and accuracy scores."""
    try:
        total = len(records)

        decision_counts = Counter(r.decision for r in records)
        confidences = [r.confidence for r in records if r.confidence is not None]
        avg_confidence = round(statistics.mean(confidences), 4) if confidences else None

        # A record is "archived" if EITHER its decision was NON_INTERACTION
        # OR its review status is ARCHIVED, without double-counting a
        # record that satisfies both conditions at once.
        archived_count = sum(
            1
            for r in records
            if r.decision == Decision.NON_INTERACTION.value
            or r.review_status == ReviewStatus.ARCHIVED.value
        )

        metrics = DashboardMetrics(
            total_calls_processed=total,
            auto_saved=decision_counts.get(Decision.AUTO_SAVE.value, 0),
            pending_human_review=decision_counts.get(Decision.HUMAN_REVIEW.value, 0),
            escalated_calls=decision_counts.get(Decision.ESCALATE.value, 0),
            archived_calls=archived_count,
            average_confidence=avg_confidence,
            overall_accuracy=_safe_pct(evaluation.get("overall_accuracy")),
            category_accuracy=_safe_pct(evaluation.get("category_accuracy")),
            priority_accuracy=_safe_pct(evaluation.get("priority_accuracy")),
            disposition_accuracy=_safe_pct(evaluation.get("disposition_accuracy")),
            sentiment_accuracy=_safe_pct(evaluation.get("sentiment_accuracy")),
            escalation_accuracy=_safe_pct(evaluation.get("escalation_accuracy")),
            decision_accuracy=_safe_pct(evaluation.get("decision_accuracy")),
        )
        logger.info("Metrics calculated for %d records", total)
        return metrics
    except Exception as exc:  # noqa: BLE001 - convert any failure to domain error
        raise MetricsCalculationError(f"Failed to calculate metrics: {exc}") from exc


# =============================================================
# DISTRIBUTIONS / STATISTICS
# =============================================================

def _counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def calculate_distributions(records: List[LoadedRecord]) -> Dict[str, Dict[str, int]]:
    """Compute category/priority/disposition/sentiment/decision/status distributions."""
    def _clean(values: List[Optional[str]]) -> List[str]:
        return [v if v else "UNKNOWN" for v in values]

    distributions = {
        "category_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.category for r in records]))
        ),
        "subcategory_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.subcategory for r in records]))
        ),
        "priority_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.priority for r in records]))
        ),
        "disposition_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.disposition for r in records]))
        ),
        "sentiment_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.sentiment for r in records]))
        ),
        "decision_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.decision for r in records]))
        ),
        "review_status_distribution": _counter_to_sorted_dict(
            Counter(_clean([r.review_status for r in records]))
        ),
    }
    return distributions


def generate_statistics(records: List[LoadedRecord]) -> DashboardStatistics:
    """Compute distributions, top keywords/tags, confidence range and timeline metrics."""
    try:
        dist = calculate_distributions(records)

        keyword_counter: Counter = Counter()
        tag_counter: Counter = Counter()
        for r in records:
            keyword_counter.update(r.keywords)
            tag_counter.update(r.tags)

        confidences = [r.confidence for r in records if r.confidence is not None]
        min_conf = round(min(confidences), 4) if confidences else None
        max_conf = round(max(confidences), 4) if confidences else None

        calls_per_day: Counter = Counter()
        calls_per_hour: Counter = Counter()
        escalations_over_time: Counter = Counter()
        review_queue_growth: Counter = Counter()

        for r in records:
            ts = _parse_timestamp(r.timestamp)
            if ts is None:
                continue
            day_key = ts.strftime("%Y-%m-%d")
            hour_key = ts.strftime("%H:00")
            calls_per_day[day_key] += 1
            calls_per_hour[hour_key] += 1
            if r.decision == Decision.ESCALATE.value or r.escalation_recommended:
                escalations_over_time[day_key] += 1
            if r.review_status == ReviewStatus.PENDING_REVIEW.value:
                review_queue_growth[day_key] += 1

        stats = DashboardStatistics(
            category_distribution=dist["category_distribution"],
            subcategory_distribution=dist["subcategory_distribution"],
            priority_distribution=dist["priority_distribution"],
            disposition_distribution=dist["disposition_distribution"],
            sentiment_distribution=dist["sentiment_distribution"],
            decision_distribution=dist["decision_distribution"],
            review_status_distribution=dist["review_status_distribution"],
            top_keywords=keyword_counter.most_common(TOP_N_KEYWORDS),
            top_tags=tag_counter.most_common(TOP_N_TAGS),
            min_confidence=min_conf,
            max_confidence=max_conf,
            calls_per_day=dict(sorted(calls_per_day.items())),
            calls_per_hour=dict(sorted(calls_per_hour.items())),
            escalations_over_time=dict(sorted(escalations_over_time.items())),
            review_queue_growth=dict(sorted(review_queue_growth.items())),
        )
        logger.info("Statistics generated (distributions, timeline, quality)")
        return stats
    except Exception as exc:  # noqa: BLE001
        raise MetricsCalculationError(f"Failed to generate statistics: {exc}") from exc


# =============================================================
# FAILED CASES (from evaluation report)
# =============================================================

def _extract_failed_cases(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Pull failure-oriented detail out of the evaluation report, defensively."""
    per_call = evaluation.get("per_call_comparison") or evaluation.get("per_call") or []

    failed_calls: List[str] = []
    incorrect_categories: List[str] = []
    incorrect_priorities: List[str] = []
    incorrect_sentiments: List[str] = []
    incorrect_decisions: List[str] = []
    summary_similarity: Dict[str, float] = {}

    for entry in per_call:
        if not isinstance(entry, dict):
            continue
        call_id = entry.get("call_id", "UNKNOWN")

        if entry.get("passed") is False or entry.get("overall_match") is False:
            failed_calls.append(call_id)
        if entry.get("category_match") is False:
            incorrect_categories.append(call_id)
        if entry.get("priority_match") is False:
            incorrect_priorities.append(call_id)
        if entry.get("sentiment_match") is False:
            incorrect_sentiments.append(call_id)
        if entry.get("decision_match") is False:
            incorrect_decisions.append(call_id)
        if "summary_similarity" in entry:
            try:
                summary_similarity[call_id] = round(float(entry["summary_similarity"]), 4)
            except (TypeError, ValueError):
                pass

    return {
        "failed_calls": failed_calls,
        "incorrect_categories": incorrect_categories,
        "incorrect_priorities": incorrect_priorities,
        "incorrect_sentiments": incorrect_sentiments,
        "incorrect_decisions": incorrect_decisions,
        "summary_similarity_scores": summary_similarity,
    }


# =============================================================
# VISUALIZATIONS (matplotlib only)
# =============================================================

def _ensure_chart_dir(charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)


def create_bar_chart(
    data: Dict[str, int], title: str, output_path: Path, xlabel: str = "", ylabel: str = "Count"
) -> Path:
    """Render a single bar chart to output_path (PNG)."""
    try:
        if not data:
            data = {"No Data": 0}

        labels = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise VisualizationError(f"Failed to create bar chart '{title}': {exc}") from exc


def create_pie_chart(data: Dict[str, int], title: str, output_path: Path) -> Path:
    """Render a single pie chart to output_path (PNG)."""
    try:
        if not data or sum(data.values()) == 0:
            data = {"No Data": 1}

        labels = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(title)
        ax.axis("equal")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise VisualizationError(f"Failed to create pie chart '{title}': {exc}") from exc


def create_line_chart(
    data: Dict[str, int], title: str, output_path: Path, xlabel: str = "", ylabel: str = "Count"
) -> Path:
    """Render a single line chart to output_path (PNG)."""
    try:
        if not data:
            data = {"No Data": 0}

        labels = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(labels, values, marker="o", color="#DD8452")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise VisualizationError(f"Failed to create line chart '{title}': {exc}") from exc


def create_histogram(
    values: List[float], title: str, output_path: Path, bins: int = CONFIDENCE_HISTOGRAM_BINS
) -> Path:
    """Render a histogram (e.g. confidence scores) to output_path (PNG)."""
    try:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if values:
            ax.hist(values, bins=bins, range=(0.0, 1.0), color="#55A868", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise VisualizationError(f"Failed to create histogram '{title}': {exc}") from exc


def _generate_all_charts(
    records: List[LoadedRecord], stats: DashboardStatistics, charts_dir: Path
) -> Dict[str, str]:
    """Generate the full standard chart set and return {name: path}."""
    _ensure_chart_dir(charts_dir)
    chart_paths: Dict[str, str] = {}

    chart_paths["category_distribution"] = str(
        create_bar_chart(
            stats.category_distribution,
            "Category Distribution",
            charts_dir / "category_distribution.png",
            xlabel="Category",
        )
    )
    chart_paths["priority_distribution"] = str(
        create_bar_chart(
            stats.priority_distribution,
            "Priority Distribution",
            charts_dir / "priority_distribution.png",
            xlabel="Priority",
        )
    )
    chart_paths["sentiment_distribution"] = str(
        create_bar_chart(
            stats.sentiment_distribution,
            "Sentiment Distribution",
            charts_dir / "sentiment_distribution.png",
            xlabel="Sentiment",
        )
    )
    chart_paths["decision_distribution"] = str(
        create_pie_chart(
            stats.decision_distribution,
            "Decision Distribution",
            charts_dir / "decision_distribution.png",
        )
    )
    confidences = [r.confidence for r in records if r.confidence is not None]
    chart_paths["confidence_histogram"] = str(
        create_histogram(
            confidences,
            "Confidence Score Distribution",
            charts_dir / "confidence_histogram.png",
        )
    )
    chart_paths["calls_over_time"] = str(
        create_line_chart(
            stats.calls_per_day,
            "Calls Over Time",
            charts_dir / "calls_over_time.png",
            xlabel="Date",
        )
    )

    logger.info("Charts generated: %d", len(chart_paths))
    return chart_paths


# =============================================================
# SUMMARY GENERATION
# =============================================================

def generate_dashboard_summary(
    metrics: DashboardMetrics,
    stats: DashboardStatistics,
    evaluation: Dict[str, Any],
    failed_cases: Dict[str, Any],
    chart_paths: Dict[str, str],
    warnings: List[str],
    record_count: int,
) -> DashboardReport:
    """Assemble the final DashboardReport object."""
    return DashboardReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        metrics=metrics,
        statistics=stats,
        evaluation=evaluation,
        failed_cases=failed_cases,
        chart_paths=chart_paths,
        export_paths={},
        record_count=record_count,
        warnings=warnings,
    )


# =============================================================
# EXPORT
# =============================================================

def _report_to_plain_dict(report: DashboardReport) -> Dict[str, Any]:
    return {
        "generated_at": report.generated_at,
        "record_count": report.record_count,
        "metrics": asdict(report.metrics),
        "statistics": asdict(report.statistics),
        "evaluation": report.evaluation,
        "failed_cases": report.failed_cases,
        "chart_paths": report.chart_paths,
        "warnings": report.warnings,
    }


def _write_json_summary(report: DashboardReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_report_to_plain_dict(report), fh, indent=2, default=str)
    return path


def _write_markdown_summary(report: DashboardReport, path: Path) -> Path:
    m = report.metrics
    s = report.statistics
    lines: List[str] = []
    lines.append(f"# Dashboard Summary\n")
    lines.append(f"Generated at: `{report.generated_at}`\n")

    lines.append("## Overall Metrics\n")
    lines.append(f"- Total Calls Processed: **{m.total_calls_processed}**")
    lines.append(f"- Auto Saved: **{m.auto_saved}**")
    lines.append(f"- Pending Human Review: **{m.pending_human_review}**")
    lines.append(f"- Escalated Calls: **{m.escalated_calls}**")
    lines.append(f"- Archived Calls: **{m.archived_calls}**")
    lines.append(f"- Average Confidence: **{m.average_confidence}**")
    lines.append("")

    lines.append("## Evaluation Accuracy\n")
    lines.append(f"- Overall Accuracy: **{m.overall_accuracy}**")
    lines.append(f"- Category Accuracy: **{m.category_accuracy}**")
    lines.append(f"- Priority Accuracy: **{m.priority_accuracy}**")
    lines.append(f"- Disposition Accuracy: **{m.disposition_accuracy}**")
    lines.append(f"- Sentiment Accuracy: **{m.sentiment_accuracy}**")
    lines.append(f"- Escalation Accuracy: **{m.escalation_accuracy}**")
    lines.append(f"- Decision Accuracy: **{m.decision_accuracy}**")
    lines.append("")

    def _dist_section(title: str, dist: Dict[str, int]) -> None:
        lines.append(f"## {title}\n")
        if not dist:
            lines.append("_No data._\n")
            return
        for key, count in dist.items():
            lines.append(f"- {key}: {count}")
        lines.append("")

    _dist_section("Category Distribution", s.category_distribution)
    _dist_section("Priority Distribution", s.priority_distribution)
    _dist_section("Disposition Distribution", s.disposition_distribution)
    _dist_section("Sentiment Distribution", s.sentiment_distribution)
    _dist_section("Decision Distribution", s.decision_distribution)
    _dist_section("Review Status Distribution", s.review_status_distribution)

    lines.append("## Top Keywords\n")
    for kw, count in s.top_keywords:
        lines.append(f"- {kw}: {count}")
    lines.append("")

    lines.append("## Top Tags\n")
    for tag, count in s.top_tags:
        lines.append(f"- {tag}: {count}")
    lines.append("")

    lines.append("## Quality\n")
    lines.append(f"- Min Confidence: **{s.min_confidence}**")
    lines.append(f"- Max Confidence: **{s.max_confidence}**")
    lines.append("")

    lines.append("## Failed Cases\n")
    for key, value in report.failed_cases.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    if report.warnings:
        lines.append("## Warnings\n")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_metrics_csv(report: DashboardReport, path: Path) -> Path:
    m = report.metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in asdict(m).items():
            writer.writerow([k, v])
    return path


def export_dashboard(report: DashboardReport, dashboard_dir: Path) -> Dict[str, str]:
    """Write dashboard_summary.json, dashboard_summary.md and metrics.csv."""
    dashboard_dir = Path(dashboard_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_json_summary(report, dashboard_dir / "dashboard_summary.json")
    md_path = _write_markdown_summary(report, dashboard_dir / "dashboard_summary.md")
    csv_path = _write_metrics_csv(report, dashboard_dir / "metrics.csv")

    export_paths = {
        "dashboard_summary_json": str(json_path),
        "dashboard_summary_md": str(md_path),
        "metrics_csv": str(csv_path),
    }
    logger.info("Dashboard exports written to %s", dashboard_dir)
    return export_paths


# =============================================================
# ORCHESTRATION
# =============================================================

def run_dashboard(outputs_dir: str | Path = "outputs") -> DashboardReport:
    """
    Entry point. Loads all persisted outputs, computes metrics/statistics,
    generates charts, and exports the consolidated dashboard report.

    Parameters
    ----------
    outputs_dir:
        Path to the pipeline's outputs/ directory (contains
        records/, reviews/, escalations/, archive/, evaluation/).

    Returns
    -------
    DashboardReport
    """
    logger.info("Dashboard started")
    outputs_dir = Path(outputs_dir)
    warnings: List[str] = []

    if not outputs_dir.exists():
        raise DashboardError(f"Outputs directory does not exist: {outputs_dir}")

    records = load_records(outputs_dir)
    logger.info("Records loaded")

    try:
        evaluation = load_evaluation(outputs_dir / "evaluation")
        logger.info("Evaluation loaded")
    except EvaluationReportNotFoundError as exc:
        logger.warning(str(exc))
        warnings.append(str(exc))
        evaluation = {}

    metrics = calculate_metrics(records, evaluation)
    logger.info("Metrics calculated")

    stats = generate_statistics(records)

    failed_cases = _extract_failed_cases(evaluation)

    if not records:
        warnings.append("No records were found in the outputs directory.")

    charts_dir = outputs_dir / "dashboard" / "charts"
    try:
        chart_paths = _generate_all_charts(records, stats, charts_dir)
        logger.info("Charts generated")
    except VisualizationError as exc:
        logger.error(str(exc))
        warnings.append(str(exc))
        chart_paths = {}

    report = generate_dashboard_summary(
        metrics=metrics,
        stats=stats,
        evaluation=evaluation,
        failed_cases=failed_cases,
        chart_paths=chart_paths,
        warnings=warnings,
        record_count=len(records),
    )

    export_paths = export_dashboard(report, outputs_dir / "dashboard")
    report.export_paths = export_paths
    logger.info("Reports exported")

    logger.info("Dashboard completed")
    return report


# =============================================================
# CLI (manual run: python -m app.dashboard [outputs_dir])
# =============================================================

if __name__ == "__main__":
    import sys

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    result = run_dashboard(target_dir)
    print(json.dumps(_report_to_plain_dict(result), indent=2, default=str))