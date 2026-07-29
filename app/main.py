"""
app/main.py

Application orchestrator for the AI-Powered After-Call Automation Platform.

Responsibility (and ONLY responsibility):
    Wire together the existing, independently-implemented modules into
    a single end-to-end pipeline, and report the outcome.

This module contains:
    - NO business logic (that lives in business_rules.py)
    - NO AI logic (that lives in documentation_engine.py)
    - NO retrieval logic (that lives in handbook_search.py)
    - NO validation logic (that lives in intake.py / customer_lookup.py)
    - NO routing logic (that lives in review.py)
    - NO persistence logic (that lives in storage.py)

It only calls, in order, the public entry points already exposed by:
    intake.py -> customer_lookup.py -> handbook_search.py ->
    documentation_engine.py -> business_rules.py -> review.py -> storage.py

Pipeline:
    Transcript -> Intake -> Customer Lookup -> Handbook Search ->
    Documentation Engine -> Business Rules -> Review -> Storage ->
    PipelineResult

Usage:
    python app/main.py
    python app/main.py data/transcripts/K-001.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
load_dotenv()

from app.business_rules import BusinessDecision, CONFIDENCE_THRESHOLD, evaluate_record
from app.customer_lookup import Customer, get_customer, CustomerNotFoundError
from app.documentation_engine import (
    DEFAULT_MODEL_NAME,
    DocumentationRecord,
    create_documentation,
)
from app.handbook_search import retrieve_rules
from app.intake import TranscriptInput, load_transcript
from app.review import ReviewResult, route_record
from app.storage import DEFAULT_BASE_DIR as DEFAULT_STORAGE_DIR, save_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --------------------------------------------------------------------------
# Configuration (defaults; all overridable via run_pipeline() arguments)
# --------------------------------------------------------------------------

DEFAULT_TRANSCRIPT_PATH: Path = Path("data/transcripts/K-001.txt")
DEFAULT_CUSTOMERS_FILE: Path = Path("data/customers.json")
DEFAULT_HANDBOOK_DIR: Path = Path("data/handbook")
DEFAULT_VECTOR_STORE_DIR: Path = Path("data/vector_db")
DEFAULT_OUTPUTS_DIR: Path = DEFAULT_STORAGE_DIR
DEFAULT_TOP_K: int = 10  # kept in sync with handbook_search._DEFAULT_TOP_K

_STATUS_SUCCESS: str = "SUCCESS"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class PipelineError(Exception):
    """Base exception for pipeline orchestration failures."""


class PipelineStageError(PipelineError):
    """
    Raised when a single pipeline stage fails.

    Attributes:
        stage: Name of the stage that failed (e.g. "customer_lookup").
        message: The underlying error message from that stage.
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"Stage '{stage}' failed: {message}")


# --------------------------------------------------------------------------
# Output structure
# --------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    The complete outcome of a single after-call automation run.

    Attributes:
        transcript: The loaded transcript, or None if intake failed.
        customer: The resolved customer, or None if lookup did not
            complete.
        handbook_context: Retrieved handbook chunks, or None if
            retrieval did not complete.
        documentation: The generated documentation record, or None if
            generation did not complete.
        decision: The final business routing decision, or None if
            evaluation did not complete.
        review: The routing outcome (queue, status, assignment), or
            None if routing did not complete.
        storage_path: Path of the persisted JSON record, or None if
            persistence did not complete.
        execution_time: Total wall-clock time for the run, in seconds.
        status: "SUCCESS" if every stage completed, otherwise a string
            describing which stage failed and why.
    """

    transcript: TranscriptInput | None
    customer: Customer | None
    handbook_context: list[dict[str, Any]] | None
    documentation: DocumentationRecord | None
    decision: BusinessDecision | None
    review: ReviewResult | None
    storage_path: Path | None
    execution_time: float
    status: str


# --------------------------------------------------------------------------
# Stage execution helper
# --------------------------------------------------------------------------


def _execute_stage(stage_name: str, stage_fn: Callable[[], _T]) -> _T:
    """
    Run a single pipeline stage with consistent logging and error wrapping.

    Args:
        stage_name: Short identifier for the stage, used in logs and
            error messages.
        stage_fn: A zero-argument callable that performs the stage and
            returns its result.

    Returns:
        Whatever `stage_fn` returns.

    Raises:
        PipelineStageError: If `stage_fn` raises any exception.
    """
    try:
        return stage_fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised below
        raise PipelineStageError(stage_name, str(exc)) from exc


# --------------------------------------------------------------------------
# Pipeline stages (thin wrappers over existing module functions)
# --------------------------------------------------------------------------


def _stage_intake(transcript_path: Path) -> TranscriptInput:
    """Run the intake stage: load and validate the transcript file."""
    transcript = _execute_stage("intake", lambda: load_transcript(transcript_path))
    logger.info("Transcript loaded: %s", transcript.file_name)
    return transcript


def _stage_customer_lookup(transcript: TranscriptInput, customers_file: Path) -> Customer:
    """Resolve customer; allow unknown/missing for non-interaction calls."""
    cid = (transcript.customer_id or "").strip()

    if not cid or cid.lower() in ("unknown", "n/a", "none", "null"):
        logger.warning("No valid customer_id (%r); using placeholder customer", cid)
        return Customer(
            customer_id=cid or "unknown",
            name=None,
            email=None,
            phone=None,
            plan=None,
            extra={},
        )

    try:
        customer = _execute_stage(
            "customer_lookup",
            lambda: get_customer(customers_file, cid),
        )
    except PipelineStageError as exc:
        # CustomerNotFoundError comes through as stage error
        if "not found" in str(exc).lower():
            logger.warning("Customer %s not found; using placeholder", cid)
            return Customer(
                customer_id=cid,
                name=None,
                email=None,
                phone=None,
                plan=None,
                extra={},
            )
        raise

    logger.info("Customer found: %s", customer.customer_id)
    return customer

def _stage_handbook_search(
    transcript: TranscriptInput,
    handbook_dir: Path,
    vector_store_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run the handbook search stage: retrieve relevant handbook context via RAG."""
    handbook_context = _execute_stage(
        "handbook_search",
        lambda: retrieve_rules(
            transcript.transcript_text,
            handbook_dir=handbook_dir,
            persist_directory=vector_store_dir,
            top_k=top_k,
        ),
    )
    logger.info("Handbook retrieved: %d chunk(s)", len(handbook_context))
    return handbook_context


def _stage_documentation(
    transcript: TranscriptInput,
    customer: Customer,
    handbook_context: list[dict[str, Any]],
    model_name: str,
    api_key: str | None,
) -> DocumentationRecord:
    """Run the documentation stage: generate a grounded CRM documentation record."""
    documentation = _execute_stage(
        "documentation_engine",
        lambda: create_documentation(
            transcript,
            customer,
            handbook_context,
            model_name=model_name,
            api_key=api_key,
        ),
    )
    logger.info("Documentation created: category=%s", documentation.category)
    return documentation


def _stage_business_rules(
    documentation: DocumentationRecord,
    confidence_threshold: float,
) -> BusinessDecision:
    """Run the business rules stage: deterministically decide the routing outcome."""
    decision = _execute_stage(
        "business_rules",
        lambda: evaluate_record(documentation, confidence_threshold=confidence_threshold),
    )
    logger.info("Business decision completed: %s", decision.decision.value)
    return decision


def _stage_review(
    documentation: DocumentationRecord,
    decision: BusinessDecision,
) -> ReviewResult:
    """Run the review stage: route the record to its queue based on the decision."""
    review_result = _execute_stage("review", lambda: route_record(documentation, decision))
    logger.info(
        "Record routed: queue=%s status=%s",
        review_result.queue.value,
        review_result.status.value,
    )
    return review_result


def _stage_storage(
    review_result: ReviewResult,
    call_id: str | None,
    outputs_dir: Path,
) -> Path:
    """Run the storage stage: persist the routed record as JSON under outputs_dir."""
    storage_path = _execute_stage(
        "storage",
        lambda: save_record(review_result, call_id=call_id, base_dir=outputs_dir),
    )
    logger.info("Record saved: %s", storage_path)
    return storage_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_pipeline(
    transcript_path: Path,
    *,
    customers_file: Path = DEFAULT_CUSTOMERS_FILE,
    handbook_dir: Path = DEFAULT_HANDBOOK_DIR,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> PipelineResult:
    """
    Run the complete after-call automation pipeline for one transcript.

    Executes intake, customer lookup, handbook search, documentation
    generation, business rule evaluation, review routing, and
    persistence in sequence. Execution stops at the first stage that
    fails; any results already obtained from earlier stages are
    preserved in the returned PipelineResult.

    Args:
        transcript_path: Path to the transcript file to process.
        customers_file: Path to the customers JSON file.
        handbook_dir: Path to the handbook Markdown directory.
        vector_store_dir: Path to the persisted handbook vector store.
        outputs_dir: Root directory under which the routed record is
            persisted (records/reviews/escalations/archive).
        top_k: Number of handbook chunks to retrieve.
        model_name: Gemini model name used for documentation generation.
        api_key: Optional explicit Gemini API key.
        confidence_threshold: Minimum confidence required for
            AUTO_SAVE eligibility.

    Returns:
        A PipelineResult describing everything produced, and whether
        the run succeeded.
    """
    started_at = time.perf_counter()
    logger.info("Pipeline started: transcript_path=%s", transcript_path)

    transcript: TranscriptInput | None = None
    customer: Customer | None = None
    handbook_context: list[dict[str, Any]] | None = None
    documentation: DocumentationRecord | None = None
    decision: BusinessDecision | None = None
    review_result: ReviewResult | None = None
    storage_path: Path | None = None

    try:
        transcript = _stage_intake(transcript_path)
        customer = _stage_customer_lookup(transcript, customers_file)
        handbook_context = _stage_handbook_search(
            transcript, handbook_dir, vector_store_dir, top_k
        )
        documentation = _stage_documentation(
            transcript, customer, handbook_context, model_name, api_key
        )
        decision = _stage_business_rules(documentation, confidence_threshold)
        review_result = _stage_review(documentation, decision)
        storage_path = _stage_storage(review_result, transcript.call_id, outputs_dir)
    except PipelineStageError as exc:
        execution_time = time.perf_counter() - started_at
        status = f"FAILED: {exc.stage} - {exc.message}"
        logger.error("Pipeline finished with failure: %s", status)
        logger.info("Execution time: %.3fs", execution_time)
        return PipelineResult(
            transcript=transcript,
            customer=customer,
            handbook_context=handbook_context,
            documentation=documentation,
            decision=decision,
            review=review_result,
            storage_path=storage_path,
            execution_time=execution_time,
            status=status,
        )

    execution_time = time.perf_counter() - started_at
    logger.info("Pipeline finished: status=%s", _STATUS_SUCCESS)
    logger.info("Execution time: %.3fs", execution_time)

    return PipelineResult(
        transcript=transcript,
        customer=customer,
        handbook_context=handbook_context,
        documentation=documentation,
        decision=decision,
        review=review_result,
        storage_path=storage_path,
        execution_time=execution_time,
        status=_STATUS_SUCCESS,
    )


def process_transcript(transcript_path: str | Path) -> PipelineResult:
    """
    Run the pipeline for a single transcript using default configuration.

    This is the convenience entry point used by the CLI: it resolves
    the given path and delegates to run_pipeline() with the module's
    default customers file, handbook directory, and vector store
    directory.

    Args:
        transcript_path: Path (or path-like string) to the transcript
            file to process.

    Returns:
        A PipelineResult describing the outcome of the run.
    """
    return run_pipeline(Path(transcript_path))


# --------------------------------------------------------------------------
# CLI presentation
# --------------------------------------------------------------------------


def print_summary(result: PipelineResult) -> None:
    """
    Print a concise, human-readable summary of a PipelineResult.

    Args:
        result: The pipeline result to summarize.
    """
    print("=" * 60)
    print("AFTER-CALL AUTOMATION - PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Status:          {result.status}")
    print(f"Execution time:  {result.execution_time:.3f}s")

    if result.transcript is not None:
        print(f"Transcript file: {result.transcript.file_name}")
        print(f"Call ID:         {result.transcript.call_id}")

    if result.customer is not None:
        print(f"Customer:        {result.customer.customer_id} ({result.customer.name})")

    if result.handbook_context is not None:
        print(f"Handbook chunks: {len(result.handbook_context)}")

    if result.documentation is not None:
        print(f"Category:        {result.documentation.category}")
        print(f"Priority:        {result.documentation.priority}")
        print(f"Confidence:      {result.documentation.confidence:.2f}")

    if result.decision is not None:
        print(f"Decision:        {result.decision.decision.value}")
        print(f"Reason:          {result.decision.reason}")
        print(f"Triggered rules: {', '.join(result.decision.triggered_rules)}")

    if result.review is not None:
        print(f"Queue:           {result.review.queue.value}")
        print(f"Review status:   {result.review.status.value}")

    if result.storage_path is not None:
        print(f"Saved to:        {result.storage_path}")

    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline entry point.

    Args:
        argv: Argument list to parse; defaults to sys.argv[1:] when
            None.

    Returns:
        Parsed arguments with a single `transcript_path` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Run the after-call automation pipeline for one transcript."
    )
    parser.add_argument(
        "transcript_path",
        nargs="?",
        default=None,
        help=(
            "Path to the transcript file to process. "
            f"Defaults to {DEFAULT_TRANSCRIPT_PATH} if not provided."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """
    Command-line entry point: run the pipeline once and print a summary.

    Accepts an optional transcript path argument; falls back to
    DEFAULT_TRANSCRIPT_PATH when none is supplied. Exits with status
    code 1 if the pipeline did not complete successfully.
    """
    args = _parse_args()
    transcript_path = Path(args.transcript_path) if args.transcript_path else DEFAULT_TRANSCRIPT_PATH

    result = process_transcript(transcript_path)
    print_summary(result)

    if result.status != _STATUS_SUCCESS:
        sys.exit(1)


if __name__ == "__main__":
    main()
