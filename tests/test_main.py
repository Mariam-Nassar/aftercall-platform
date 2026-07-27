import warnings
warnings.filterwarnings("ignore")

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app import main as pipeline
from app.intake import TranscriptInput, TranscriptNotFoundError
from app.customer_lookup import Customer, CustomerNotFoundError
from app.documentation_engine import DocumentationRecord
from app.business_rules import BusinessDecision, DecisionType


@pytest.fixture
def fake_transcript() -> TranscriptInput:
    return TranscriptInput(
        call_id="K-001", customer_id="C-8842", timestamp="2026-07-25T14:32:00Z",
        transcript_text="Agent: hi\nCustomer: internet is down",
        file_name="K-001.txt", file_path=Path("data/transcripts/K-001.txt"),
        loaded_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def fake_customer() -> Customer:
    return Customer(customer_id="C-8842", name="Ahmed Fathy", email=None, phone=None, plan="Premium", extra={})


@pytest.fixture
def fake_handbook_context() -> list[dict]:
    return [{"source": "10_categories.md", "content": "Technical Support", "score": 0.9}]


@pytest.fixture
def fake_documentation() -> DocumentationRecord:
    return DocumentationRecord(
        summary="Internet outage reported.", issue="Connectivity down.",
        root_cause="Not stated", resolution="Not stated", pending_actions=[],
        category="Technical Support", subcategory="Connectivity", priority="High",
        disposition="Unresolved", sentiment="Negative", keywords=["internet"], tags=["connectivity"],
        escalation_recommended=False, confidence=0.9,
        grounding={
            "summary": ["Transcript Line 2"], "issue": ["Transcript Line 2"],
            "category": ["Handbook Rule 1"], "priority": ["Handbook Rule 1"],
            "disposition": ["Transcript Line 2"], "sentiment": ["Transcript Line 2"],
        },
    )


@pytest.fixture
def fake_decision() -> BusinessDecision:
    return BusinessDecision(
        decision=DecisionType.AUTO_SAVE, reason="Complete and grounded.",
        triggered_rules=["Rule 1", "Rule 3", "Rule 10"], confidence=0.9,
        timestamp=datetime.now(timezone.utc),
    )


def _patch_happy_path(monkeypatch, transcript, customer, handbook_context, documentation, decision):
    monkeypatch.setattr(pipeline, "load_transcript", lambda path: transcript)
    monkeypatch.setattr(pipeline, "get_customer", lambda file_path, customer_id: customer)
    monkeypatch.setattr(pipeline, "retrieve_rules", lambda *a, **k: handbook_context)
    monkeypatch.setattr(pipeline, "create_documentation", lambda *a, **k: documentation)
    monkeypatch.setattr(pipeline, "evaluate_record", lambda *a, **k: decision)


def test_run_pipeline_success(monkeypatch, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision):
    _patch_happy_path(monkeypatch, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision)
    result = pipeline.run_pipeline(Path("data/transcripts/K-001.txt"))
    assert result.status == "SUCCESS"
    assert result.transcript is fake_transcript
    assert result.decision is fake_decision


def test_process_transcript_delegates_to_run_pipeline(monkeypatch, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision):
    _patch_happy_path(monkeypatch, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision)
    result = pipeline.process_transcript("data/transcripts/K-001.txt")
    assert result.status == "SUCCESS"


def test_pipeline_stops_after_intake_failure(monkeypatch):
    def _raise(*_a, **_k):
        raise TranscriptNotFoundError("missing file")
    monkeypatch.setattr(pipeline, "load_transcript", _raise)

    called = {"customer_lookup": False}
    monkeypatch.setattr(pipeline, "get_customer", lambda *a, **k: called.__setitem__("customer_lookup", True))

    result = pipeline.run_pipeline(Path("does/not/exist.txt"))
    assert result.status.startswith("FAILED: intake")
    assert result.transcript is None
    assert called["customer_lookup"] is False


def test_pipeline_stops_after_customer_lookup_failure(monkeypatch, fake_transcript):
    monkeypatch.setattr(pipeline, "load_transcript", lambda path: fake_transcript)
    def _raise(*_a, **_k):
        raise CustomerNotFoundError("no such customer")
    monkeypatch.setattr(pipeline, "get_customer", _raise)

    called = {"handbook_search": False}
    monkeypatch.setattr(pipeline, "retrieve_rules", lambda *a, **k: called.__setitem__("handbook_search", True))

    result = pipeline.run_pipeline(Path("data/transcripts/K-001.txt"))
    assert result.status.startswith("FAILED: customer_lookup")
    assert result.customer is None
    assert called["handbook_search"] is False


def test_pipeline_stops_when_transcript_has_no_customer_id(monkeypatch, fake_transcript):
    transcript_without_id = pipeline.TranscriptInput(
        call_id=fake_transcript.call_id, customer_id=None, timestamp=fake_transcript.timestamp,
        transcript_text=fake_transcript.transcript_text, file_name=fake_transcript.file_name,
        file_path=fake_transcript.file_path, loaded_at=fake_transcript.loaded_at,
    )
    monkeypatch.setattr(pipeline, "load_transcript", lambda path: transcript_without_id)
    result = pipeline.run_pipeline(Path("data/transcripts/K-001.txt"))
    assert result.status.startswith("FAILED: customer_lookup")
    assert "customer_id" in result.status


def test_pipeline_stops_after_documentation_failure(monkeypatch, fake_transcript, fake_customer, fake_handbook_context):
    monkeypatch.setattr(pipeline, "load_transcript", lambda path: fake_transcript)
    monkeypatch.setattr(pipeline, "get_customer", lambda *a, **k: fake_customer)
    monkeypatch.setattr(pipeline, "retrieve_rules", lambda *a, **k: fake_handbook_context)
    def _raise(*_a, **_k):
        raise RuntimeError("LLM exploded")
    monkeypatch.setattr(pipeline, "create_documentation", _raise)

    called = {"business_rules": False}
    monkeypatch.setattr(pipeline, "evaluate_record", lambda *a, **k: called.__setitem__("business_rules", True))

    result = pipeline.run_pipeline(Path("data/transcripts/K-001.txt"))
    assert result.status.startswith("FAILED: documentation_engine")
    assert result.documentation is None
    assert called["business_rules"] is False


def test_print_summary_success(capsys, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision):
    result = pipeline.PipelineResult(
        transcript=fake_transcript, customer=fake_customer, handbook_context=fake_handbook_context,
        documentation=fake_documentation, decision=fake_decision, execution_time=1.234, status="SUCCESS",
    )
    pipeline.print_summary(result)
    captured = capsys.readouterr()
    assert "SUCCESS" in captured.out
    assert "AUTO_SAVE" in captured.out
    assert "C-8842" in captured.out


def test_print_summary_failure_only_shows_available_fields(capsys):
    result = pipeline.PipelineResult(
        transcript=None, customer=None, handbook_context=None, documentation=None, decision=None,
        execution_time=0.05, status="FAILED: intake - Transcript file not found",
    )
    pipeline.print_summary(result)
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Customer:" not in captured.out


def test_main_uses_default_path_when_no_arg(monkeypatch, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision):
    _patch_happy_path(monkeypatch, fake_transcript, fake_customer, fake_handbook_context, fake_documentation, fake_decision)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    pipeline.main()


def test_main_exits_nonzero_on_failure(monkeypatch):
    def _raise(*_a, **_k):
        raise TranscriptNotFoundError("missing")
    monkeypatch.setattr(pipeline, "load_transcript", _raise)
    monkeypatch.setattr(sys, "argv", ["main.py", "does/not/exist.txt"])

    with pytest.raises(SystemExit) as exc_info:
        pipeline.main()
    assert exc_info.value.code == 1