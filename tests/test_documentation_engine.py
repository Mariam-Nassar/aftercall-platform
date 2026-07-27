import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.intake import TranscriptInput
from app.customer_lookup import Customer
from app import documentation_engine as engine
from app.documentation_engine import (
    build_prompt,
    call_llm,
    parse_json,
    validate_output,
    create_documentation,
    DocumentationRecord,
    LLMResponseError,
    InvalidJSONError,
    ValidationError,
)


@pytest.fixture
def transcript() -> TranscriptInput:
    return TranscriptInput(
        call_id="K-001",
        customer_id="C-8842",
        timestamp="2026-07-25T14:32:00Z",
        transcript_text=(
            "Agent: Thank you for calling support, how can I help?\n"
            "Customer: My internet has been down since this morning.\n"
        ),
        file_name="K-001.txt",
        file_path=Path("data/transcripts/K-001.txt"),
        loaded_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def customer() -> Customer:
    return Customer(
        customer_id="C-8842",
        name="Ahmed Fathy",
        email="ahmed@example.com",
        phone=None,
        plan="Premium",
        extra={},
    )


@pytest.fixture
def handbook_context() -> list[dict]:
    return [
        {"source": "10_categories.md", "content": "Technical Support category.", "score": 0.91},
        {"source": "20_disposition_priority.md", "content": "Outages are High priority.", "score": 0.88},
    ]


@pytest.fixture
def valid_llm_json() -> dict:
    return {
        "summary": "Customer reported an internet outage since this morning.",
        "issue": "Internet connectivity is down.",
        "root_cause": "Not stated",
        "resolution": "Not stated",
        "pending_actions": ["Dispatch technician"],
        "category": "Technical Support",
        "subcategory": "Connectivity",
        "priority": "High",
        "disposition": "Unresolved",
        "sentiment": "Negative",
        "keywords": ["internet", "outage"],
        "tags": ["connectivity"],
        "escalation_recommended": True,
        "confidence": 0.85,
        "grounding": {
            "issue": ["Transcript Line 2"],
            "category": ["Handbook Rule 1"],
            "priority": ["Handbook Rule 2"],
        },
    }


def test_build_prompt_includes_transcript_customer_and_handbook(
    transcript, customer, handbook_context
):
    system_instruction, user_prompt = build_prompt(transcript, customer, handbook_context)
    assert "forbidden from inventing" in system_instruction
    assert "internet has been down" in user_prompt
    assert "Ahmed Fathy" in user_prompt
    assert "10_categories.md" in user_prompt


def test_parse_json_valid(valid_llm_json):
    raw = json.dumps(valid_llm_json)
    parsed = parse_json(raw)
    assert parsed["category"] == "Technical Support"


def test_parse_json_strips_markdown_fences(valid_llm_json):
    raw = "```json\n" + json.dumps(valid_llm_json) + "\n```"
    parsed = parse_json(raw)
    assert parsed["issue"] == valid_llm_json["issue"]


def test_parse_json_invalid_raises():
    with pytest.raises(InvalidJSONError):
        parse_json("this is not json {")


def test_parse_json_non_object_raises():
    with pytest.raises(InvalidJSONError):
        parse_json("[1, 2, 3]")


def test_validate_output_valid_record(valid_llm_json):
    record = validate_output(valid_llm_json)
    assert isinstance(record, DocumentationRecord)
    assert record.category == "Technical Support"
    assert record.escalation_recommended is True
    assert record.confidence == 0.85


def test_validate_output_missing_string_field_defaults_to_not_stated(valid_llm_json):
    del valid_llm_json["root_cause"]
    record = validate_output(valid_llm_json)
    assert record.root_cause == "Not stated"


def test_validate_output_missing_list_field_defaults_to_empty(valid_llm_json):
    del valid_llm_json["pending_actions"]
    record = validate_output(valid_llm_json)
    assert record.pending_actions == []


def test_validate_output_missing_escalation_raises(valid_llm_json):
    del valid_llm_json["escalation_recommended"]
    with pytest.raises(ValidationError):
        validate_output(valid_llm_json)


def test_validate_output_non_numeric_confidence_raises(valid_llm_json):
    valid_llm_json["confidence"] = "high"
    with pytest.raises(ValidationError):
        validate_output(valid_llm_json)


def test_validate_output_confidence_out_of_range_is_clamped(valid_llm_json):
    valid_llm_json["confidence"] = 1.7
    record = validate_output(valid_llm_json)
    assert record.confidence == 1.0


def test_validate_output_invalid_grounding_shape_is_dropped(valid_llm_json):
    valid_llm_json["grounding"] = {"issue": ["Transcript Line 2"], "priority": 123}
    record = validate_output(valid_llm_json)
    assert record.grounding["issue"] == ["Transcript Line 2"]
    assert "priority" not in record.grounding


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    def __init__(self, response_text: str = "", raise_error: bool = False, **_: object) -> None:
        self._response_text = response_text
        self._raise_error = raise_error

    def generate_content(self, _prompt: str):
        if self._raise_error:
            raise RuntimeError("simulated network failure")
        return _FakeResponse(self._response_text)


def test_call_llm_returns_text(monkeypatch, valid_llm_json):
    raw = json.dumps(valid_llm_json)
    monkeypatch.setattr(engine.genai, "GenerativeModel", lambda **kwargs: _FakeModel(response_text=raw))
    result = call_llm("system", "user", api_key=None)
    assert result == raw


def test_call_llm_empty_response_raises(monkeypatch):
    monkeypatch.setattr(engine.genai, "GenerativeModel", lambda **kwargs: _FakeModel(response_text="   "))
    with pytest.raises(LLMResponseError):
        call_llm("system", "user")


def test_call_llm_api_failure_raises(monkeypatch):
    monkeypatch.setattr(engine.genai, "GenerativeModel", lambda **kwargs: _FakeModel(raise_error=True))
    with pytest.raises(LLMResponseError):
        call_llm("system", "user")


def test_create_documentation_full_flow(monkeypatch, transcript, customer, handbook_context, valid_llm_json):
    raw = json.dumps(valid_llm_json)
    monkeypatch.setattr(engine.genai, "GenerativeModel", lambda **kwargs: _FakeModel(response_text=raw))
    record = create_documentation(transcript, customer, handbook_context)
    assert isinstance(record, DocumentationRecord)
    assert record.category == "Technical Support"
    assert record.escalation_recommended is True


def test_create_documentation_propagates_invalid_json(monkeypatch, transcript, customer, handbook_context):
    monkeypatch.setattr(engine.genai, "GenerativeModel", lambda **kwargs: _FakeModel(response_text="not json"))
    with pytest.raises(InvalidJSONError):
        create_documentation(transcript, customer, handbook_context)