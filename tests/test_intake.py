import pytest
from pathlib import Path

from app.intake import (
    load_transcript,
    extract_metadata,
    TranscriptNotFoundError,
    TranscriptEmptyError,
)


@pytest.fixture
def valid_transcript_file(tmp_path: Path) -> Path:
    content = (
        "Call ID: K-001\n"
        "Customer ID: C-8842\n"
        "Timestamp: 2026-07-25T14:32:00Z\n\n"
        "Agent: Hello, how can I help?\n"
        "Customer: My internet is down.\n"
    )
    file_path = tmp_path / "K-001.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_load_transcript_extracts_metadata(valid_transcript_file):
    result = load_transcript(valid_transcript_file)
    assert result.call_id == "K-001"
    assert result.customer_id == "C-8842"
    assert result.timestamp == "2026-07-25T14:32:00Z"


def test_load_transcript_preserves_text_exactly(valid_transcript_file):
    result = load_transcript(valid_transcript_file)
    assert "Customer: My internet is down." in result.transcript_text


def test_missing_file_raises_not_found():
    with pytest.raises(TranscriptNotFoundError):
        load_transcript(Path("this/file/does/not/exist.txt"))


def test_empty_file_raises_empty_error(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(TranscriptEmptyError):
        load_transcript(empty_file)


def test_missing_metadata_returns_none(tmp_path):
    file_path = tmp_path / "no_metadata.txt"
    file_path.write_text("Agent: Hi\nCustomer: Hello\n", encoding="utf-8")
    result = load_transcript(file_path)
    assert result.call_id is None
    assert result.customer_id is None
    assert result.timestamp is None