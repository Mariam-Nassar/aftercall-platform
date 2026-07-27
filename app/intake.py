"""
app/intake.py

Intake module for the AI-Powered After-Call Automation Platform.

Responsibility (and ONLY responsibility):
    Receive a raw call transcript from disk, validate it, extract any
    metadata that is explicitly present in the file, and return an
    immutable, structured representation of that transcript.

This module does NOT:
    - look up customers
    - search a handbook or knowledge base
    - call any LLM or vector database
    - apply business rules
    - summarize, clean, rewrite, or otherwise alter transcript text
    - persist, route, or review anything

It is a pure "intake" boundary: transcript in, structured object out.
Downstream modules (e.g. customer lookup) consume the TranscriptInput
object produced here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class TranscriptNotFoundError(Exception):
    """Raised when the requested transcript file does not exist."""


class TranscriptEmptyError(Exception):
    """Raised when the transcript file exists but contains no content."""


class TranscriptReadError(Exception):
    """Raised when the transcript file exists but cannot be read."""


# --------------------------------------------------------------------------
# Data structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptInput:
    """
    Immutable representation of a raw, validated call transcript.

    Attributes:
        call_id: Call identifier extracted from the transcript, if present.
        customer_id: Customer identifier extracted from the transcript,
            if present.
        timestamp: Call timestamp extracted from the transcript, if present.
        transcript_text: The complete, unmodified transcript text.
        file_name: Name of the source file (e.g. "K-001.txt").
        file_path: Full path to the source file.
        loaded_at: UTC datetime at which this object was created.
    """

    call_id: str | None
    customer_id: str | None
    timestamp: str | None
    transcript_text: str
    file_name: str
    file_path: Path
    loaded_at: datetime


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_transcript(file_path: Path) -> None:
    """
    Validate that a transcript file exists and is safe to read.

    Args:
        file_path: Path to the transcript file.

    Raises:
        TranscriptNotFoundError: If the file does not exist or is not a
            regular file.
        TranscriptReadError: If the file exists but is not readable
            (e.g. permission denied).
    """
    if not file_path.exists() or not file_path.is_file():
        raise TranscriptNotFoundError(
            f"Transcript file not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8"):
            pass
    except OSError as exc:
        raise TranscriptReadError(
            f"Transcript file could not be read: {file_path}"
        ) from exc


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_transcript(file_path: Path) -> str:
    """
    Read the complete, unmodified contents of a transcript file.

    Args:
        file_path: Path to the transcript file.

    Returns:
        The raw transcript text, exactly as stored on disk.

    Raises:
        TranscriptReadError: If the file cannot be decoded or read.
        TranscriptEmptyError: If the file content is empty or whitespace
            only.
    """
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise TranscriptReadError(
            f"Transcript file is not valid UTF-8: {file_path}"
        ) from exc
    except OSError as exc:
        raise TranscriptReadError(
            f"Transcript file could not be read: {file_path}"
        ) from exc

    if not raw_text.strip():
        raise TranscriptEmptyError(
            f"Transcript file is empty: {file_path}"
        )

    return raw_text


# --------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------


def _extract_field(pattern: str, text: str) -> str | None:
    """
    Extract the first value matching a labeled field pattern.

    Args:
        pattern: Field label to search for at the start of a line
            (e.g. "Call ID").
        text: Transcript text to search within.

    Returns:
        The extracted value with surrounding whitespace stripped, or
        None if the field is not present.
    """
    match = re.search(
        rf"^{re.escape(pattern)}\s*[:\-]\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None

    value = match.group(1).strip()
    return value or None


def extract_metadata(
    transcript_text: str,
) -> tuple[str | None, str | None, str | None]:
    """
    Extract call metadata that is explicitly present in the transcript.

    Recognized labels (case-insensitive): "Call ID", "Customer ID",
    "Timestamp". No value is invented or inferred; missing fields are
    returned as None.

    Args:
        transcript_text: The raw transcript text to inspect.

    Returns:
        A tuple of (call_id, customer_id, timestamp).
    """
    call_id = _extract_field("Call ID", transcript_text)
    customer_id = _extract_field("Customer ID", transcript_text)
    timestamp = _extract_field("Timestamp", transcript_text)
    return call_id, customer_id, timestamp


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def load_transcript(file_path: Path) -> TranscriptInput:
    """
    Load, validate, and structure a single call transcript.

    This is the main entry point of the intake module. It performs no
    business logic: it only prepares the transcript for downstream
    modules such as customer lookup.

    Args:
        file_path: Path to the transcript file to load.

    Returns:
        A TranscriptInput containing the unmodified transcript text and
        any metadata found in it.

    Raises:
        TranscriptNotFoundError: If the file does not exist.
        TranscriptEmptyError: If the file is empty.
        TranscriptReadError: If the file cannot be read or decoded.
    """
    logger.info("Loading transcript: %s", file_path)

    validate_transcript(file_path)
    transcript_text = read_transcript(file_path)
    call_id, customer_id, timestamp = extract_metadata(transcript_text)

    transcript_input = TranscriptInput(
        call_id=call_id,
        customer_id=customer_id,
        timestamp=timestamp,
        transcript_text=transcript_text,
        file_name=file_path.name,
        file_path=file_path,
        loaded_at=datetime.now(timezone.utc),
    )

    logger.info(
        "Transcript loaded: file=%s call_id=%s customer_id=%s",
        file_path.name,
        call_id,
        customer_id,
    )
    return transcript_input
