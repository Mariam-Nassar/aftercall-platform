"""
app/customer_lookup.py

Customer lookup module for the AI-Powered After-Call Automation Platform.

Responsibility (and ONLY responsibility):
    Load customer records from data/customers.json and provide lookup
    of a single customer by customer_id, returned as a strongly typed,
    immutable Customer object.

This module does NOT:
    - call any LLM or AI service
    - search a handbook or knowledge base
    - apply business rules, routing, or review logic
    - merge or interpret transcript data
    - summarize, classify, or score anything

It is a pure data-access boundary: customer_id in, Customer out.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class CustomerDataError(Exception):
    """Raised when the customer data file is missing, unreadable, or malformed."""


class CustomerNotFoundError(Exception):
    """Raised when a requested customer_id does not exist in the data set."""


# --------------------------------------------------------------------------
# Data structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Customer:
    """
    Immutable representation of a single customer record.

    Attributes:
        customer_id: Unique identifier of the customer.
        name: Full name of the customer.
        email: Contact email address, if available.
        phone: Contact phone number, if available.
        plan: Subscription/service plan name, if available.
        extra: Any additional fields present in the source record that
            are not explicitly modeled above, kept as a read-only dict
            of raw values.
    """

    customer_id: str
    name: str | None
    email: str | None
    phone: str | None
    plan: str | None
    extra: dict[str, Any]


# --------------------------------------------------------------------------
# Required fields for a valid raw customer record
# --------------------------------------------------------------------------

_REQUIRED_FIELD = "customer_id"
_KNOWN_FIELDS = {"customer_id", "name", "email", "phone", "plan"}


# --------------------------------------------------------------------------
# File loading (isolated IO)
# --------------------------------------------------------------------------


def _read_customers_file(file_path: Path) -> list[dict[str, Any]]:
    """
    Read and parse the raw customers JSON file.

    Args:
        file_path: Path to the customers JSON file.

    Returns:
        A list of raw customer records as parsed from JSON.

    Raises:
        CustomerDataError: If the file does not exist, is not valid
            JSON, or its top-level structure is not a list.
    """
    if not file_path.exists() or not file_path.is_file():
        raise CustomerDataError(f"Customer data file not found: {file_path}")

    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CustomerDataError(
            f"Customer data file could not be read: {file_path}"
        ) from exc

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise CustomerDataError(
            f"Customer data file is not valid JSON: {file_path}"
        ) from exc

    if not isinstance(data, list):
        raise CustomerDataError(
            f"Customer data file must contain a JSON array: {file_path}"
        )

    return data


def _validate_record(record: Any, index: int) -> dict[str, Any]:
    """
    Validate that a single raw record has the minimum required structure.

    Args:
        record: Raw record parsed from the JSON array.
        index: Position of the record in the array, used for error
            messages.

    Returns:
        The validated record, unchanged.

    Raises:
        CustomerDataError: If the record is not an object, or is
            missing the required "customer_id" field.
    """
    if not isinstance(record, dict):
        raise CustomerDataError(
            f"Customer record at index {index} must be a JSON object."
        )

    if not record.get(_REQUIRED_FIELD):
        raise CustomerDataError(
            f"Customer record at index {index} is missing required "
            f"field '{_REQUIRED_FIELD}'."
        )

    return record


def _to_customer(record: dict[str, Any]) -> Customer:
    """
    Convert a validated raw record into a Customer dataclass instance.

    Args:
        record: A validated raw customer record.

    Returns:
        A Customer instance. Any fields not explicitly modeled on
        Customer are preserved in the `extra` mapping.
    """
    extra = {key: value for key, value in record.items() if key not in _KNOWN_FIELDS}

    return Customer(
        customer_id=str(record[_REQUIRED_FIELD]),
        name=record.get("name"),
        email=record.get("email"),
        phone=record.get("phone"),
        plan=record.get("plan"),
        extra=extra,
    )


@lru_cache(maxsize=None)
def _load_customers_cached(file_path_str: str) -> tuple[Customer, ...]:
    """
    Load, validate, and convert all customer records, cached in memory.

    The result is cached per unique file path so repeated calls do not
    re-read or re-parse the JSON file. This is the sole in-memory cache
    used by the module; no mutable module-level variables are used.

    Args:
        file_path_str: String form of the path to the customers JSON
            file (must be a string so the result is hashable for
            caching).

    Returns:
        A tuple of Customer instances loaded from the file.

    Raises:
        CustomerDataError: If the file is missing, malformed, or
            contains invalid records.
    """
    file_path = Path(file_path_str)
    logger.info("Loading customer data from %s", file_path)

    raw_records = _read_customers_file(file_path)
    customers = tuple(
        _to_customer(_validate_record(record, index))
        for index, record in enumerate(raw_records)
    )

    logger.info("Loaded %d customer record(s) from %s", len(customers), file_path)
    return customers


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def load_customers(file_path: Path) -> tuple[Customer, ...]:
    """
    Load all customers from the given JSON file.

    Results are cached in memory after the first successful load for a
    given file path, so subsequent calls do not re-read the file from
    disk.

    Args:
        file_path: Path to the customers JSON file
            (e.g. data/customers.json).

    Returns:
        A tuple of all Customer records found in the file.

    Raises:
        CustomerDataError: If the file is missing, malformed, or
            contains invalid records.
    """
    return _load_customers_cached(str(file_path))


def find_customer(file_path: Path, customer_id: str) -> Customer | None:
    """
    Search for a customer by customer_id without raising if not found.

    Args:
        file_path: Path to the customers JSON file.
        customer_id: Identifier of the customer to search for.

    Returns:
        The matching Customer, or None if no customer with that ID
        exists.

    Raises:
        CustomerDataError: If the underlying customer data cannot be
            loaded.
    """
    customers = load_customers(file_path)
    for customer in customers:
        if customer.customer_id == customer_id:
            return customer
    return None


def get_customer(file_path: Path, customer_id: str) -> Customer:
    """
    Retrieve a customer by customer_id, raising if not found.

    Args:
        file_path: Path to the customers JSON file.
        customer_id: Identifier of the customer to retrieve.

    Returns:
        The matching Customer.

    Raises:
        CustomerDataError: If the underlying customer data cannot be
            loaded.
        CustomerNotFoundError: If no customer with the given ID exists.
    """
    customer = find_customer(file_path, customer_id)
    if customer is None:
        raise CustomerNotFoundError(f"Customer not found: {customer_id}")
    return customer


def clear_cache() -> None:
    """
    Clear the in-memory customer data cache.

    Useful for tests or for reloading data after the underlying JSON
    file has changed on disk.
    """
    _load_customers_cached.cache_clear()