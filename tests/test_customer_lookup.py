import json
import pytest
from pathlib import Path

from app.customer_lookup import (
    load_customers,
    find_customer,
    get_customer,
    clear_cache,
    CustomerNotFoundError,
    CustomerDataError,
)


@pytest.fixture
def customers_file(tmp_path: Path) -> Path:
    data = [
        {"customer_id": "C-1", "name": "Ali", "plan": "Basic"},
        {"customer_id": "C-2", "name": "Mona", "email": "mona@x.com"},
    ]
    file_path = tmp_path / "customers.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    clear_cache()
    return file_path


def test_load_customers_returns_all_records(customers_file):
    assert len(load_customers(customers_file)) == 2


def test_get_customer_found(customers_file):
    customer = get_customer(customers_file, "C-1")
    assert customer.name == "Ali"


def test_find_customer_not_found_returns_none(customers_file):
    assert find_customer(customers_file, "C-999") is None


def test_get_customer_not_found_raises(customers_file):
    with pytest.raises(CustomerNotFoundError):
        get_customer(customers_file, "C-999")


def test_missing_file_raises_data_error(tmp_path):
    clear_cache()
    with pytest.raises(CustomerDataError):
        load_customers(tmp_path / "does_not_exist.json")


def test_invalid_json_raises_data_error(tmp_path):
    clear_cache()
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CustomerDataError):
        load_customers(bad_file)