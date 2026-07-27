import hashlib
import pytest
from pathlib import Path
from langchain_core.embeddings import Embeddings

from app.handbook_search import (
    load_documents,
    split_documents,
    build_vector_store,
    load_vector_store,
    get_vector_store,
    search_handbook,
    retrieve_rules,
    HandbookDirectoryNotFoundError,
    HandbookEmptyError,
    VectorStoreError,
    EmbeddingConfigurationError,
)


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings so tests never call a real API."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def handbook_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "handbook"
    directory.mkdir()
    (directory / "10_categories.md").write_text(
        "# Categories\n\n## Billing\nInvoices, charges, refunds.\n", encoding="utf-8"
    )
    (directory / "20_disposition_priority.md").write_text(
        "# Disposition Priority\n\nHigh priority: outages.\n", encoding="utf-8"
    )
    return directory


@pytest.fixture
def persist_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma_store"


def test_load_documents_reads_all_markdown_files(handbook_dir):
    docs = load_documents(handbook_dir)
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"10_categories.md", "20_disposition_priority.md"}


def test_load_documents_missing_dir_raises():
    with pytest.raises(HandbookDirectoryNotFoundError):
        load_documents(Path("does/not/exist"))


def test_load_documents_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty_handbook"
    empty_dir.mkdir()
    with pytest.raises(HandbookEmptyError):
        load_documents(empty_dir)


def test_split_documents_produces_chunks(handbook_dir):
    docs = load_documents(handbook_dir)
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= len(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata


def test_split_documents_empty_raises():
    with pytest.raises(HandbookEmptyError):
        split_documents([])


def test_build_and_search_round_trip(handbook_dir, persist_dir):
    docs = load_documents(handbook_dir)
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)

    store = build_vector_store(chunks, persist_dir, embeddings=FakeEmbeddings())
    results = search_handbook("billing invoice refund", store, top_k=2)

    assert len(results) <= 2
    for item in results:
        assert set(item.keys()) == {"source", "content", "score"}
        assert isinstance(item["score"], float)


def test_get_vector_store_reuses_existing_store(handbook_dir, persist_dir, monkeypatch):
    get_vector_store(handbook_dir, persist_dir, embeddings=FakeEmbeddings())

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("load_documents should not be called when store exists")

    monkeypatch.setattr("app.handbook_search.load_documents", _fail_if_called)

    store = get_vector_store(handbook_dir, persist_dir, embeddings=FakeEmbeddings())
    results = search_handbook("outage priority", store, top_k=1)
    assert len(results) == 1


def test_load_vector_store_missing_raises(persist_dir):
    with pytest.raises(VectorStoreError):
        load_vector_store(persist_dir, embeddings=FakeEmbeddings())


def test_search_handbook_empty_query_raises(handbook_dir, persist_dir):
    docs = load_documents(handbook_dir)
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    store = build_vector_store(chunks, persist_dir, embeddings=FakeEmbeddings())
    with pytest.raises(ValueError):
        search_handbook("   ", store, top_k=2)


def test_retrieve_rules_empty_transcript_raises(handbook_dir, persist_dir):
    with pytest.raises(ValueError):
        retrieve_rules("", handbook_dir, persist_dir, embeddings=FakeEmbeddings())


def test_retrieve_rules_full_flow(handbook_dir, persist_dir):
    results = retrieve_rules(
        "customer complaining about billing invoice",
        handbook_dir,
        persist_dir,
        top_k=2,
        embeddings=FakeEmbeddings(),
    )
    assert 0 < len(results) <= 2
    assert all("source" in r and "content" in r and "score" in r for r in results)


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EmbeddingConfigurationError):
        from app.handbook_search import _build_embeddings
        _build_embeddings()