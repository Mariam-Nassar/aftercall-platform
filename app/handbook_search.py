"""
app/handbook_search.py

Handbook retrieval module for the AI-Powered After-Call Automation Platform.

Responsibility (and ONLY responsibility):
    Provide a Retrieval-Augmented Generation (RAG) retrieval layer over
    the internal documentation handbook. Given a query (typically a
    call transcript), return the Top-K most relevant handbook chunks,
    each annotated with its source filename and a similarity score.

This module does NOT:
    - call any LLM to generate text
    - summarize, classify, or interpret retrieved content
    - generate CRM records or structured documentation
    - look up customers
    - apply business rules, routing, or review logic

It is a pure retrieval boundary: query in, ranked handbook chunks out.
Generation (turning these chunks into an answer/decision) is strictly
the responsibility of a downstream module.

Runtime requirements:
    - Environment variable GOOGLE_API_KEY must be set for
      GoogleGenerativeAIEmbeddings to authenticate.
    - Packages: langchain-core, langchain-text-splitters,
      langchain-google-genai, langchain-chroma, chromadb.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 150
_DEFAULT_TOP_K = 5
_COLLECTION_NAME = "handbook"
_MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class HandbookDirectoryNotFoundError(Exception):
    """Raised when the handbook directory does not exist."""


class HandbookEmptyError(Exception):
    """Raised when the handbook directory contains no Markdown documents."""


class VectorStoreError(Exception):
    """Raised when the vector store cannot be built, loaded, or queried."""


class EmbeddingConfigurationError(Exception):
    """Raised when the embedding provider is misconfigured (e.g. missing API key)."""


# --------------------------------------------------------------------------
# Result structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HandbookResult:
    """
    A single retrieved handbook chunk.

    Attributes:
        source: Filename of the handbook document the chunk came from.
        content: The exact chunk text as stored in the vector database.
        score: Similarity score for the query (higher means more
            relevant; derived from the vector store's distance metric).
    """

    source: str
    content: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this result to a plain dictionary.

        Returns:
            A dict with keys "source", "content", and "score".
        """
        return {"source": self.source, "content": self.content, "score": self.score}


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------


def load_documents(handbook_dir: Path) -> list[Document]:
    """
    Load every Markdown document from the handbook directory.

    Args:
        handbook_dir: Path to the directory containing handbook
            Markdown files.

    Returns:
        A list of Document objects, one per file, with `page_content`
        set to the raw file text and `metadata` containing at least
        "source" (the filename) and "file_path" (the full path).

    Raises:
        HandbookDirectoryNotFoundError: If the directory does not
            exist or is not a directory.
        HandbookEmptyError: If no Markdown files are found.
    """
    if not handbook_dir.exists() or not handbook_dir.is_dir():
        raise HandbookDirectoryNotFoundError(
            f"Handbook directory not found: {handbook_dir}"
        )

    markdown_files = sorted(handbook_dir.glob("*.md"))
    if not markdown_files:
        raise HandbookEmptyError(
            f"No Markdown documents found in handbook directory: {handbook_dir}"
        )

    documents: list[Document] = []
    for file_path in markdown_files:
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name, "file_path": str(file_path)},
            )
        )

    logger.info("Loaded %d handbook document(s) from %s", len(documents), handbook_dir)
    return documents


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def split_documents(
    documents: list[Document],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split loaded documents into semantic chunks suitable for embedding.

    Args:
        documents: Documents previously produced by load_documents().
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between
            consecutive chunks, used to preserve context across splits.

    Returns:
        A list of chunked Document objects. Each chunk retains the
        "source" and "file_path" metadata of its parent document.

    Raises:
        HandbookEmptyError: If `documents` is empty.
    """
    if not documents:
        raise HandbookEmptyError("Cannot split an empty document list.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_MARKDOWN_SEPARATORS,
    )

    chunks = splitter.split_documents(documents)
    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def _build_embeddings(model: str = _DEFAULT_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:
    """
    Construct the embedding function used for the vector store.

    Args:
        model: Name of the Google Generative AI embedding model to use.

    Returns:
        A configured GoogleGenerativeAIEmbeddings instance.

    Raises:
        EmbeddingConfigurationError: If the GOOGLE_API_KEY environment
            variable is not set.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        raise EmbeddingConfigurationError(
            "GOOGLE_API_KEY environment variable is not set; cannot "
            "create embeddings."
        )

    return GoogleGenerativeAIEmbeddings(model=model)


# --------------------------------------------------------------------------
# Vector store construction / loading
# --------------------------------------------------------------------------


def _vector_store_exists(persist_directory: Path) -> bool:
    """
    Check whether a persisted Chroma vector store already exists on disk.

    Args:
        persist_directory: Directory where the Chroma store is (or
            would be) persisted.

    Returns:
        True if the directory exists and contains at least one file,
        False otherwise.
    """
    return persist_directory.exists() and any(persist_directory.iterdir())


def build_vector_store(
    documents: list[Document],
    persist_directory: Path,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> Chroma:
    """
    Build a new persistent Chroma vector store from document chunks.

    Args:
        documents: Chunked documents to embed and store (typically the
            output of split_documents()).
        persist_directory: Directory where the Chroma store will be
            persisted to disk.
        embeddings: Optional pre-configured embedding function. If not
            provided, one is created from the default model.

    Returns:
        The populated Chroma vector store instance.

    Raises:
        HandbookEmptyError: If `documents` is empty.
        EmbeddingConfigurationError: If embeddings cannot be configured.
        VectorStoreError: If the vector store cannot be created.
    """
    if not documents:
        raise HandbookEmptyError("Cannot build a vector store from zero chunks.")

    embeddings = embeddings or _build_embeddings()
    persist_directory.mkdir(parents=True, exist_ok=True)

    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=_COLLECTION_NAME,
            persist_directory=str(persist_directory),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise VectorStoreError(
            f"Failed to build vector store at {persist_directory}: {exc}"
        ) from exc

    logger.info(
        "Built vector store with %d chunk(s) at %s", len(documents), persist_directory
    )
    return vector_store


def load_vector_store(
    persist_directory: Path,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> Chroma:
    """
    Load an existing persistent Chroma vector store from disk.

    Args:
        persist_directory: Directory containing a previously persisted
            Chroma store.
        embeddings: Optional pre-configured embedding function. If not
            provided, one is created from the default model.

    Returns:
        The loaded Chroma vector store instance.

    Raises:
        VectorStoreError: If no store exists at the given path or it
            cannot be loaded.
        EmbeddingConfigurationError: If embeddings cannot be configured.
    """
    if not _vector_store_exists(persist_directory):
        raise VectorStoreError(
            f"No existing vector store found at {persist_directory}"
        )

    embeddings = embeddings or _build_embeddings()

    try:
        vector_store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise VectorStoreError(
            f"Failed to load vector store at {persist_directory}: {exc}"
        ) from exc

    logger.info("Loaded existing vector store from %s", persist_directory)
    return vector_store


def get_vector_store(
    handbook_dir: Path,
    persist_directory: Path,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """
    Get a ready-to-query vector store, reusing an existing one if present.

    If a Chroma store already exists at `persist_directory`, it is
    loaded as-is. Otherwise, the handbook is loaded from
    `handbook_dir`, chunked, embedded, and persisted.

    Args:
        handbook_dir: Directory containing the handbook Markdown files
            (only read if the store must be built).
        persist_directory: Directory where the Chroma store is (or
            will be) persisted.
        embeddings: Optional pre-configured embedding function.
        chunk_size: Maximum characters per chunk (used only when
            building a new store).
        chunk_overlap: Overlap between chunks (used only when building
            a new store).

    Returns:
        A Chroma vector store ready for similarity search.

    Raises:
        HandbookDirectoryNotFoundError: If the handbook directory does
            not exist and a new store must be built.
        HandbookEmptyError: If the handbook directory has no documents.
        EmbeddingConfigurationError: If embeddings cannot be configured.
        VectorStoreError: If the store cannot be built or loaded.
    """
    if _vector_store_exists(persist_directory):
        return load_vector_store(persist_directory, embeddings=embeddings)

    documents = load_documents(handbook_dir)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return build_vector_store(chunks, persist_directory, embeddings=embeddings)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def _to_handbook_results(
    raw_results: list[tuple[Document, float]],
) -> list[HandbookResult]:
    """
    Convert raw Chroma similarity-search output into HandbookResult objects.

    Args:
        raw_results: List of (Document, distance_score) tuples as
            returned by Chroma's similarity_search_with_score.

    Returns:
        A list of HandbookResult instances, one per input tuple.
    """
    results: list[HandbookResult] = []
    for document, score in raw_results:
        source = document.metadata.get("source", "unknown")
        results.append(
            HandbookResult(source=source, content=document.page_content, score=float(score))
        )
    return results


def search_handbook(
    query: str,
    vector_store: Chroma,
    top_k: int = _DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Perform a similarity search against the handbook vector store.

    Args:
        query: Free-text query to search for (e.g. a call transcript
            or an extracted excerpt of one).
        vector_store: A Chroma vector store, typically produced by
            get_vector_store() or load_vector_store().
        top_k: Maximum number of results to return.

    Returns:
        A list of dictionaries, each with keys "source", "content",
        and "score", ordered from most to least relevant.

    Raises:
        ValueError: If `query` is empty or `top_k` is not positive.
        VectorStoreError: If the similarity search fails.
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    try:
        raw_results = vector_store.similarity_search_with_score(query, k=top_k)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    results = _to_handbook_results(raw_results)
    logger.info("Retrieved %d handbook chunk(s) for query", len(results))
    return [result.to_dict() for result in results]


def retrieve_rules(
    transcript_text: str,
    handbook_dir: Path,
    persist_directory: Path,
    top_k: int = _DEFAULT_TOP_K,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant handbook chunks for a given transcript.

    This is the primary entry point for downstream modules: it takes
    raw transcript text, ensures a vector store is available (building
    it on first use, reusing it afterward), and returns the Top-K
    matching handbook chunks. No generation, summarization, or
    decision-making is performed.

    Args:
        transcript_text: Raw call transcript text to use as the
            retrieval query.
        handbook_dir: Directory containing the handbook Markdown files.
        persist_directory: Directory where the Chroma store is (or
            will be) persisted.
        top_k: Maximum number of handbook chunks to return.
        embeddings: Optional pre-configured embedding function.

    Returns:
        A list of dictionaries with keys "source", "content", and
        "score", ordered from most to least relevant.

    Raises:
        ValueError: If `transcript_text` is empty.
        HandbookDirectoryNotFoundError: If the handbook directory does
            not exist and a new store must be built.
        HandbookEmptyError: If the handbook directory has no documents.
        EmbeddingConfigurationError: If embeddings cannot be configured.
        VectorStoreError: If the store cannot be built, loaded, or
            queried.
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Transcript text must not be empty.")

    vector_store = get_vector_store(
        handbook_dir=handbook_dir,
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
    return search_handbook(transcript_text, vector_store, top_k=top_k)