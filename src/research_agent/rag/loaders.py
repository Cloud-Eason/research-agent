"""Document loaders: PDF (PyMuPDF), web pages (trafilatura), plain text."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from research_agent.rag.models import Document, make_doc_id

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


class LoaderError(RuntimeError):
    pass


def load_pdf(path: Path) -> list[Document]:
    """One Document per non-empty page, so citations can point at page numbers."""
    import pymupdf

    source = str(path.resolve())
    doc_id = make_doc_id(source)
    documents: list[Document] = []
    with pymupdf.open(path) as pdf:
        meta = pdf.metadata or {}
        title = str(meta.get("title") or "").strip() or path.stem
        for index, page in enumerate(pdf):
            text = str(page.get_text("text"))
            if text.strip():
                documents.append(
                    Document(doc_id=doc_id, title=title, source=source, text=text, page=index + 1)
                )
    if not documents:
        raise LoaderError(f"{path}: no extractable text (scanned PDF? needs OCR)")
    return documents


def load_url(url: str) -> Document:
    """Download a page and keep only the main content (no nav, ads, footers)."""
    import trafilatura
    from trafilatura.metadata import extract_metadata

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise LoaderError(f"{url}: download failed")
    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=True, favor_precision=True
    )
    if not text:
        raise LoaderError(f"{url}: no main content extracted")
    meta = extract_metadata(downloaded)
    title = str(getattr(meta, "title", "") or "").strip() or url
    return Document(doc_id=make_doc_id(url), title=title, source=url, text=str(text))


def load_text_file(path: Path) -> Document:
    source = str(path.resolve())
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(doc_id=make_doc_id(source), title=path.stem, source=source, text=text)


def load_any(target: str) -> Iterator[Document]:
    """Load a URL, a file, or every supported file under a directory."""
    if target.startswith(("http://", "https://")):
        yield load_url(target)
        return
    path = Path(target)
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                yield from load_any(str(child))
        return
    if not path.exists():
        raise LoaderError(f"{target}: not found")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from load_pdf(path)
    elif suffix in TEXT_SUFFIXES:
        yield load_text_file(path)
    else:
        raise LoaderError(f"{target}: unsupported file type '{suffix}'")
