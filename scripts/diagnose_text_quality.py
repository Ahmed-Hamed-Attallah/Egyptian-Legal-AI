#!/usr/bin/env python3
"""
Diagnose Arabic text corruption in the indexed corpus.

1. Scans Qdrant payloads and reports which documents contain mojibake.
2. For affected PDFs, compares the embedded text layer against a fresh OCR
   pass, proving whether OCR repairs the document before you re-index.

Read-only: nothing is written, deleted, or re-indexed.

Usage:
    python -m scripts.diagnose_text_quality
    python -m scripts.diagnose_text_quality --pdf data/documents/jobs/default/.../file.pdf
    python -m scripts.diagnose_text_quality --limit 4000
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Codepoints that effectively never occur in modern Egyptian legal text.
# Their presence is the signature of a broken PDF ToUnicode mapping.
SUSPECT_CHARS = set("؀؁؂؃؄ؠؽؾؿ۝۞")

ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")
# Apostrophe/quote glued between Arabic letters: 'لكتاب  ->  الكتاب
GLUED_QUOTE = re.compile(r"[\u0621-\u064A]['\"][\u0621-\u064A]")

CORRUPT_RATIO = 0.01  # >1% impossible characters => unusable text layer


def score_text(text: str) -> Dict[str, Any]:
    """Return corruption metrics for a piece of Arabic text."""
    body = text or ""
    total = len(body)
    if total == 0:
        return {"chars": 0, "suspect": 0, "glued": 0, "ratio": 0.0, "verdict": "empty"}

    suspect = sum(1 for ch in body if ch in SUSPECT_CHARS)
    glued = len(GLUED_QUOTE.findall(body))
    arabic = len(ARABIC_LETTER.findall(body))
    ratio = (suspect + glued) / total

    if ratio >= CORRUPT_RATIO:
        verdict = "CORRUPT"
    elif arabic / max(1, total) < 0.35:
        verdict = "weak"
    else:
        verdict = "ok"

    return {
        "chars": total,
        "suspect": suspect,
        "glued": glued,
        "arabic": arabic,
        "ratio": round(ratio, 4),
        "verdict": verdict,
    }


def clip(text: str, width: int = 110) -> str:
    flat = " ".join((text or "").split())
    return flat[:width] + ("…" if len(flat) > width else "")


def source_label(metadata: Dict[str, Any]) -> str:
    for key in ("source_path", "file_path", "storage_key", "file_name", "filename"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "<unknown source>"


# ---------------------------------------------------------------------------
# Qdrant scan
# ---------------------------------------------------------------------------
def scan_corpus(limit: int) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    try:
        from src.vector_store.qdrant_store import QdrantVectorStore
    except Exception as exc:
        return {}, f"Cannot import vector store: {exc}"

    try:
        store = QdrantVectorStore()
        chunks = store.scroll_all(page_size=256, max_points=limit)
    except Exception as exc:
        return {}, f"Cannot read Qdrant collection: {exc}"

    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"chunks": 0, "corrupt": 0, "suspect": 0, "chars": 0, "sources": set(), "sample": ""}
    )

    for chunk in chunks:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        key = str(chunk.document_id)
        group = groups[key]
        metrics = score_text(chunk.content)
        group["chunks"] += 1
        group["chars"] += metrics["chars"]
        group["suspect"] += metrics["suspect"] + metrics["glued"]
        group["sources"].add(source_label(metadata))
        if metrics["verdict"] == "CORRUPT":
            group["corrupt"] += 1
            if not group["sample"]:
                group["sample"] = clip(chunk.content)

    return dict(groups), None


def report_corpus(groups: Dict[str, Dict[str, Any]]) -> List[Path]:
    if not groups:
        print("No chunks found in the vector store.")
        return []

    print("=" * 78)
    print("INDEXED CORPUS — ARABIC TEXT QUALITY")
    print("=" * 78)

    pdf_candidates: List[Path] = []
    for document_id, group in sorted(
        groups.items(), key=lambda kv: kv[1]["corrupt"], reverse=True
    ):
        chunks = group["chunks"]
        corrupt = group["corrupt"]
        share = corrupt / max(1, chunks)
        flag = "CORRUPT" if share >= 0.2 else ("mixed" if corrupt else "ok")
        sources = sorted(group["sources"])

        print(f"\n[{flag:>7}] document {document_id}")
        print(f"          chunks: {chunks}   corrupt: {corrupt} ({share:.0%})")
        print(f"          impossible chars: {group['suspect']} / {group['chars']}")
        for src in sources[:3]:
            print(f"          source: {src}")
            path = Path(src)
            if path.suffix.lower() == ".pdf" and path.exists():
                pdf_candidates.append(path)
        if group["sample"]:
            print(f"          sample: {group['sample']}")

    print("\n" + "=" * 78)
    total_chunks = sum(g["chunks"] for g in groups.values())
    total_corrupt = sum(g["corrupt"] for g in groups.values())
    print(f"TOTAL: {total_corrupt} / {total_chunks} chunks corrupted "
          f"({total_corrupt / max(1, total_chunks):.0%})")
    print("=" * 78)
    return sorted(set(pdf_candidates))


# ---------------------------------------------------------------------------
# PDF probe: embedded text layer vs OCR
# ---------------------------------------------------------------------------
def ocr_image(image_path: Path) -> Tuple[str, Optional[str]]:
    """OCR a rendered page with Tesseract Arabic, via pytesseract or the CLI."""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image, lang="ara"), None
    except ImportError:
        pass
    except Exception as exc:
        return "", f"pytesseract failed: {exc}"

    binary = shutil.which("tesseract")
    if binary is None:
        return "", "Tesseract not installed (sudo apt install tesseract-ocr tesseract-ocr-ara)"

    try:
        completed = subprocess.run(
            [binary, str(image_path), "stdout", "-l", "ara", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            return "", f"tesseract exit {completed.returncode}: {completed.stderr.strip()[:200]}"
        return completed.stdout, None
    except Exception as exc:
        return "", f"tesseract failed: {exc}"


def probe_pdf(path: Path, pages: int) -> None:
    print("\n" + "=" * 78)
    print(f"PDF PROBE — {path}")
    print("=" * 78)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed; cannot probe PDFs.")
        return

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        print(f"Cannot open PDF: {exc}")
        return

    page_count = min(pages, document.page_count)
    print(f"Pages: {document.page_count} (probing {page_count})\n")

    embedded_scores: List[float] = []
    ocr_scores: List[float] = []

    with tempfile.TemporaryDirectory(prefix="mizan_ocr_") as tmp:
        for index in range(page_count):
            page = document[index]
            embedded = page.get_text("text") or ""
            metrics = score_text(embedded)
            embedded_scores.append(metrics["ratio"])

            print(f"--- page {index + 1} ---")
            print(f"  embedded text : {metrics['verdict']:>7} "
                  f"(ratio={metrics['ratio']}, chars={metrics['chars']})")
            print(f"  sample        : {clip(embedded)}")

            image_path = Path(tmp) / f"page_{index + 1}.png"
            try:
                page.get_pixmap(dpi=300).save(str(image_path))
            except Exception as exc:
                print(f"  render failed : {exc}")
                continue

            ocr_text, error = ocr_image(image_path)
            if error:
                print(f"  OCR unavailable: {error}")
                continue

            ocr_metrics = score_text(ocr_text)
            ocr_scores.append(ocr_metrics["ratio"])
            print(f"  OCR (ara)     : {ocr_metrics['verdict']:>7} "
                  f"(ratio={ocr_metrics['ratio']}, chars={ocr_metrics['chars']})")
            print(f"  OCR sample    : {clip(ocr_text)}")

    document.close()

    if not embedded_scores:
        return

    embedded_bad = sum(1 for r in embedded_scores if r >= CORRUPT_RATIO)
    print("\nVERDICT")
    if embedded_bad == 0:
        print("  Embedded text layer is clean. Corruption was introduced AFTER extraction")
        print("  (preprocessing / chunking). Inspect text_cleaner.py and arabic_normalizer.py.")
        return

    print(f"  Embedded text layer is CORRUPT on {embedded_bad}/{len(embedded_scores)} probed pages.")
    if ocr_scores and all(r < CORRUPT_RATIO for r in ocr_scores):
        print("  OCR produces clean Arabic. Re-ingest this document with OCR forced.")
    elif ocr_scores:
        print("  OCR is also imperfect — raise DPI to 400 and verify the 'ara' language pack.")
    else:
        print("  Install Tesseract + Arabic data to confirm the OCR repair path:")
        print("    sudo apt install tesseract-ocr tesseract-ocr-ara")
        print("    pip install pytesseract pillow")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Arabic text corruption.")
    parser.add_argument("--pdf", type=Path, default=None, help="Probe a specific PDF.")
    parser.add_argument("--pages", type=int, default=2, help="Pages to probe per PDF.")
    parser.add_argument("--limit", type=int, default=20000, help="Max Qdrant points to scan.")
    parser.add_argument("--skip-corpus", action="store_true", help="Only probe the given PDF.")
    args = parser.parse_args()

    if not args.skip_corpus:
        groups, error = scan_corpus(args.limit)
        if error:
            print(f"Corpus scan skipped: {error}")
        else:
            candidates = report_corpus(groups)
            if args.pdf is None:
                for path in candidates[:2]:
                    probe_pdf(path, args.pages)

    if args.pdf is not None:
        if not args.pdf.exists():
            print(f"PDF not found: {args.pdf}")
            return 1
        probe_pdf(args.pdf, args.pages)

    return 0


if __name__ == "__main__":
    sys.exit(main())