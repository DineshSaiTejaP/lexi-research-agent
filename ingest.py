"""
Ingestion pipeline — loads judgment PDFs, chunks them, and stores in ChromaDB.

Run once before starting the app:
    python ingest.py

The pipeline extracts text from each PDF, splits into overlapping chunks,
embeds with sentence-transformers, and persists to a local ChromaDB store.
"""

import os
import re
import json
from pathlib import Path

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", BASE_DIR / "data" / "judgments"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION = "lexi_judgments"
CHUNK_SIZE = 800    # characters (~200 tokens)
CHUNK_OVERLAP = 100
BATCH_SIZE = 64


def read_pdf(path: Path) -> str:
    """Return full text of a PDF, joining all pages."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


def parse_metadata(text: str, doc_id: str) -> dict:
    """
    Pull structured fields out of the first ~1000 chars of a judgment.
    Falls back gracefully when patterns don't match.
    """
    header = text[:1000]

    year = None
    m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", header)
    if m:
        year = int(m.group(1))

    court = "Unknown Court"
    for pattern in [
        r"(Supreme Court of India)",
        r"(IN THE (?:HIGH COURT|SUPREME COURT)[^\\n]{0,60})",
        r"(High Court of [A-Za-z\s]+)",
        r"(Motor Accident Claims Tribunal[^,\n]{0,40})",
        r"(National Consumer Disputes Redressal Commission)",
    ]:
        m = re.search(pattern, header, re.IGNORECASE)
        if m:
            court = m.group(1).strip()
            break

    case_name = doc_id
    m = re.search(
        r"([A-Z][A-Za-z\s\.&]+?)\s+(?:vs?\.?|versus)\s+([A-Z][A-Za-z\s\.&]+)",
        header,
    )
    if m:
        case_name = f"{m.group(1).strip()} vs {m.group(2).strip()}"

    topics = []
    checks = {
        "unlicensed driver":    r"unlicens|without.{0,10}licen|invalid.{0,10}licen",
        "insurance liability":  r"insur.{0,15}liabilit",
        "contributory negligence": r"contributory negligence",
        "compensation":         r"compensation|quantum|award|damages",
        "commercial vehicle":   r"commercial vehicle|truck|lorry|bus|goods vehicle",
        "death claim":          r"death|deceased|fatal|killed",
        "third party":          r"third.?party",
        "Section 149":          r"section\s*149",
        "Section 166":          r"section\s*166",
    }
    for label, pattern in checks.items():
        if re.search(pattern, text, re.IGNORECASE):
            topics.append(label)

    is_motor = bool(re.search(
        r"motor accident|MACT|road accident|vehicular accident", text, re.IGNORECASE
    ))

    return {
        "doc_id": doc_id,
        "year": year,
        "court": court,
        "case_name": case_name,
        "is_motor_accident": is_motor,
        "topics": json.dumps(topics),
    }


def chunk_document(text: str, meta: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for i, piece in enumerate(splitter.split_text(text)):
        chunk_meta = {**meta, "chunk_index": i, "chunk_id": f"{meta['doc_id']}_c{i:04d}"}
        chunks.append({"text": piece, "meta": chunk_meta})
    return chunks


def main():
    print("Lexi — Ingestion Pipeline")
    print("-" * 40)

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CORPUS_DIR}. Aborting.")
        return

    print(f"Found {len(pdfs)} documents\n")

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    all_texts, all_metas, all_ids = [], [], []

    for pdf_path in pdfs:
        doc_id = pdf_path.stem.upper()  # doc_001.pdf -> DOC_001
        print(f"  {pdf_path.name} ...", end=" ", flush=True)

        try:
            text = read_pdf(pdf_path)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        meta = parse_metadata(text, doc_id)
        chunks = chunk_document(text, meta)

        for c in chunks:
            all_texts.append(c["text"])
            all_metas.append(c["meta"])
            all_ids.append(c["meta"]["chunk_id"])

        print(f"{len(chunks)} chunks — {meta['case_name'][:50]}")

    print(f"\nEmbedding {len(all_texts)} chunks...")
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=all_metas[i : i + BATCH_SIZE],
            ids=all_ids[i : i + BATCH_SIZE],
        )
        done = min(i + BATCH_SIZE, len(all_texts))
        print(f"  {done}/{len(all_texts)}", end="\r")

    print(f"\nDone. {len(pdfs)} documents, {len(all_texts)} chunks indexed.")
    print(f"Vector store: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
