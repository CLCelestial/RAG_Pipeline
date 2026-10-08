"""
Step 2: read the 5 use case PDFs, clean them, chunk them by section and store them in ChromaDB.

    python ingestion.py            # build / rebuild the vector store
    python ingestion.py --preview  # only show the cleaned sections and chunks, no embedding
"""
import argparse
import re
import shutil
import time

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()   # hides a 'consider pymupdf_layout' hint, not needed here

# ---------------------------------------------------------------- cleaning rules
NOISE = [r"^Page \d+ of \d+$", r"^AI USE CASE INTEGRATION IN TEACHING-LEARNING$",
         r"^Part A: AI Use Case Profile$", r"^\*+$"]
BULLET_CHARS = ["", "●", "•", "", "▪"]
ITEM_START = re.compile(r"^(- |\d{1,2}\)\s|[a-zA-Z]\.\s|Step \d+|Stage \d+)")      # new list item
HEADING = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z&/\-\(\) ]{2,60}?):\s*(.*)$")    # "9. Objectives:"
SECTION_ALIASES = {"AI Application Vertical": "Domain", "Application Domain": "Domain",
                   "Faculty Owner": "Faculty Owners", "Faculty Owner(s)": "Faculty Owners"}
# short metadata sections get merged into one "Overview" chunk per use case
OVERVIEW_SECTIONS = {"Use Case ID", "Use Case Title", "Domain", "Collaborating Organization",
                     "Faculty Owners", "Beneficiaries", "Dataset", "Header"}


class TableRow:
    """One table row. Rows that spill onto the next page are merged back into it."""

    def __init__(self, cells, header):
        self.cells = [(c or "").replace("\n", " ").strip() for c in cells]
        self.header = header

    def extend(self, cells):
        for i, c in enumerate(cells):
            c = (c or "").replace("\n", " ").strip()
            if c:
                self.cells[i] = (self.cells[i] + " " + c).strip()

    def render(self):
        c = [re.sub(r"\s+", " ", x) for x in self.cells]
        if self.header and self.header[0].lower() == "area":       # AI governance table
            return f"- {c[0]} (Relevance: {c[1]}): {c[2]}"
        if self.header and len(self.header) == len(c):
            return "- " + " | ".join(f"{h}: {v}" for h, v in zip(self.header, c) if v)
        return "- " + " | ".join(v for v in c if v)


def use_case_id(path):
    m = re.search(r"KJS-[A-Z]+-\d+", path.name)
    if not m:
        raise ValueError(f"{path.name}: file name must contain the use case id, e.g. KJS-AGR-01")
    return m.group(0)


def extract_elements(pdf_path):
    """Text blocks and table rows of a PDF, in reading order, with page numbers."""
    elements, last_row, header = [], None, None
    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            tables = page.find_tables().tables
            t_rects = [pymupdf.Rect(t.bbox) for t in tables]
            items = []
            for b in page.get_text("blocks"):
                if b[6] != 0 or b[4].strip().startswith("Fig."):          # images, figure captions
                    continue
                if any(pymupdf.Rect(b[:4]).intersects(r) for r in t_rects):
                    continue                                               # read through the table instead
                items.append((b[1], "text", b[4]))
            items += [(t.bbox[1], "table", t) for t in tables]
            items.sort(key=lambda x: x[0])

            for _, kind, obj in items:
                if kind == "text":
                    elements.append((pno, obj))
                    continue
                for row in obj.extract():
                    cells = [(c or "") for c in row]
                    if cells[0].strip().lower() in ("area", "workflow stage"):
                        header = [c.replace("\n", " ").strip() for c in cells]
                    elif not cells[0].strip() and last_row is not None:
                        last_row.extend(cells)
                    else:
                        last_row = TableRow(cells, header)
                        elements.append((pno, last_row))
    return elements


def clean_lines(elements):
    """Drop headers/footers, fix bullets and join lines that were wrapped by the PDF."""
    raw = []
    for pno, el in elements:
        if isinstance(el, TableRow):
            raw.append((pno, el.render(), True))
            continue
        for line in el.split("\n"):
            line = line.strip()
            for b in BULLET_CHARS:
                line = line.replace(b, "-")
            if line and not any(re.match(p, line) for p in NOISE):
                raw.append((pno, line, False))

    out, pending_bullet = [], False
    for pno, line, is_table in raw:
        if is_table:
            out.append([pno, line]); pending_bullet = False
        elif line in ("-", "o"):
            pending_bullet = True
        elif pending_bullet:
            out.append([pno, "- " + line]); pending_bullet = False
        elif not out or HEADING.match(line) or ITEM_START.match(line):
            out.append([pno, line])
        elif out[-1][1].endswith("-") and not out[-1][1].endswith(" -"):
            out[-1][1] += line                       # "human-" + "in-the-loop"
        else:
            out[-1][1] += " " + line
    return [(p, re.sub(r"\s+", " ", t).strip()) for p, t in out]


def split_sections(lines):
    sections, cur = [], {"name": "Header", "pages": set(), "lines": []}
    for pno, line in lines:
        m = HEADING.match(line)
        if m:
            if cur["lines"]:
                sections.append(cur)
            name = SECTION_ALIASES.get(m.group(2).strip(), m.group(2).strip())
            cur = {"name": name, "pages": {pno}, "lines": [m.group(3)] if m.group(3) else []}
        else:
            cur["pages"].add(pno)
            cur["lines"].append(line)
    sections.append(cur)
    return sections


def page_label(pages):
    p = sorted(pages)
    if len(p) > 1 and p[-1] - p[0] == len(p) - 1:
        return f"{p[0]}-{p[-1]}"
    return ", ".join(map(str, p))


# ---------------------------------------------------------------- chunking
def load_corpus(data_dir=config.DATA_DIR):
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {data_dir}. Put the 5 use case PDFs there first.")
    corpus = {}
    for path in pdfs:
        secs = split_sections(clean_lines(extract_elements(path)))
        title = next((" ".join(s["lines"]) for s in secs if s["name"] == "Use Case Title"), path.stem)
        corpus[use_case_id(path)] = {"file": path.name, "title": title, "sections": secs}
    return corpus


def make_doc(uid, title, section, pages, body, file):
    header = f"[{uid} | {title} | Section: {section}]"   # tells the retriever + LLM where the chunk is from
    return Document(page_content=f"{header}\n{body}",
                    metadata={"use_case_id": uid, "title": title, "section": section,
                              "pages": page_label(pages), "source": file})


def build_chunks(corpus):
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
                                              separators=["\n", ". ", "; ", ", ", " "])
    docs = []
    for uid, info in corpus.items():
        overview, ov_pages, body_docs = [], set(), []
        for s in info["sections"]:
            body = "\n".join(s["lines"]).strip()
            if not body:
                continue
            if s["name"] in OVERVIEW_SECTIONS:
                overview.append(f"{s['name']}: {body}")
                ov_pages |= s["pages"]
            else:
                for piece in splitter.split_text(body):
                    body_docs.append(make_doc(uid, info["title"], s["name"], s["pages"], piece, info["file"]))
        docs.append(make_doc(uid, info["title"], "Overview", ov_pages, "\n".join(overview), info["file"]))
        docs.extend(body_docs)
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i
    return docs


def build_naive_chunks(data_dir=config.DATA_DIR):
    """Baseline used in eval.py: raw text, fixed-size chunks, no cleaning, no section headers."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    docs = []
    for path in sorted(data_dir.glob("*.pdf")):
        with pymupdf.open(path) as doc:
            text = "".join(p.get_text() for p in doc)
        docs += [Document(page_content=t, metadata={"use_case_id": use_case_id(path), "source": path.name})
                 for t in splitter.split_text(text)]
    return docs


# ---------------------------------------------------------------- vector store
def build_vectorstore(docs, persist_dir=config.CHROMA_DIR, embeddings=None):
    if persist_dir.exists():
        shutil.rmtree(persist_dir)          # full rebuild, so re-running never duplicates chunks
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings or config.get_embeddings(),
        ids=[f"chunk-{d.metadata['chunk_id']}" for d in docs],
        collection_name=config.COLLECTION,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="print sections/chunks only, skip embedding")
    args = ap.parse_args()

    corpus = load_corpus()
    print(f"{'use case':<12}{'sections':>9}  title")
    for uid, info in corpus.items():
        print(f"{uid:<12}{len(info['sections']):>9}  {info['title'][:70]}")
    missing = set(config.EXPECTED_USE_CASES) - set(corpus)
    if missing:
        print("WARNING: missing use cases:", sorted(missing))

    docs = build_chunks(corpus)
    lens = [len(d.page_content) for d in docs]
    print(f"\n{len(docs)} chunks | avg {sum(lens) / len(lens):.0f} chars | max {max(lens)} chars")
    for uid in corpus:
        print(f"  {uid}: {sum(d.metadata['use_case_id'] == uid for d in docs)} chunks")

    if args.preview:
        sample = next(d for d in docs if d.metadata["section"].startswith("Modern AI Governance"))
        print("\nSample chunk:\n" + sample.page_content[:700] + "\n" + str(sample.metadata))
        return

    print(f"\nEmbedding with Ollama '{config.EMBED_MODEL}' ...")
    t0 = time.time()
    vs = build_vectorstore(docs)
    print(f"Stored {vs._collection.count()} chunks in {config.CHROMA_DIR} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
