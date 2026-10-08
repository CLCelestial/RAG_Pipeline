"""
Step 1 check: run this before anything else.

    python check_setup.py

It checks Python, the installed packages, Ollama + the two models, and that the 5 PDFs are in data/.
Every line should say OK before you move on to ingestion.py.
"""
import importlib
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXPECTED = ["KJS-AGR-01", "KJS-CES-01", "KJS-CES-02", "KJS-HLC-01", "KJS-SRS-01"]
PACKAGES = {"langchain_core": "langchain-core", "langchain_ollama": "langchain-ollama",
            "langchain_chroma": "langchain-chroma", "langchain_text_splitters": "langchain-text-splitters",
            "chromadb": "chromadb", "pymupdf": "pymupdf", "rank_bm25": "rank-bm25",
            "streamlit": "streamlit", "pandas": "pandas", "matplotlib": "matplotlib"}
ok_all = True


def report(ok, msg, warn_only=False):
    global ok_all
    if not warn_only:
        ok_all &= ok
    print(("  OK    " if ok else ("  WARN  " if warn_only else "  FAIL  ")) + msg)


print("1. Python")
v = sys.version_info
py_ok = (3, 10) <= (v.major, v.minor) <= (3, 13)
report(py_ok, f"Python {v.major}.{v.minor}.{v.micro} (need 3.10 to 3.13)")
if not py_ok:
    print("        ChromaDB does not run on this Python yet. Install Python 3.12 next to it and rebuild the venv:")
    print("        py -3.12 -m venv .venv   (Windows)   |   python3.12 -m venv .venv   (Mac/Linux)")
report(sys.prefix != sys.base_prefix, f"virtual environment active ({sys.prefix})", warn_only=True)

print("2. Packages")
for mod, pip_name in PACKAGES.items():
    try:
        m = importlib.import_module(mod)
        report(True, f"{pip_name} {getattr(m, '__version__', '')}")
    except Exception as e:
        report(False, f"{pip_name} not importable ({e.__class__.__name__}). Run: pip install -r requirements.txt")

print("3. Ollama")
try:
    import config
    url, models_needed = config.OLLAMA_URL, [config.EMBED_MODEL, config.LLM_MODEL]
except Exception:
    url, models_needed = "http://localhost:11434", ["nomic-embed-text", "llama3.2:3b"]
try:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as r:
        tags = json.load(r)
    names = [m["name"] for m in tags.get("models", [])]
    report(True, f"Ollama server reachable at {url}")
    for need in models_needed:
        have = any(n == need or n == need + ":latest" or n.split(":")[0] == need for n in names)
        report(have, f"model '{need}' pulled" + ("" if have else f"  ->  run: ollama pull {need}"))
except Exception as e:
    report(False, f"Ollama not reachable at {url} ({e}). Start the Ollama app or run: ollama serve")

print("4. Source PDFs in data/")
pdfs = sorted((BASE / "data").glob("*.pdf"))
found = {}
for p in pdfs:
    m = re.search(r"KJS-[A-Z]+-\d+", p.name)
    found[m.group(0) if m else p.name] = p
for uid in EXPECTED:
    report(uid in found, f"{uid}" + (f"  ({found[uid].name})" if uid in found else "  MISSING"))
extra = [k for k in found if k not in EXPECTED]
if extra:
    report(False, f"unexpected files in data/: {extra}")
try:
    import pymupdf
except ImportError:
    pymupdf = None
if pymupdf:
    for uid in EXPECTED:
        if uid in found:
            with pymupdf.open(found[uid]) as d:
                chars = sum(len(pg.get_text()) for pg in d)
                report(chars > 1000, f"{uid}: {len(d)} pages, {chars} characters of text extracted")

print("\nALL CHECKS PASSED - go to Step 2 (python ingestion.py --preview)" if ok_all
      else "\nSome checks failed - fix the FAIL lines above before moving on.")
