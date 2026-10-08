"""
Step 5: evaluation. Results are written to results/ (CSV files + a bar chart).

    python eval.py                       # retrieval metrics + chunking comparison + LLM answers
    python eval.py --skip-llm            # retrieval only (fast, no LLM calls)
    python eval.py --compare-embeddings nomic-embed-text mxbai-embed-large all-minilm
"""
import argparse
import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langchain_chroma import Chroma

import config
from ingestion import build_chunks, build_naive_chunks, load_corpus
from rag import RAGPipeline, normalize

# (question, expected use case ids, evidence phrases that must be in a retrieved chunk, answer keywords)
# "a|b" in keywords means either one counts. Empty use case list = out of scope, should be refused.
TEST_SET = [
    ("Which organizations are collaborating on the sugarcane irrigation advisory use case?", ["KJS-AGR-01"], ["KIAAR"], ["KIAAR", "Godavari"]),
    ("How many farmers are associated with GBL and KIAAR?", ["KJS-AGR-01"], ["18,000"], ["18,000", "25,000"]),
    ("In which languages should the LLM generate irrigation advisories for farmers?", ["KJS-AGR-01"], ["Kannada"], ["Kannada", "Marathi", "Hindi"]),
    ("What does STEPS stand for in the irrigation advisory use case?", ["KJS-AGR-01"], ["Efficient Plot Survey"], ["Efficient Plot Survey"]),
    ("What are the IMD heatwave criteria for coastal regions?", ["KJS-CES-01"], ["coastal regions"], ["4.5", "37"]),
    ("In what file format is the IMD maximum temperature data provided for the heatwave project?", ["KJS-CES-01"], ["GRD files"], ["GRD"]),
    ("Into how many regions and seasons is the temperature data segmented for heatwave analysis?", ["KJS-CES-01"], ["seven IMD-defined regions"], ["seven|7", "four|4"]),
    ("Why are IoT-enabled Automated Weather Stations used in the heatwave system?", ["KJS-CES-01"], ["Automated Weather Stations"], ["local", "validat"]),
    ("What Reynolds number range is used in the CFD simulations for battery cooling?", ["KJS-CES-02"], ["400-700"], ["400", "700"]),
    ("What type of lithium-ion cells are used in the battery pack and how many?", ["KJS-CES-02"], ["21700"], ["21700", "ten|10"]),
    ("Which nanofluid is used as the coolant in the battery thermal management system?", ["KJS-CES-02"], ["Al2O3/water"], ["Al2O3", "water"]),
    ("Which functional safety standard should the BTMS comply with?", ["KJS-CES-02"], ["ISO 26262"], ["26262"]),
    ("How long does the urine strip reader wait before capturing the image?", ["KJS-HLC-01"], ["120-second"], ["120"]),
    ("What hardware is used to build the urine test strip reader?", ["KJS-HLC-01"], ["Raspberry Pi 5"], ["Raspberry Pi", "LED", "camera"]),
    ("Which analytes does the urine strip reader measure?", ["KJS-HLC-01"], ["bilirubin"], ["glucose", "protein", "bilirubin", "ketone"]),
    ("Under which regulation is the urine strip reader classified as a medical device?", ["KJS-HLC-01"], ["CDSCO"], ["CDSCO", "IVD|In Vitro"]),
    ("What communication modes does SomaiyaSat support?", ["KJS-SRS-01"], ["M17"], ["M17", "Codec2", "SSTV", "TT&C|telemetry"]),
    ("What is the role of SomaiyaPod in the mission?", ["KJS-SRS-01"], ["SomaiyaPod"], ["deploy"]),
    ("Which tools are used for mission analysis in the PocketQube mission?", ["KJS-SRS-01"], ["GMAT"], ["STK", "GMAT"]),
    ("What environmental qualification tests are done on SomaiyaSat before launch?", ["KJS-SRS-01"], ["Thermal Vacuum"], ["EMI", "vibration", "vacuum"]),
    # cross-document
    ("Which use cases mention compliance with the DPDP Act?", ["KJS-AGR-01", "KJS-HLC-01"], ["DPDP"], ["AGR|irrigation|sugarcane", "HLC|HC|urine"]),
    ("Which use cases rate Generative AI / LLMs as high relevance?", ["KJS-AGR-01", "KJS-CES-01"], ["Generative AI / LLMs (Relevance: High)"], ["AGR|irrigation|sugarcane", "CES-01|heatwave"]),
    # out of scope
    ("What is the total budget sanctioned for the heatwave project?", [], [], []),
    ("What accuracy did the urine strip regression model achieve on the test set?", [], [], []),
    ("Who won the FIFA World Cup in 2022?", [], [], []),
]
REFUSAL_MARKERS = ["couldn't find", "could not find", "not mentioned", "not available in", "does not contain",
                   "not provided", "no information"]
METRICS = ["Hit@1", "Hit@3", "Hit@5", "MRR@5"]


def is_relevant(doc, expected_ucs, evidence):
    return (doc.metadata["use_case_id"] in expected_ucs
            and any(normalize(e) in normalize(doc.page_content) for e in evidence))


def retrieval_eval(retrieve_fn):
    """retrieve_fn(question, k) -> [(doc, score)]"""
    rows = []
    for q, ucs, ev, _ in TEST_SET:
        if not ucs:
            continue
        hits = retrieve_fn(q, 5)
        rel = [is_relevant(d, ucs, ev) for d, _ in hits]
        first = next((i + 1 for i, r in enumerate(rel) if r), None)
        found = {d.metadata["use_case_id"] for d, _ in hits}
        rows.append({"question": q, "first_relevant_rank": first, "MRR@5": 1 / first if first else 0.0,
                     "Hit@1": int(any(rel[:1])), "Hit@3": int(any(rel[:3])), "Hit@5": int(any(rel[:5])),
                     "uc_coverage@5": len(found & set(ucs)) / len(ucs)})
    return pd.DataFrame(rows)


def temp_store(docs, embeddings):
    """In-memory Chroma collection for comparisons (not saved to disk)."""
    return Chroma.from_documents(docs, embeddings, collection_name=f"tmp-{uuid.uuid4().hex[:8]}",
                                 collection_metadata={"hnsw:space": "cosine"})


def dense_fn(vs):
    return lambda q, k: [(d, 1 - s) for d, s in vs.similarity_search_with_score(q, k=k)]


def keyword_recall(answer, keywords):
    a = normalize(answer).replace(" ", "")
    hits = [any(normalize(alt).replace(" ", "") in a for alt in kw.split("|")) for kw in keywords]
    return sum(hits) / len(hits) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--compare-embeddings", nargs="*", default=None, metavar="MODEL")
    args = ap.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR

    rag = RAGPipeline()
    n_q = sum(1 for t in TEST_SET if t[1])
    print(f"{len(TEST_SET)} test questions ({n_q} answerable) | {len(rag.docs)} chunks in Chroma\n")

    # 1. retrieval mode comparison
    per_q = {m: retrieval_eval(lambda q, k, m=m: rag.retrieve(q, k, mode=m)) for m in ["dense", "bm25", "hybrid"]}
    by_mode = pd.DataFrame({m: df[METRICS + ["uc_coverage@5"]].mean() for m, df in per_q.items()}).T.round(3)
    by_mode.to_csv(out / "retrieval_by_mode.csv")
    pd.concat({m: df for m, df in per_q.items()}).to_csv(out / "retrieval_per_question.csv")
    print("== Retrieval by mode ==\n", by_mode, "\n")
    miss = per_q["hybrid"][per_q["hybrid"]["Hit@1"] == 0][["question", "first_relevant_rank"]]
    print("Hybrid misses at rank 1:\n", miss.to_string(index=False), "\n")

    ax = by_mode[METRICS].plot.bar(figsize=(8, 3.5), rot=0, width=0.75,
                                   color=["#9ecae1", "#4292c6", "#08519c", "#f28e2b"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.set_title(f"Retrieval quality by mode ({n_q} questions)")
    ax.legend(ncol=4, loc="lower center", fontsize=8); plt.tight_layout()
    plt.savefig(out / "retrieval_by_mode.png", dpi=150); plt.close()

    # 2. section-aware vs naive chunking (dense retrieval, same embedding model)
    naive_docs = build_naive_chunks()
    naive_res = retrieval_eval(dense_fn(temp_store(naive_docs, config.get_embeddings())))
    chunk_cmp = pd.DataFrame({
        f"naive fixed-size ({len(naive_docs)} chunks)": naive_res[METRICS].mean(),
        f"section-aware ({len(rag.docs)} chunks)": per_q["dense"][METRICS].mean()}).T.round(3)
    chunk_cmp.to_csv(out / "chunking_comparison.csv")
    print("== Chunking comparison (dense) ==\n", chunk_cmp, "\n")

    # 3. optional embedding model comparison (each model must be pulled in Ollama first)
    if args.compare_embeddings:
        chunks = build_chunks(load_corpus())
        rows = []
        for name in args.compare_embeddings:
            emb = config.get_embeddings(name)
            t0 = time.time()
            vs = temp_store(chunks, emb)
            row = retrieval_eval(dense_fn(vs))[METRICS].mean().to_dict()
            row.update({"model": name, "dim": len(emb.embed_query("x")), "index_time_s": round(time.time() - t0, 1)})
            rows.append(row)
        emb_df = pd.DataFrame(rows).set_index("model").round(3)
        emb_df.to_csv(out / "embedding_comparison.csv")
        print("== Embedding models (dense) ==\n", emb_df, "\n")

    if args.skip_llm:
        return

    # 4. end-to-end answers
    print(f"== Generating answers with {config.LLM_MODEL} (this takes a while on CPU) ==")
    rows = []
    for i, (q, ucs, _, kws) in enumerate(TEST_SET, 1):
        r = rag.ask(q)
        refused = any(m in r["answer"].lower() for m in REFUSAL_MARKERS)
        qtype = "out-of-scope" if not ucs else ("cross-doc" if len(ucs) > 1 else "single")
        rows.append({"type": qtype, "question": q, "keyword_recall": keyword_recall(r["answer"], kws),
                     "refused": refused, "cited": bool(r["cited"]), "latency_s": round(r["latency"], 1),
                     "answer": r["answer"]})
        print(f"  [{i}/{len(TEST_SET)}] {r['latency']:.1f}s  {q[:70]}")
    e2e = pd.DataFrame(rows)
    e2e.to_csv(out / "answers.csv", index=False)

    ins, oos = e2e[e2e.type != "out-of-scope"], e2e[e2e.type == "out-of-scope"]
    summary = (f"LLM: {config.LLM_MODEL} | embeddings: {config.EMBED_MODEL} | mode: {config.RETRIEVAL_MODE}, k={config.TOP_K}\n"
               f"Keyword recall (answerable, n={len(ins)}): {ins.keyword_recall.mean():.3f}\n"
               f"Wrong refusals on answerable questions: {int(ins.refused.sum())} / {len(ins)}\n"
               f"Correct refusals on out-of-scope questions: {int(oos.refused.sum())} / {len(oos)}\n"
               f"Answers with at least one citation: {ins.cited.mean():.0%}\n"
               f"Mean LLM latency: {e2e.latency_s.mean():.1f}s")
    (out / "summary.txt").write_text(summary + "\n\n" + by_mode.to_string() + "\n\n" + chunk_cmp.to_string(),
                                     encoding="utf-8")
    print("\n== Answer quality ==\n" + summary)
    print(f"\nAll results saved in {out}")


if __name__ == "__main__":
    main()
