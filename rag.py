"""
Step 3: retrieval + generation.

    python rag.py "What Reynolds number range is used in the CFD study?"
    python rag.py                      # interactive mode, type 'exit' to quit
    python rag.py --mode dense -k 3 "..."
"""
import argparse
import re
import time

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rank_bm25 import BM25Okapi

import config

NOT_FOUND = "I couldn't find this in the AI use case documents."

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a question answering assistant for five AI use case documents from K J Somaiya "
     "(sugarcane irrigation advisory, heatwave early warning, EV battery thermal management, "
     "urine test strip reader, PocketQube satellite).\n"
     "Rules:\n"
     "1. Use ONLY the numbered context blocks below. Do not use outside knowledge.\n"
     "2. Cite the blocks you used inline like [1] or [2][4], right after the sentence they support.\n"
     f"3. If the context does not contain the answer, reply exactly: \"{NOT_FOUND}\"\n"
     "4. When the question is not about a single use case, say which use case (ID) each fact comes from.\n"
     "5. Be concise. Use bullet points for lists. Keep numbers and units exactly as written.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's last question as a standalone question using the chat history, so it "
               "can be understood without the history. Keep names of use cases, systems and technical terms. "
               "Return only the rewritten question, nothing else."),
    ("human", "Chat history:\n{history}\n\n{scope}Last question: {question}"),
])


def normalize(text):
    text = text.lower().replace("₂", "2").replace("₃", "3").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize(text))


def strip_think(text):
    # some Ollama models (qwen3, deepseek-r1) print their reasoning in <think> tags
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class RAGPipeline:
    def __init__(self, llm_model=config.LLM_MODEL, embed_model=config.EMBED_MODEL, persist_dir=config.CHROMA_DIR):
        if not persist_dir.exists():
            raise SystemExit("chroma_db not found. Run `python ingestion.py` first.")
        self.vs = Chroma(collection_name=config.COLLECTION, persist_directory=str(persist_dir),
                         embedding_function=config.get_embeddings(embed_model))
        data = self.vs.get(include=["documents", "metadatas"])
        if not data["ids"]:
            raise SystemExit("The Chroma collection is empty. Run `python ingestion.py` again.")
        docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
        self.docs = {d.metadata["chunk_id"]: d for d in docs}
        self.bm25_ids = list(self.docs)
        self.bm25 = BM25Okapi([tokenize(self.docs[i].page_content) for i in self.bm25_ids])

        self.llm = ChatOllama(model=llm_model, temperature=config.LLM_TEMPERATURE,
                              num_ctx=config.LLM_NUM_CTX, base_url=config.OLLAMA_URL)
        self.qa_chain = QA_PROMPT | self.llm | StrOutputParser()
        self.rewrite_chain = REWRITE_PROMPT | self.llm | StrOutputParser()

    # ------------------------------------------------------------ retrieval
    def dense(self, query, k=5, use_case=None):
        flt = {"use_case_id": use_case} if use_case else None
        hits = self.vs.similarity_search_with_score(query, k=k, filter=flt)
        return [(d, 1 - float(dist)) for d, dist in hits]          # cosine distance -> similarity

    def keyword(self, query, k=5, use_case=None):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = [(self.docs[self.bm25_ids[i]], float(scores[i])) for i in np.argsort(scores)[::-1]]
        if use_case:
            ranked = [(d, s) for d, s in ranked if d.metadata["use_case_id"] == use_case]
        return ranked[:k]

    def hybrid(self, query, k=5, use_case=None, pool=20, rrf_k=60):
        """Reciprocal Rank Fusion of dense and BM25 results: score = sum of 1 / (60 + rank)."""
        fused = {}
        for ranked in (self.dense(query, pool, use_case), self.keyword(query, pool, use_case)):
            for rank, (d, _) in enumerate(ranked, start=1):
                cid = d.metadata["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1 / (rrf_k + rank)
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self.docs[cid], score) for cid, score in best]

    def retrieve(self, query, k=config.TOP_K, use_case=None, mode=config.RETRIEVAL_MODE):
        return {"dense": self.dense, "bm25": self.keyword, "hybrid": self.hybrid}[mode](query, k, use_case)

    # ------------------------------------------------------------ generation
    @staticmethod
    def format_context(hits):
        return "\n\n".join(
            f"[{i}] (source: {d.metadata['use_case_id']}, section: {d.metadata['section']}, "
            f"page {d.metadata['pages']})\n{d.page_content}"
            for i, (d, _) in enumerate(hits, 1))

    def condense(self, question, history, use_case=None):
        if not history:
            return question
        hist = "\n".join(f"{role}: {text}" for role, text in history[-6:])
        scope = ""
        if use_case:  # the filter wins over whatever topic the earlier chat was about
            title = next(d.metadata["title"] for d in self.docs.values() if d.metadata["use_case_id"] == use_case)
            scope = (f"IMPORTANT: the search is restricted to use case {use_case} ({title}). "
                     f"The rewritten question must be about this use case only.\n\n")
        out = self.rewrite_chain.invoke({"history": hist, "scope": scope, "question": question})
        return strip_think(out) or question

    def prepare(self, question, k=config.TOP_K, use_case=None, mode=config.RETRIEVAL_MODE, history=None):
        standalone = self.condense(question, history, use_case)
        hits = self.retrieve(standalone, k, use_case, mode)
        return standalone, hits, {"context": self.format_context(hits), "question": standalone}

    @staticmethod
    def cited(answer, n_hits):
        return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= n_hits})

    def ask(self, question, k=config.TOP_K, use_case=None, mode=config.RETRIEVAL_MODE, history=None):
        standalone, hits, inputs = self.prepare(question, k, use_case, mode, history)
        t0 = time.time()
        answer = strip_think(self.qa_chain.invoke(inputs))
        return {"question": question, "standalone": standalone, "answer": answer, "hits": hits,
                "cited": self.cited(answer, len(hits)), "latency": time.time() - t0}

    def stream(self, inputs):
        """Yields answer tokens, used by the Streamlit app so text appears as it is generated."""
        yield from self.qa_chain.stream(inputs)


def print_result(r):
    if r["standalone"] != r["question"]:
        print(f"(rewritten as: {r['standalone']})")
    print("\n" + r["answer"] + "\n\nSources:")
    for i, (d, s) in enumerate(r["hits"], 1):
        m = d.metadata
        mark = "*" if i in r["cited"] else " "
        print(f" {mark}[{i}] {m['use_case_id']} | {m['section']} | page {m['pages']} | score {s:.3f}")
    print(f"(LLM time {r['latency']:.1f}s, * = cited)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--use-case", default=None, help="e.g. KJS-CES-02")
    args = ap.parse_args()

    rag = RAGPipeline()
    print(f"Loaded {len(rag.docs)} chunks | LLM: {config.LLM_MODEL} | embeddings: {config.EMBED_MODEL}")
    if args.question:
        print_result(rag.ask(" ".join(args.question), args.k, args.use_case, args.mode))
        return
    history = []
    while True:
        q = input("\nQuestion (exit to quit): ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        r = rag.ask(q, args.k, args.use_case, args.mode, history)
        print_result(r)
        history += [("user", q), ("assistant", r["answer"])]


if __name__ == "__main__":
    main()
