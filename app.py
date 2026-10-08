"""
Step 4: Streamlit chat UI.

    streamlit run app.py
"""
import streamlit as st

import config
from rag import RAGPipeline, strip_think

st.set_page_config(page_title="AI Use Case Q&A", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading vector store and model...")
def load_pipeline(llm_model):
    return RAGPipeline(llm_model=llm_model)


EXAMPLES = [
    "What is the problem statement of the heatwave use case?",
    "How does the urine strip reader convert pad colours into concentrations?",
    "What are the operational challenges in the PocketQube mission?",
    "Compare the human-in-the-loop requirements across all five use cases.",
]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("Ollama model", config.LLM_MODEL)
    rag = load_pipeline(llm_model)

    titles = {}
    for d in rag.docs.values():
        titles.setdefault(d.metadata["use_case_id"], d.metadata["title"])
    options = ["All use cases"] + sorted(titles)
    use_case = st.selectbox("Use case filter", options,
                            format_func=lambda x: x if x == "All use cases" else f"{x} | {titles[x][:40]}")
    mode = st.radio("Retrieval mode", ["hybrid", "dense", "bm25"], horizontal=True)
    k = st.slider("Top-k chunks", 2, 10, config.TOP_K)
    use_history = st.checkbox("Use chat history for follow-ups", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{len(rag.docs)} chunks from {len(titles)} use case PDFs | embeddings: {config.EMBED_MODEL}")


def show_sources(sources, standalone=None):
    with st.expander(f"Sources ({len(sources)} chunks)"):
        if standalone:
            st.caption(f"Rewritten query: {standalone}")
        for s in sources:
            badge = "✅ cited" if s["cited"] else "not cited"
            st.markdown(f"**[{s['n']}] {s['use_case_id']} | {s['section']} | page {s['pages']}** "
                        f"(score {s['score']:.3f}, {badge})")
            text = s["text"][:600] + ("..." if len(s["text"]) > 600 else "")
            # one paragraph per line, otherwise markdown merges "B. AI Challenges" into the bullet above it
            st.caption("\n\n".join(line for line in text.split("\n") if line.strip()))


# ---------------------------------------------------------------- chat
st.title("AI Use Case Q&A (RAG)")
st.caption("Answers come only from the five KJSSE AI use case documents, with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"], msg.get("standalone"))

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, width="stretch"):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about the AI use cases") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [(m["role"], m["content"]) for m in st.session_state.messages] if use_history else None
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                standalone, hits, inputs = rag.prepare(
                    question, k=k, use_case=None if use_case == "All use cases" else use_case,
                    mode=mode, history=history)
            answer = strip_think(st.write_stream(rag.stream(inputs)))
        except Exception as e:  # usually Ollama not running or model not pulled
            answer, hits, standalone = f"Error: {e}\n\nIs `ollama serve` running and is the model pulled?", [], None
            st.error(answer)
        cited = rag.cited(answer, len(hits))
        sources = [{"n": i, "use_case_id": d.metadata["use_case_id"], "section": d.metadata["section"],
                    "pages": d.metadata["pages"], "score": s, "cited": i in cited,
                    "text": d.page_content.split("\n", 1)[-1]} for i, (d, s) in enumerate(hits, 1)]
        standalone = standalone if standalone != question else None
        if sources:
            show_sources(sources, standalone)
    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "sources": sources, "standalone": standalone})
