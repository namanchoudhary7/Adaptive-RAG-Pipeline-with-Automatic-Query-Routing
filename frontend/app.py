"""
Streamlit frontend for the Adaptive RAG API.
Run with: uv run streamlit run frontend/app.py
"""

import time, httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Adaptive RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🔍 Adaptive RAG")
    st.caption("Self-correcting retrieval pipeline")
    st.divider()

    st.subheader("System status")
    try:
        health = httpx.get(url=f"{API_BASE}/health", timeout=5).json()
        st.success("API online")
        st.metric("Chunks Indexed", health.get("docs_indexed", '-'))
        st.caption(f"Embedding: `{health.get('embedding_model', '—')}`")
        api_ready = True
    except Exception:
        st.error("API offline — start the backend first")
        st.code("uv run uvicorn backend.main:app --reload")
        api_ready = False

    st.divider()

    st.subheader("Retrieval strategies")
    st.markdown("""
**Semantic** — embedding similarity search. Best for conceptual questions.

**BM25** — keyword frequency search. Best for specific class/function names.

**Hybrid** — both strategies fused. Best for mixed queries.
""")
    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · FastAPI")

# --- Main UI ---
st.title("Ask the docs")
st.caption("Querying FastAPI documentation with adaptive retrieval + self-correction")

EXAMPLES = [
    "How does dependency injection work in FastAPI?",
    "What parameters does HTTPException accept?",
    "How do I use OAuth2PasswordBearer to secure a route?",
    "How do I add CORS middleware to a FastAPI app?",
    "What is the difference between async and sync route handlers?",
]

st.subheader("Example queries")
cols = st.columns(len(EXAMPLES))

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.user_query = example

# Chat Input
query = st.chat_input("Ask anything about the indexed documents...")

if st.session_state.user_query:
    query = st.session_state.user_query
    st.session_state.user_query = "" # Clear after use

if query and api_ready:
    # 1. Show user message
    with st.chat_message("user"):
        st.write(query)

    # 2. Show loading status with updates
    with st.status("🧠 Analyzing query and searching docs...", expanded=True) as status:
        st.write("Routing intent...")
        start = time.perf_counter()
        
        try:
            resp = httpx.post(
                url=f"{API_BASE}/query",
                json={"query": query},
                timeout=None,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = round((time.perf_counter() - start) * 1000)
            
            st.write(f"Routed to **{data['retrieval_strategy'].upper()}** strategy.")
            if data["retry_count"] > 0:
                st.write(f"🔄 Context insufficient. Rewrote query {data['retry_count']} time(s).")
            st.write("Generating final response...")
            
            status.update(label=f"Done in {elapsed}ms", state="complete", expanded=False)
            error = None
            
        except Exception as e:
            status.update(label="Error occurred", state="error", expanded=False)
            data = None
            error = f"API Error: {e}"

    if error:
        st.error(error)
        st.stop()

    # 3. Show Assistant response and telemetry
    with st.chat_message("assistant"):
        st.markdown(data["answer"])
        
        st.divider()
        st.caption("Pipeline Telemetry")
        trace_cols = st.columns(4)
        
        strategy = data["retrieval_strategy"]
        strategy_colours = {"semantic": "🟣", "bm25": "🟡", "hybrid": "🟢"}
        trace_cols[0].metric("Strategy", f"{strategy_colours.get(strategy, '⚪')} {strategy.upper()}")
        
        grade = data["relevance_grade"]
        trace_cols[1].metric("Grade", f"{'✅' if grade == 'sufficient' else '⚠️'} {grade.upper()}")
        
        retries = data["retry_count"]
        trace_cols[2].metric("Retries", f"{'🔄 ' if retries > 0 else ''}{retries}")
        
        confidence = data["confidence"]
        confidence_colours = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        trace_cols[3].metric("Confidence", f"{confidence_colours.get(confidence, '⚪')} {confidence.upper()}")

        st.write("") 
        with st.expander(f"📚 View {len(data['sources'])} Retrieved Sources"):
            for i, source in enumerate(data["sources"], 1):
                score = source["relevance_score"]
                score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                
                st.markdown(f"**[{i}] {source['source'].split('/')[-1] or source['source']}**")
                st.caption(f"Relevance: `{score_bar} {score:.2f}`")
                st.info(source["content"])
                st.divider()