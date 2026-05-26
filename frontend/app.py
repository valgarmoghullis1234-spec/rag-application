"""
Streamlit UI — Upload documents and chat with them

Features:
  - Authentication: login screen protects the app
  - Streaming: tokens stream word-by-word
  - Conversation Memory: full chat history sent each turn
  - Multi-Document Synthesis: badge when answer spans multiple docs
  - Re-ranking: rerank_score surfaced in Sources expander
  - Guardrails: off-topic questions flagged
  - Agent Mode: Claude drives multi-step retrieval with live tool-call trace
"""

import json
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_URL      = os.getenv("BACKEND_URL", "http://localhost:8000")
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "changeme123")

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")


# ── Authentication ───────────────────────────────────────────────────────────

def show_login():
    st.title("📚 RAG Document Chatbot")
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            if username == APP_USERNAME and password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")


if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()


# ── Main app ─────────────────────────────────────────────────────────────────

st.title("📚 RAG Document Chatbot")
st.caption("Upload your documents and ask questions — answers come from your files.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="Supported: PDF, Word (.docx), plain text",
    )

    if uploaded_file and st.button("Ingest Document"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            )
        if resp.ok:
            data = resp.json()
            st.success(f"Ingested **{data['filename']}** — {data['chunks']} chunks created.")
            st.rerun()
        else:
            st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")

    st.divider()

    st.subheader("Uploaded Documents")
    try:
        docs_resp = requests.get(f"{API_URL}/documents")
        documents = docs_resp.json().get("documents", []) if docs_resp.ok else []
    except Exception:
        documents = []
        st.warning("Could not reach the API server.")

    if not documents:
        st.info("No documents uploaded yet.")
    else:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                resp = requests.delete(f"{API_URL}/documents/{doc}")
                if resp.ok:
                    st.success(f"Deleted {doc}")
                    st.rerun()
                else:
                    st.error("Delete failed.")

    st.divider()

    source_filter = None
    if documents:
        filter_choice = st.selectbox(
            "Search within (optional)",
            options=["All documents"] + documents,
        )
        if filter_choice != "All documents":
            source_filter = filter_choice

    st.divider()

    # ── Query mode toggle ────────────────────────────────────────────────────
    st.subheader("Query Mode")
    use_agent = st.toggle(
        "🤖 Agent mode",
        value=False,
        help="Claude makes multiple targeted searches — better for complex questions",
    )
    if use_agent:
        st.caption(
            "✅ Best for: comparisons across docs, multi-hop questions, "
            "full-document summaries, self-correcting when first search is weak."
        )
    else:
        st.caption("⚡ Fast single-pass retrieval — best for direct factual questions.")

    st.divider()

    st.subheader("Conversation")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    msg_count = len(st.session_state.get("messages", []))
    if msg_count:
        st.caption(f"{msg_count} message{'s' if msg_count != 1 else ''} in memory")

    st.divider()

    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()


# ── Helper: render one assistant message ─────────────────────────────────────

def render_assistant_message(msg: dict):
    """Render a stored assistant message (from chat history replay)."""
    st.markdown(msg["content"])

    # Mode badge
    if msg.get("agent_mode"):
        iters = msg.get("iterations", "?")
        tc    = len(msg.get("tool_calls", []))
        st.caption(f"🤖 Agent — {tc} tool call{'s' if tc != 1 else ''}, {iters} iteration{'s' if iters != 1 else ''}")
    elif msg.get("off_topic"):
        st.caption("⚠️ Off-topic — question not related to uploaded documents")
    elif msg.get("multi_doc"):
        st.caption("📑 Multi-document answer")

    # Agent tool trace
    if msg.get("agent_mode") and msg.get("tool_calls"):
        with st.expander("🔍 Agent reasoning trace"):
            for i, tc in enumerate(msg["tool_calls"], 1):
                tool  = tc["tool"]
                inp   = tc["input"]
                summ  = tc.get("summary", "")
                if tool == "search":
                    q      = inp.get("query", "")
                    src    = f" in `{inp['source']}`" if inp.get("source") else ""
                    label  = f"**Search:** \"{q}\"{src}"
                elif tool == "list_documents":
                    label  = "**List documents**"
                elif tool == "get_section":
                    label  = f"**Get section:** `{inp.get('section', '')}` from `{inp.get('source', '')}`"
                else:
                    label  = f"**{tool}**"
                st.write(f"{i}. {label}")
                if summ:
                    st.caption(f"   → {summ}")

    # Sources (normal RAG)
    if not msg.get("agent_mode") and msg.get("sources"):
        with st.expander("Sources used"):
            for s in msg["sources"]:
                rerank = f" | rerank {s['rerank_score']}" if s.get("rerank_score") else ""
                st.write(
                    f"- **{s['source']}** | {s.get('section', '')} | "
                    f"chunk {s['chunk_index']} | similarity {s['similarity']}"
                    f"{rerank} | {s.get('match_type', 'vector')}"
                )


# ── Chat history replay ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg)
        else:
            st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about your documents..."):
    if not documents:
        st.warning("Please upload at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_text  = ""

            # ── AGENT MODE ────────────────────────────────────────────────────
            if use_agent:
                tool_calls_log = []
                iterations     = 1
                agent_sources  = []

                with st.status("🤖 Agent is researching...", expanded=True) as status:
                    try:
                        with requests.post(
                            f"{API_URL}/query/agent/stream",
                            json={
                                "question"     : question,
                                "source_filter": source_filter,
                                "history"      : history_payload,
                            },
                            stream=True,
                            timeout=120,
                        ) as resp:
                            if resp.ok:
                                for raw_line in resp.iter_lines():
                                    if not raw_line:
                                        continue
                                    data = json.loads(raw_line)

                                    if data["type"] == "agent_start":
                                        st.write("Starting research...")

                                    elif data["type"] == "tool_call":
                                        tool = data["tool"]
                                        inp  = data["input"]
                                        itr  = data.get("iteration", "")
                                        if tool == "search":
                                            q   = inp.get("query", "")
                                            src = f" in `{inp['source']}`" if inp.get("source") else ""
                                            st.write(f"🔍 Searching: *\"{q}\"*{src}")
                                        elif tool == "list_documents":
                                            st.write("📋 Listing available documents...")
                                        elif tool == "get_section":
                                            st.write(
                                                f"📖 Reading section: *{inp.get('section', '')}* "
                                                f"from `{inp.get('source', '')}`"
                                            )

                                    elif data["type"] == "tool_result":
                                        st.caption(f"   → {data.get('summary', '')}")

                                    elif data["type"] == "token":
                                        full_text += data["text"]
                                        response_placeholder.markdown(full_text + "▌")

                                    elif data["type"] == "done":
                                        tool_calls_log = data.get("tool_calls", [])
                                        iterations     = data.get("iterations", 1)
                                        agent_sources  = data.get("sources", [])
                                        tc_count = len(tool_calls_log)
                                        status.update(
                                            label=(
                                                f"✅ Done — {tc_count} tool call{'s' if tc_count != 1 else ''}, "
                                                f"{iterations} iteration{'s' if iterations != 1 else ''}"
                                            ),
                                            state="complete",
                                            expanded=False,
                                        )
                            else:
                                full_text = f"Error: {resp.json().get('detail', 'Unknown error')}"
                                status.update(label="❌ Error", state="error")

                    except Exception as e:
                        full_text = f"Could not reach the API server: {e}"
                        status.update(label="❌ Error", state="error")

                response_placeholder.markdown(full_text)

                # Agent badge + tool trace
                tc_count = len(tool_calls_log)
                st.caption(
                    f"🤖 Agent — {tc_count} tool call{'s' if tc_count != 1 else ''}, "
                    f"{iterations} iteration{'s' if iterations != 1 else ''}"
                )
                if tool_calls_log:
                    with st.expander("🔍 Agent reasoning trace"):
                        for i, tc in enumerate(tool_calls_log, 1):
                            tool  = tc["tool"]
                            inp   = tc["input"]
                            summ  = tc.get("summary", "")
                            if tool == "search":
                                q     = inp.get("query", "")
                                src   = f" in `{inp['source']}`" if inp.get("source") else ""
                                label = f"**Search:** \"{q}\"{src}"
                            elif tool == "list_documents":
                                label = "**List documents**"
                            elif tool == "get_section":
                                label = f"**Get section:** `{inp.get('section', '')}` from `{inp.get('source', '')}`"
                            else:
                                label = f"**{tool}**"
                            st.write(f"{i}. {label}")
                            if summ:
                                st.caption(f"   → {summ}")

                st.session_state.messages.append({
                    "role"      : "assistant",
                    "content"   : full_text,
                    "agent_mode": True,
                    "tool_calls": tool_calls_log,
                    "iterations": iterations,
                    "sources"   : agent_sources,
                    "off_topic" : False,
                    "multi_doc" : False,
                })

            # ── NORMAL RAG MODE ───────────────────────────────────────────────
            else:
                sources   = []
                multi_doc = False
                off_topic = False

                try:
                    with requests.post(
                        f"{API_URL}/query/stream",
                        json={
                            "question"     : question,
                            "source_filter": source_filter,
                            "history"      : history_payload,
                        },
                        stream=True,
                        timeout=60,
                    ) as resp:
                        if resp.ok:
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                data = json.loads(raw_line)
                                if data["type"] == "metadata":
                                    sources   = data["sources"]
                                    multi_doc = data["multi_doc"]
                                    off_topic = data.get("off_topic", False)
                                elif data["type"] == "token":
                                    full_text += data["text"]
                                    response_placeholder.markdown(full_text + "▌")
                        else:
                            full_text = f"Error: {resp.json().get('detail', 'Unknown error')}"

                except Exception as e:
                    full_text = f"Could not reach the API server: {e}"

                response_placeholder.markdown(full_text)

                if off_topic:
                    st.caption("⚠️ Off-topic — question not related to uploaded documents")
                elif multi_doc:
                    st.caption("📑 Multi-document answer")

                if sources:
                    with st.expander("Sources used"):
                        for s in sources:
                            rerank = f" | rerank {s['rerank_score']}" if s.get("rerank_score") else ""
                            st.write(
                                f"- **{s['source']}** | {s.get('section', '')} | "
                                f"chunk {s['chunk_index']} | similarity {s['similarity']}"
                                f"{rerank} | {s.get('match_type', 'vector')}"
                            )

                st.session_state.messages.append({
                    "role"      : "assistant",
                    "content"   : full_text,
                    "agent_mode": False,
                    "sources"   : sources,
                    "multi_doc" : multi_doc,
                    "off_topic" : off_topic,
                })
