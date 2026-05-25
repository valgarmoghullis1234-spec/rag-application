"""
Streamlit UI — Upload documents and chat with them

Level 1 upgrades:
  - Conversation Memory: full chat history sent to the backend each turn
  - Multi-Document Querying: badge shown when answer draws from multiple docs
  - Re-ranking: rerank_score surfaced in the Sources expander

Level 2 upgrades:
  - Authentication: login screen protects the app before showing anything
  - Streaming: tokens stream word-by-word from the /query/stream endpoint
  - Guardrails: off-topic questions are flagged in the UI
"""

import json
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_URL      = "http://localhost:8000"
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


# ── Main app (only shown after login) ───────────────────────────────────────

st.title("📚 RAG Document Chatbot")
st.caption("Upload your documents and ask questions — answers come from your files.")


# ── Sidebar: document management ────────────────────────────────────────────

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


# ── Main: chat interface ─────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("off_topic"):
                st.caption("⚠️ Off-topic — question not related to uploaded documents")
            elif msg.get("multi_doc"):
                st.caption("Multi-document answer")
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for s in msg["sources"]:
                        rerank = f" | rerank {s['rerank_score']}" if s.get("rerank_score") else ""
                        st.write(
                            f"- **{s['source']}** | {s.get('section', '')} | "
                            f"chunk {s['chunk_index']} | similarity {s['similarity']}"
                            f"{rerank} | {s.get('match_type', 'vector')}"
                        )


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
            sources    = []
            multi_doc  = False
            off_topic  = False

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
                st.caption("Multi-document answer")

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
            "role"     : "assistant",
            "content"  : full_text,
            "sources"  : sources,
            "multi_doc": multi_doc,
            "off_topic": off_topic,
        })
