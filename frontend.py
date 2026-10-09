import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Page setup
st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 MirAI Student Policy Advisor")
st.markdown(
    "Ask questions regarding academic guidelines, student conduct, attendance, "
    "and institutional policies. Answers are strictly grounded in official handbook documentation."
)

# Sidebar: PDF Handbook Upload & Status
with st.sidebar:
    st.header("Policy Handbook")
    st.markdown("Upload the official student policy handbook (PDF) to build or refresh the advisor's knowledge base.")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF handbook",
        type=["pdf"],
        help="Upload the institutional handbook to index.",
    )

    if uploaded_pdf is not None:
        if st.button("Upload & Index Handbook", use_container_width=True):
            with st.spinner("Uploading and indexing handbook with ChromaDB..."):
                try:
                    files = {
                        "file": (
                            uploaded_pdf.name,
                            uploaded_pdf.getvalue(),
                            "application/pdf",
                        )
                    }
                    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)

                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            st.success(
                                f"✅ **Handbook Indexed Successfully!**\n\n"
                                f"- **File:** {data.get('filename')}\n"
                                f"- **Pages:** {data.get('num_pages')}\n"
                                f"- **Chunks:** {data.get('num_chunks')}"
                            )
                        except ValueError:
                            st.success("✅ Handbook indexed successfully!")
                    else:
                        try:
                            err_detail = resp.json().get("detail", resp.text)
                        except Exception:
                            err_detail = resp.text
                        st.error(f"❌ Ingestion Failed ({resp.status_code}):\n{err_detail}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out while ingesting the handbook. The document may be very large.")
                except requests.exceptions.ConnectionError:
                    st.error(f"🔌 Could not connect to backend at `{BACKEND_URL}`. Ensure FastAPI is running.")
                except Exception as ex:
                    st.error(f"⚠️ An unexpected error occurred: {str(ex)}")

    st.divider()

    # Backend Connection Status Check
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            indexed_count = health_data.get("indexed_chunks", 0)
            st.caption(f"🟢 **Backend Connected** | Indexed Chunks: `{indexed_count}`")
        else:
            st.caption(f"🟠 **Backend Responded with HTTP {health_resp.status_code}**")
    except Exception:
        st.caption(f"🔴 **Backend Offline** (`{BACKEND_URL}`)")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Retrieved Handbook Context", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Handbook")
                    page = meta.get("page")
                    page_str = f" (Page {page + 1})" if isinstance(page, int) else (f" (Page {page})" if page else "")
                    st.markdown(f"**Excerpt {idx} — {source_name}{page_str}**")
                    st.text(src.get("content", "").strip())

# Chat Input & Response Handling
if prompt := st.chat_input("Ask a question about student policies..."):
    # Append user question to session state and render
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend chat endpoint
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook and drafting verified advice..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": prompt},
                    timeout=60,
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        answer = data.get("answer", "No answer returned.")
                        retrieved_context = data.get("retrieved_context", [])

                        st.markdown(answer)

                        if retrieved_context:
                            with st.expander("📚 Retrieved Handbook Context", expanded=False):
                                for idx, src in enumerate(retrieved_context, 1):
                                    meta = src.get("metadata", {})
                                    source_name = meta.get("source", "Handbook")
                                    page = meta.get("page")
                                    page_str = (
                                        f" (Page {page + 1})"
                                        if isinstance(page, int)
                                        else (f" (Page {page})" if page else "")
                                    )
                                    st.markdown(f"**Excerpt {idx} — {source_name}{page_str}**")
                                    st.text(src.get("content", "").strip())

                        # Persist assistant response in session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_context,
                        })

                    except ValueError:
                        err_msg = "Received malformed JSON response from the backend service."
                        st.error(f"⚠️ {err_msg}")
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})

                else:
                    try:
                        err_detail = response.json().get("detail", response.text)
                    except Exception:
                        err_detail = response.text
                    err_msg = f"Backend Error ({response.status_code}): {err_detail}"
                    st.error(f"❌ {err_msg}")
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

            except requests.exceptions.Timeout:
                err_msg = "⏱️ Request timed out while waiting for the policy advisor response."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except requests.exceptions.ConnectionError:
                err_msg = f"🔌 Could not connect to backend at `{BACKEND_URL}`. Make sure FastAPI is running."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as ex:
                err_msg = f"⚠️ An unexpected error occurred: {str(ex)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
