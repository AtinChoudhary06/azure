import os
import streamlit as st
import requests

# Point this at your deployed FastAPI URL in production via an env var,
# so you don't have to edit code when you redeploy.
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")

st.title("📄 PDF RAG Chatbot")
st.caption("Upload a PDF, then ask questions about it.")

with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading and processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=180)
                    if response.status_code == 200:
                        data = response.json()
                        msg = f"Indexed {data['chunks_indexed']} chunks from {data['filename']}"
                        if data.get("old_chunks_replaced"):
                            msg += f" (replaced {data['old_chunks_replaced']} old chunks)"
                        st.success(msg)
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")

    st.divider()
    st.caption("Backend status:")
    try:
        health = requests.get(f"{API_BASE_URL}/", timeout=5)
        if health.status_code == 200:
            st.success("API is running")
        else:
            st.warning("API responded but with an error")
    except requests.exceptions.RequestException:
        st.error("Cannot reach API — is it running?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question},
                    timeout=60,
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = f"Error: {response.text}"
            except requests.exceptions.RequestException as e:
                answer = f"Error connecting to backend: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})