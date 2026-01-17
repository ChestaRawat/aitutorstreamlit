import streamlit as st
from backend import process_pdf, build_faiss, ask_question
import hashlib
import time

# ---------------- Utility ----------------
def get_pdf_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Tutor")
st.subheader("Ask questions from your uploaded PDF")

# ---------------- Session State ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.pdf_hash = None
    st.session_state.answer = None
    st.session_state.last_ask_time = 0

# ---------------- PDF Upload ----------------
uploaded_pdf = st.file_uploader(
    "📄 Upload your study PDF",
    type=["pdf"]
)

# 🔥 RESET WHEN PDF IS REMOVED
if uploaded_pdf is None and st.session_state.pdf_hash is not None:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.pdf_hash = None
    st.session_state.answer = None
    st.session_state.last_ask_time = 0

# ---------------- PDF Processing ----------------
if uploaded_pdf is not None:
    current_hash = get_pdf_hash(uploaded_pdf)

    if st.session_state.pdf_hash != current_hash:
        # ✅ Clear old data completely
        st.session_state.chunks = None
        st.session_state.index = None
        st.session_state.answer = None

        with st.spinner("Processing new PDF..."):
            chunks = process_pdf(uploaded_pdf)
            index = build_faiss(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.pdf_hash = current_hash

        st.success("✅ New PDF processed successfully")

# ---------------- Question Input ----------------
st.divider()

question = st.text_input(
    "Ask a question from the PDF:",
    placeholder="e.g. What is photosynthesis?",
    disabled=st.session_state.chunks is None
)

# ---------------- Ask Button ----------------
if st.button("Ask Tutor"):
    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")

    elif question.strip() == "":
        st.warning("Please enter a question.")

    elif time.time() - st.session_state.last_ask_time < 3:
        st.warning("⏳ Please wait 3 seconds before asking another question.")

    else:
        st.session_state.last_ask_time = time.time()

        with st.spinner("Thinking..."):
            st.session_state.answer = ask_question(
                question,
                st.session_state.chunks,
                st.session_state.index
            )

# ---------------- Show Answer ----------------
if st.session_state.answer:
    st.success("📘 Answer")
    st.write(st.session_state.answer)