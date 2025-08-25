import os
import io
import re
from typing import List, Tuple

import streamlit as st
import pdfplumber
from docx import Document as DocxDocument
import faiss
import numpy as np
from groq import Groq

# --------------
# Helper functions
# --------------

def read_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def read_docx(file_bytes: bytes) -> str:
    buffer = io.BytesIO(file_bytes)
    doc = DocxDocument(buffer)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY. Please set it in Streamlit Secrets or environment.")
        st.stop()
    return Groq(api_key=api_key)

def llm_chat(messages, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=800):
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# Since Groq does not provide embeddings yet, we simulate embeddings with a simple hashing trick.
def embed_texts(texts: List[str]) -> np.ndarray:
    dim = 512
    vectors = []
    for txt in texts:
        vec = np.zeros(dim, dtype="float32")
        for word in txt.split():
            idx = hash(word) % dim
            vec[idx] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.stack(vectors)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def rag_answer(query: str, chunks: List[str], index, chunk_embeddings: np.ndarray, top_k: int = 4) -> Tuple[str, List[int]]:
    q_emb = embed_texts([query])[0].reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    system = {"role": "system", "content": "You are a careful legal explainer. Use only the provided CONTEXT to answer."}
    user_msg = {"role": "user", "content": "CONTEXT:\n---\n" + "\n\n".join(retrieved) + "\n---\n\nQuestion: " + query}
    answer = llm_chat([system, user_msg], max_tokens=700)
    return answer, I[0].tolist()

# --------------
# Streamlit App
# --------------

st.set_page_config(page_title="Legal Demystifier", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Demystifier (Free Groq Version)")
st.caption("Upload a legal document to get a plain-language summary, clause breakdown, risk flags, and ask questions.")

with st.sidebar:
    st.header("Settings")
    st.write("Set your GROQ_API_KEY in Streamlit Secrets or environment.")
    st.markdown("---")
    do_summary = st.checkbox("Generate Summary", value=True)
    do_clauses = st.checkbox("Extract Key Clauses", value=True)
    do_risks = st.checkbox("Flag Potential Risks", value=True)
    st.markdown("---")
    chunk_size = st.slider("Chunk size (words)", 600, 2000, 1200, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 50, 400, 150, 10)

uploaded = st.file_uploader("Upload a contract/policy (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded is not None:
    ext = uploaded.name.lower().split(".")[-1]
    file_bytes = uploaded.read()
    try:
        if ext == "pdf":
            raw_text = read_pdf(file_bytes)
        elif ext == "docx":
            raw_text = read_docx(file_bytes)
        else:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    cleaned = clean_text(raw_text)
    if len(cleaned) < 50:
        st.warning("That file seems to have very little text. Are you sure it's the right document?")

    chunks = chunk_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.success(f"Document loaded. {len(cleaned):,} characters split into {len(chunks)} chunks.")

    with st.spinner("Embedding document..."):
        try:
            embs = embed_texts(chunks)
            index = build_faiss_index(embs)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

    col1, col2 = st.columns(2)

    if do_summary:
        with col1:
            st.subheader("📌 Plain-Language Summary")
            system = {"role": "system", "content": "You are a legal explainer. Summarize concisely in plain language."}
            user_msg = {"role": "user", "content": "\n\n".join(chunks[:6])}
            try:
                summary = llm_chat([system, user_msg], max_tokens=900)
                st.write(summary)
            except Exception as e:
                st.error(f"Summary failed: {e}")

    if do_clauses:
        with col1:
            st.subheader("📄 Clause Breakdown")
            clause_prompt = {"role": "user", "content": "From the legal text, extract key clauses in plain language." + "\n\n" + "\n\n".join(chunks[:8])}
            try:
                clauses = llm_chat([{"role": "system", "content": "Extract clauses clearly."}, clause_prompt], max_tokens=1000)
                st.write(clauses)
            except Exception as e:
                st.error(f"Clause extraction failed: {e}")

    if do_risks:
        with col1:
            st.subheader("🚩 Potential Risks & Unusual Terms")
            risk_prompt = {"role": "user", "content": "Identify risks and unusual terms in this contract:\n\n" + "\n\n".join(chunks[:10])}
            try:
                risks = llm_chat([{"role": "system", "content": "You are a cautious legal risk analyst."}, risk_prompt], max_tokens=900)
                st.write(risks)
            except Exception as e:
                st.error(f"Risk analysis failed: {e}")

    with col2:
        st.subheader("💬 Ask Questions About This Document")
        st.caption("Backed by RAG with simple hashing embeddings.")
        q = st.text_input("Your question (e.g., 'What happens if I cancel early?')")
        if st.button("Answer") and q.strip():
            with st.spinner("Searching and answering..."):
                try:
                    answer, hits = rag_answer(q.strip(), chunks, index, embs, top_k=4)
                    st.write(answer)
                    with st.expander("Show supporting excerpts"):
                        for rank, idx in enumerate(hits, start=1):
                            if 0 <= idx < len(chunks):
                                st.markdown(f"**Excerpt {rank} (chunk {idx})**")
                                st.write(chunks[idx])
                except Exception as e:
                    st.error(f"Q&A failed: {e}")

    st.markdown("---")
    st.caption("⚠️ Not legal advice. Use results for understanding; consult a qualified lawyer for decisions.")
else:
    st.info("Upload a PDF, DOCX, or TXT to begin.")
