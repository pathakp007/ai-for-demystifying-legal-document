
# ⚖️ Legal Demystifier (Streamlit)

A simple, local-first tool to **demystify legal documents** using OpenAI:
- Plain-language **summary**
- **Clause breakdown** (payment, confidentiality, IP, termination, etc.)
- **Risk flags** (unusual or one-sided terms)
- **Q&A** over the doc using RAG (FAISS + OpenAI embeddings)

> **Note**: This tool helps understanding, but is **not legal advice**.

---

## ✅ Quick Start

1. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set your OpenAI API key**

Create a `.env` or set environment variable:
```bash
# Windows (PowerShell)
setx OPENAI_API_KEY "sk-..."
# macOS/Linux
export OPENAI_API_KEY="sk-..."
```

(Optional) choose models:
```bash
export OPENAI_CHAT_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-small"
```

4. **Run the app**

```bash
streamlit run app.py
```

Open your browser (the URL is printed in the terminal) and upload a PDF/DOCX/TXT.

---

## 💡 Features

- **Summarize**: plain English TL;DR focused on obligations, fees, confidentiality, IP, termination.
- **Extract Clauses**: structured list of key terms.
- **Risk Analysis**: flags unusual/non-standard clauses to review.
- **Ask Questions**: answers grounded in the document via FAISS + embeddings.

---

## 🔧 Implementation Notes

- **Chunking**: simple word-based chunks with overlap for context preservation.
- **Embeddings**: `text-embedding-3-small` (good quality & cost).
- **Chat**: defaults to `gpt-4o-mini` (fast+capable). You can switch models via env vars.
- **RAG**: cosine similarity via FAISS (inner product on normalized vectors).
- **Security**: files are processed in-memory; nothing is stored server-side by default.

---

## ⚠️ Disclaimer

This tool does not provide legal advice. Always have a qualified attorney review critical documents and decisions.
