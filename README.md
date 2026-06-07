# Multi-Document RAG Chatbot

A Streamlit chatbot that lets users upload multiple documents and ask natural-language questions over their contents. It combines vector search, keyword search, Hugging Face embeddings, and source citations.

## Features

- Upload multiple PDFs, DOCX files, PPT/PPTX files, text files, spreadsheets, HTML, and other document-like files
- Extract text from all uploaded documents
- Split large documents into overlapping chunks
- Use Hugging Face tokenization for token-aware chunking and context limits
- Generate embeddings with Hugging Face sentence-transformer models
- Store vectors in FAISS or ChromaDB
- Run hybrid retrieval with vector similarity plus BM25 keyword search
- Rerank retrieved chunks with a cross-encoder for better evidence selection
- Generate answers with a Hugging Face hosted LLM
- Show citations with source file, page, slide, and chunk references
- Fall back to retrieved cited passages if hosted LLM generation is unavailable

## Architecture

```text
User Uploads Files
        |
        v
Text Extraction
        |
        v
Chunking
        |
        v
Embeddings
        |
        v
FAISS / ChromaDB + BM25
        |
        v
Question
        |
        v
Hybrid Search
        |
        v
Cross-Encoder Reranking
        |
        v
Relevant Chunks
        |
        v
Hugging Face LLM
        |
        v
Answer + Citations
```

## Tech Stack

| Component | Technology |
| --- | --- |
| Frontend | Streamlit |
| RAG framework | LangChain |
| Embeddings | Hugging Face sentence-transformers |
| Tokenization | Hugging Face Transformers tokenizer |
| Vector database | FAISS or ChromaDB |
| Keyword search | BM25 via rank-bm25 |
| Reranking | sentence-transformers CrossEncoder |
| LLM | Hugging Face Inference Providers |
| PDF parsing | LangChain PyPDFLoader / pypdf |
| DOCX parsing | LangChain Docx2txtLoader / docx2txt |
| PPTX parsing | python-pptx |
| PPT parsing | unstructured |
| Generic extraction fallback | unstructured partition auto |

## Setup

Create or activate the virtual environment, then install dependencies:

```powershell
cd C:\Users\Admin\PycharmProjects\PythonProject
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Add a Hugging Face token for generated answers:

```powershell
mkdir .streamlit
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

Use a fresh token in `.streamlit/secrets.toml`:

```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_new_token_here"
```

## Run

```powershell
cd C:\Users\Admin\PycharmProjects\PythonProject
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Open the local Streamlit URL shown in the terminal.

## Recommended Model Settings

Start with:

```text
Embedding model: sentence-transformers/all-mpnet-base-v2
Tokenizer model: sentence-transformers/all-mpnet-base-v2
Reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2
Hugging Face LLM: Qwen/Qwen3-8B
Hugging Face provider: nscale
Vector database: FAISS
Vector vs keyword weight: 0.65
Chunk size in tokens: 700
Chunk overlap in tokens: 100
Max context tokens: 6000
```

If the Hugging Face provider does not support your account or region, keep the same app settings and try:

```text
Hugging Face LLM: meta-llama/Llama-3.1-8B-Instruct
Hugging Face provider: novita
```

Meta Llama models may require accepting the model license on Hugging Face first.

## Notes

- FAISS is usually the simplest choice for local demos.
- ChromaDB is useful when you want a more database-like vector store.
- For best accuracy, keep reranking enabled. The first run downloads the reranker model.
- Token-aware chunking helps prevent oversized context and keeps retrieval consistent.
- PDF, DOCX, and PPTX are examples, not hard limits. Unsupported specialized loaders fall back to `unstructured`.
- Scanned PDFs and image-only files need OCR before the chatbot can answer from them accurately.
- For production, move indexing and chat calls into a FastAPI backend and keep Streamlit or React as the frontend.

## Resume-Ready Enhancements

- Multi-user authentication
- OCR for scanned PDFs
- Voice-based document Q&A
- Conversation memory
- Multi-language support
- Image extraction from documents
- Persistent vector indexes per user
- Agentic RAG workflow with LangGraph
