# FCSIT Academic Advisory Chatbot

A mobile-based academic advisory chatbot for students of FCSIT, UNIMAS. AdvisorBot uses a Retrieval-Augmented Generation (RAG) architecture to answer general academic queries grounded in official FCSIT and UNIMAS documents.

> **Note:** This repository contains the chatbot backend only. The chatbot frontend/mobile app is in this repository: [AdvisorBot Mobile App](https://github.com/cst023/FCSIT-AdvisorBot-Mobile-App)

---
<p float="left">
  <img src="assets/new chat.jpg" width="30%"/>&nbsp;&nbsp;
  <img src="assets/academic query.jpg" width="30%" />&nbsp;&nbsp;
  <img src="assets/query 2.jpg" width="30%"/>
</p>

---

## Features

- **RAG-powered responses** — answers grounded strictly in official academic documents, not general model knowledge
- **Source citations** — every answer includes the source document, page number, and link
- **VLM-assisted document processing** — a Vision Language Model (VLM) extracts content from visually complex PDFs (tables, organisational charts, flowcharts) into structured Markdown for accurate retrieval
- **Intent classification** — a lightweight LLM pre-screens messages into academic queries, greetings, and thank-you messages, avoiding unnecessary retrieval API calls
- **Controlled fallback** — when no relevant context is found, the chatbot explicitly states it cannot answer and advises the student to consult their academic advisor
- **REST API** — FastAPI endpoint for easy integration with the Flutter mobile app

---

## Knowledge Base

The chatbot retrieves from three official UNIMAS/FCSIT documents:

| Document | Coverage |
|---|---|
| FCSIT Student Handbook (Session 2025/2026) | Programme structure, curriculum, faculty info, academic rules, facilities |
| UNIMAS Academic Regulations for Undergraduate Studies | Registration, grading system, credit transfer, academic status, graduation requirements |
| UNIMAS Academic Calendar (Session 2025/2026) | Key dates, semester schedule, public holidays |

---

## System Overview

```
User Query (Flutter App)
        │
        ▼
   FastAPI Backend
        │
        ▼
  Intent Classifier (Gemma 2B)
  ┌─────┴──────┐
  │            │
Academic     Greeting / Thanks / Follow-up
  Query          → Predefined response
  │
  ▼
Embedding Model (nvidia/llama-nemotron-embed-1b-v2)
        │
        ▼
ChromaDB Vector Store (cosine similarity, top-5)
        │
        ▼
Generative Model (nvidia/nemotron-super-120b)
        │
        ▼
  Source-grounded answer with citations
```

### Document Processing Pipeline

```
PDF Documents
     │
     ▼
Convert pages to images (pdf2image)
     │
     ▼
VLM extraction per page (qwen/qwen3.5-397b-a17b)
     │   Produces structured Markdown preserving
     │   tables, charts, and visual layouts
     ▼
Page-level chunking
     │
     ▼
Embedding (nvidia/llama-nemotron-embed-1b-v2)
     │
     ▼
ChromaDB vector store
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend framework | FastAPI, Python |
| RAG orchestration | LangChain |
| Vector database | ChromaDB |
| Embedding model | nvidia/llama-nemotron-embed-1b-v2 |
| Generative model | nvidia/nemotron-super-120b |
| Intent classifier | google/gemma-2-2b-it |
| Document extraction | qwen/qwen3.5-397b-a17b (VLM) |
| Model inference API | NVIDIA NIM |
| Mobile frontend | Flutter (separate repo at [AdvisorBot Mobile App](https://github.com/cst023/FCSIT-AdvisorBot-Mobile-App)) |

---

## Setup and Usage Instructions:

You can try it in two ways:

1. CLI chat mode (run `vector_rag.py` directly)
2. Local API server mode (run FastAPI with Uvicorn) for frontend integration (e.g. Flutter app)

### Prerequisites

- Python 3.10+
- NVIDIA NIM API key — get one at [build.nvidia.com](https://build.nvidia.com/settings/api-keys)

### Installation

```bash
git clone https://github.com/cst023/FCSIT-AdvisorBot-Backend.git
cd FCSIT-AdvisorBot-Backend
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
NVIDIA_NIM_API=your_api_key_here
```

### Option 1: CLI mode

```bash
python vector_rag.py
```

Type your questions in the terminal. Enter `-1` to exit.

### Option 2: Local API server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health` — health check
- `POST /query` — send a query, returns answer + detected intent + response time

---

## Project Structure

```
├── vector_rag.py                  # RAG pipeline (intent classifier, retriever, generator)
├── main.py                        # FastAPI app and API endpoints
├── chunk_and_embed_page_level.py  # Chunking and embedding script
├── extract_pdf_content_with_vlm.ipynb  # VLM-based PDF content extraction notebook
├── requirements.txt
├── Dockerfile
├── guidebook_content.md           # Extracted FCSIT handbook content
├── academic_regulations_undergraduates_content.md
├── academic_calendar_ug_content.md
└── chroma_fcsit/                  # ChromaDB vector store (generated)
```

---

## Notes

- If you are testing with your Flutter frontend, start this backend server first.
- This repository only contains the backend; the Flutter app repository is separate at: [AdvisorBot Mobile App](https://github.com/cst023/FCSIT-AdvisorBot-Mobile-App)
