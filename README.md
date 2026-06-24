# ⚖️ LexAgent

> **Trustworthy Multi-Agent Legal & Compliance Intelligence Platform for Indian Enterprises**

LexAgent is a production-grade, multi-agent AI-system designed for reviewing commercial contracts under Indian law, analyzing regulatory compliance, drafting clauses, and fact-checking outputs using a state-of-the-art hybrid retrieval pipeline.

---

## 🏗️ System Architecture

LexAgent uses a centralized **Supervisor-Specialist** design alongside a **hybrid retrieval** backend. Every user query is dynamically routed to specialized legal agents, and all generated insights are strictly verified for factual grounding before presentation.

```
                  ┌────────────────────────┐
                  │       User / API       │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    Supervisor Agent    │
                  └───────────┬────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Contract Review  │ │ Legal Compliance │ │ Drafting Agent   │
│      Agent       │ │      Agent       │ │                  │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Grounding Verifier    │◄─── (Factual Fact-Checking)
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    Final Synthesizer   │
                  └────────────────────────┘
```

---

## 🛠️ Key Components & Pipelines

### 1. Hybrid Retrieval Pipeline
To achieve high retrieval accuracy and citation granularity:
*   **Document Chunker**: Splits PDF documents page-by-page, preserving physical page numbers (`Page X`) as citations.
*   **Dense Vector Retrieval**: Uses NVIDIA NIM Embeddings (`nvidia/nv-embedqa-e5-v5`) stored in a persistent ChromaDB vector store.
*   **Sparse Retrieval**: Leverages a pickled `BM25Okapi` index dynamically filtered by document identifier (`doc_id`).
*   **Reciprocal Rank Fusion (RRF)**: Fuses lexical and semantic search results to form the top most relevant chunks.

### 2. Multi-Agent Framework
*   **Supervisor Agent**: The dispatcher. Classifies query intent and routes requests to specialist agents in parallel using Python `asyncio`.
*   **Contract Review Agent**: Reviews termination, liability, intellectual property, and warranties against Indian corporate market standards.
*   **Compliance Agent**: Checks contract clauses against Indian statutory regulations, including the **Indian Contract Act, 1872**, **Companies Act, 2013**, **IT Act, 2000**, and **SEBI** guidelines.
*   **Drafting Agent**: Creates or redrafts clauses with optimal business parameters.
*   **Verification Agent (The Grounding Engine)**: Fact-checks LLM claims by matching assertions against retrieved source snippets, generating a factual grounding score to eliminate AI hallucination.

---

## 🚀 Setup & Installation

### Prerequisites
*   Python 3.10 or higher
*   NVIDIA NIM API Key (or OpenAI / Gemini API Key configured in compatibility mode)

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/karthikj5453/LexAgent-.git
    cd LexAgent-
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Pinned Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the project root:
    ```env
    NIM_API_KEY=your_nvidia_nim_api_key_here
    NIM_BASE_URL=https://integrate.api.nvidia.com/v1
    NIM_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
    NIM_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
    ```

---

## 🔌 API Documentation & Usage

Start the development API server:
```bash
uvicorn api.main:app --reload --port 8000
```

Access the interactive API documentation at: **`http://localhost:8000/docs`**

### Endpoints

#### 1. Upload & Index Contract
*   **Endpoint**: `POST /upload`
*   **Payload**: `multipart/form-data` with key `file` (PDF file)
*   **Response**:
    ```json
    {
      "doc_id": "8a31e8c9-fa2b-42ab-9d8a-ee6465c1979b",
      "filename": "Service_Agreement.pdf",
      "chunks_indexed": 34
    }
    ```

#### 2. Chat with Supervisor
*   **Endpoint**: `POST /chat`
*   **Payload**:
    ```json
    {
      "message": "Review this agreement and flag any risky liability clauses.",
      "session_id": "session_user_001",
      "doc_id": "8a31e8c9-fa2b-42ab-9d8a-ee6465c1979b"
    }
    ```
*   **Response**:
    ```json
    {
      "response": "...", 
      "intent": {
        "intent": "review",
        "sub_tasks": ["review"],
        "requires_document": true
      },
      "agent_results": {
        "review": { ... }
      },
      "verification_passed": true
    }
    ```
