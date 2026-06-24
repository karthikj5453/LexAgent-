````markdown
# ⚖️ LexAgent

> **Trustworthy Multi-Agent Legal & Compliance Intelligence Platform for Indian Enterprises**

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-NIM-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Multi--Agent-AI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Hybrid-RAG-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Verification-First-red?style=for-the-badge"/>
</p>

<p align="center">
  <b>Built for the NVIDIA India Agentic AI Open Hackathon</b>
</p>

---

## 🚨 The Problem

Legal AI systems can generate convincing answers, but they often suffer from a critical issue:

### Hallucinated Legal Claims

Example:

```text
"Section 999 of RBI Regulation states..."
````

The section may not exist.

In legal workflows, hallucinated citations can result in:

* Compliance violations
* Financial penalties
* Contractual disputes
* Loss of business trust

Most legal copilots focus on generating answers.

**LexAgent focuses on verifying answers.**

---

## 💡 Our Solution

LexAgent is a verification-first, multi-agent legal intelligence platform that combines:

* Hybrid Retrieval-Augmented Generation (RAG)
* Specialized Legal AI Agents
* NVIDIA NIM Inference
* Grounding Verification Engine

to provide citation-backed legal intelligence for Indian enterprises.

### Core Principle

> **No Citation = No Output**

Every legal claim must be linked to retrieved evidence before reaching the user.

---

# ⭐ Why LexAgent?

### Traditional Legal AI

```text
User
 ↓
LLM
 ↓
Answer
```

Problems:

❌ Hallucinated regulations

❌ Fake citations

❌ No explainability

❌ Difficult to trust

---

### LexAgent

```text
User
 ↓
Hybrid Retrieval
 ↓
Specialized Agents
 ↓
Verification Engine
 ↓
Evidence-Backed Answer
```

Benefits:

✅ Source-backed legal analysis

✅ Reduced hallucinations

✅ Explainable recommendations

✅ Enterprise-grade trust

---

# 🏗️ System Architecture

```text
                   User / API
                        │
                        ▼
                 FastAPI Gateway
          (Auth • Routing • Sessions)
                        │
                        ▼
                Supervisor Agent
      (Intent Classification & Task Routing)
                        │
 ┌──────────────┬──────────────┬──────────────┐
 │              │              │
 ▼              ▼              ▼

Contract     Compliance     Drafting
 Review         Agent         Agent

 └──────────────┼──────────────┘
                ▼

       Grounding Verification Engine
      (Fact Check • Citation Validation)

                ▼

          Final Synthesizer

                ▼

      Verified Legal Response
```

---

# 🔄 End-to-End Workflow

### Example Query

> Review this NDA and check whether it complies with Indian law.

### Step 1 — Upload Contract

User uploads an NDA.

### Step 2 — Intent Classification

Supervisor Agent identifies:

```json
{
  "intent": "review_and_compliance",
  "requires_document": true
}
```

### Step 3 — Parallel Agent Execution

#### Contract Review Agent

Analyzes:

* Liability clauses
* Confidentiality clauses
* Termination clauses
* Intellectual Property clauses

#### Compliance Agent

Checks against:

* Indian Contract Act, 1872
* Companies Act, 2013
* IT Act, 2000
* RBI Circulars
* SEBI Guidelines

### Step 4 — Hybrid Retrieval

```text
Query
  │
  ├── Dense Retrieval (ChromaDB)
  │
  ├── Sparse Retrieval (BM25)
  │
  ▼
  RRF Fusion
  ▼
Top Relevant Evidence
```

### Step 5 — NVIDIA NIM Reasoning

Retrieved evidence is passed to:

```text
NVIDIA NIM
 └─ Nemotron-70B
```

for structured legal reasoning.

### Step 6 — Verification Layer

Every generated claim is validated against retrieved evidence.

Example:

```text
Claim:
"Section 999 RBI Regulation"
```

Verification Result:

```text
No supporting source found.
```

Output:

❌ Claim Rejected

### Step 7 — Final Response

Only verified findings are returned to the user.

---

# 🛡️ Verification Engine (Core Innovation)

The Verification Agent is the trust layer of LexAgent.

Responsibilities:

* Validate citations
* Detect hallucinations
* Cross-check legal claims
* Assign grounding score
* Block unsupported outputs

### Why It Matters

Most legal AI tools answer questions.

**LexAgent proves its answers.**

This dramatically improves reliability in high-stakes legal environments.

---

# 🧠 Hybrid Retrieval Pipeline

### Dense Retrieval

**Technology**

* NVIDIA Embeddings
* ChromaDB

**Purpose**

Semantic understanding.

Example:

```text
Liability Protection
```

matches

```text
Limitation of Damages
```

---

### Sparse Retrieval

**Technology**

* BM25

**Purpose**

Exact citation matching.

Example:

```text
Section 138
```

returns exact references.

---

### Reciprocal Rank Fusion (RRF)

Combines:

```text
Dense Results
+
Sparse Results
```

to maximize retrieval accuracy.

---

# 🚀 NVIDIA-Powered Stack

| Layer            | Technology             |
| ---------------- | ---------------------- |
| LLM Inference    | NVIDIA NIM             |
| Foundation Model | Nemotron-70B           |
| Embeddings       | NVIDIA NV-Embed-QA-E5  |
| Optimization     | TensorRT-LLM           |
| Vector Search    | ChromaDB → RAPIDS cuVS |
| Safety Layer     | NeMo Guardrails        |
| Monitoring       | Arize Phoenix          |
| Deployment       | Docker Compose         |

### Why NVIDIA?

* Faster inference
* Optimized retrieval
* Enterprise deployment
* Multi-agent scalability
* Production-grade observability

---

# 🛠️ Technology Stack

### Frontend

* React
* TypeScript
* Tailwind CSS

### Backend

* FastAPI
* Python
* AsyncIO

### Retrieval

* ChromaDB
* BM25
* RRF
* NVIDIA Embeddings

### Agent Framework

* LangGraph
* Supervisor-Worker Architecture

### LLM Layer

* NVIDIA NIM
* Nemotron-70B

### Infrastructure

* Docker Compose
* Arize Phoenix
* NeMo Guardrails

---

# 📊 Evaluation Metrics

LexAgent is designed to be measurable.

### Retrieval Metrics

* Precision@5
* Recall@10
* RRF Ranking Quality

### Agent Metrics

* Intent Classification Accuracy
* Grounding Score
* Verification Pass Rate

### System Metrics

* Response Latency
* Token Usage
* Agent Execution Time

### Trust Metrics

* Citation Accuracy
* Hallucination Reduction Rate
* Evidence Coverage

---

# 🎯 Current MVP Scope

The hackathon MVP focuses on:

* ✅ NDA Review
* ✅ Compliance Analysis
* ✅ Risk Detection
* ✅ Citation Verification
* ✅ NVIDIA NIM Integration

Rather than supporting every legal workflow, LexAgent prioritizes one fully working and trustworthy end-to-end experience.

---

# 🔌 API Endpoints

## Upload Contract

```http
POST /upload
```

### Response

```json
{
  "doc_id": "8a31e8c9-fa2b-42ab-9d8a-ee6465c1979b",
  "filename": "Service_Agreement.pdf",
  "chunks_indexed": 34
}
```

---

## Chat with Supervisor

```http
POST /chat
```

### Request

```json
{
  "message": "Review this agreement and flag risky liability clauses.",
  "session_id": "session_user_001",
  "doc_id": "8a31e8c9-fa2b-42ab-9d8a-ee6465c1979b"
}
```

### Response

```json
{
  "response": "...",
  "intent": {
    "intent": "review",
    "sub_tasks": ["review"],
    "requires_document": true
  },
  "agent_results": {
    "review": {}
  },
  "verification_passed": true
}
```

---

# 🚀 Getting Started

```bash
git clone https://github.com/karthikj5453/LexAgent-.git
cd LexAgent-
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
NIM_API_KEY=your_nvidia_api_key
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
NIM_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
```

### Run Backend

```bash
uvicorn api.main:app --reload --port 8000
```

### Open API Docs

```text
http://localhost:8000/docs
```

---

# 🌍 Future Roadmap

### Phase 1

* NDA Review
* Compliance Analysis
* Verification Engine

### Phase 2

* Multi-contract support
* Advanced reranking
* RAPIDS cuVS integration

### Phase 3

* Legal Drafting Assistant
* Case Law Intelligence
* Explainable Legal AI

### Phase 4

* e-Courts Integration
* MCA / GST APIs
* Compliance Automation

### Phase 5

* Multilingual Legal AI
* Voice Legal Assistant
* Enterprise SaaS Platform

---

# 🏆 What Makes LexAgent Unique?

Most legal AI systems answer questions.

**LexAgent verifies answers.**

> Every recommendation is explainable.
>
> Every citation is traceable.
>
> Every legal claim is evidence-backed.

### Trust > Generation

Because in legal AI, trust matters more than generation.

```
```
