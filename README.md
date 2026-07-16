# Enterprise-Grade Secure RAG Agent Framework

A production-ready Retrieval-Augmented Generation (RAG) framework engineered with strict data security, intelligent context chunking, and stateful multi-turn session memory.

## Core Architecture

This framework moves beyond basic, stateless RAG implementations by wrapping a deterministic parsing and retrieval pipeline into a stateful workflow graph.

* **Multi-Format Unified Ingestion:** Leverages the **Docling** engine to parse diverse document formats (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`) natively, converting layouts into clean markdown representations without disjointed parser logic.
* **Semantic Chunking:** Processes raw document layouts using structural header splitting, preserving the contextual continuity of the underlying documentation.
* **Vector Isolation Strategy:** Bridges application logic to a **PostgreSQL (pgvector)** container database cluster. The ingestion schema splits structural indexes, mapping physical arrays into a highly optimized Generalized Inverted Index (GIN) matching layout.

## Advanced Protection Layers

* **Dual-Stage Security Guardrails:** Intercepts incoming requests via a semantic similarity scoring router. If an adversarial or out-of-topic prompt deviates from the indexed core data boundaries, the pipeline short-circuits execution before invoking LLM runtimes.
* **Metadata-Driven RBAC Filters:** Embeds role and classification metadata (`jsonb`) directly inside individual document chunk records. Database retrieval queries enforce tenant and role isolation restrictions directly at the database layer.
* **Stateful Memory Persistence:** Orchestrates agent lifecycles via a **LangGraph** workflow topology. The execution runtime hooks into a managed checkpoint database engine, preserving history sequences across unique session ID threads.