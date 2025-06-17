# RAG-FGTS: Question Answering System with LangChain, ChromaDB, and HuggingFace

This project is a **RAG (Retrieval-Augmented Generation)** application designed to answer questions based on FGTS data. It uses **LangChain**, **ChromaDB**, **HuggingFace LLM API**, and experiment tracking with **MLflow**.

---

## Features

- Document loading with intelligent chunking
- Persistent vector database creation using HuggingFace embeddings
- QA chain with LLM (HuggingFace Zephyr)
- Similarity filtering with `score_threshold`
- Answers strictly based on relevant retrieved context
- Experiment logging using MLflow

---

## Installation

```bash
# Clone the repository
$ git clone https://github.com/seuusuario/rag-fgts.git
$ cd rag-fgts

# Create virtual environment (optional)
$ python -m venv .venv && source .venv/bin/activate  # Linux/macOS
$ .venv\Scripts\activate  # Windows

# Install dependencies
$ pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file with your HuggingFace token:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
MLFLOW_TRACKING_URI=http://localhost:5000
```

To start MLflow locally (optional):

```bash
mlflow ui
# Access at: http://localhost:5000
```

---

## Pipeline Steps

### 1. Data ingestion(need be tested)

Place `.txt` files in `data/faq/`, then run:

```bash
python scripts/01_ingest.py
```

### 2. Start the QA system

```bash
python scripts/03_query_rag.py
```

Type your question or enter `sair` to exit.

---

## Example Usage

### Question 1:

**What personal data is processed by the FGTS Operating Agent?**

**Answer:**

> According to the available information, the personal data processed by the FGTS Operating Agent includes the work card number or CPF of the contributor, full name, address, phone number, and email. Additionally, the FGTS Operator may collect information regarding salary, FGTS deductions, and FGTS payments.

### Question 2:

**How is the value and index of the credited distribution defined?**

**Answer:**

> The credited distribution value is calculated by multiplying the account balance as of 12/31/2018 by the distribution index approved by the FGTS Board.
>
> The index is defined as 100% of the FGTS profit in 2018 divided by the total eligible account balances.
>
> For example, if the FGTS profit in 2018 was R\$ 100,000 and the total eligible balance was R\$ 1,000,000, the index would be 10%.

---

## MLflow

Each question and answer is automatically logged:

- Parameters: LLM model, embeddings, threshold, query
- Metrics: response time, number of documents retrieved
- Artifacts: generated response (not yet included)

---

## Author

Developed by David Santana.

---

## Limitations and Future Improvements

The model currently has important limitations:

- May generate **inaccurate or incomplete answers**, even with vector context.
- Sometimes suffers from **hallucination**, inventing information not present in the documents.
- Risk of **truncated answers**, especially for longer outputs.

Planned improvements include:

- Prompt refinement and instruction tuning
- Post-processing strategies
- Better chunking and context retrieval

---

## License

MIT License. Feel free to contribute and expand!

