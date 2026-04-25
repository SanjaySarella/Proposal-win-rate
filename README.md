# Federal Contract Win Rate Intelligence

**Sanjay Sarella** | M.S. Data Analytics, Oklahoma City University

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3--70B-green)](https://console.groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://streamlit.io)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-blue?logo=googlecloud)](https://cloud.google.com)

## Live App
**[Launch App](https://federal-contract-app-646m5mi6fq-uc.a.run.app/)**

---

## The Problem

Federal contracting is a $700B+ annual market where win rates average just 20–30% for most vendors. Bid positioning decisions are made on intuition — with no visibility into agency preferences, contract size patterns, seasonal award cycles, or historical outcomes.

This system changes that.

---

## What This Does

An end-to-end AI intelligence system built on 969 real federal contracts from USASpending.gov. It predicts win probability for any contract opportunity, explains every prediction using SHAP, retrieves similar historical contracts from a vector database, and generates a competitive strategy brief via an AI agent — all from a single interface.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  USASpending.gov API → 969 real federal contracts           │
│  Feature engineering → 9 predictive features               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  ML LAYER                                                    │
│  Random Forest Classifier → Win probability score           │
│  SHAP TreeExplainer → Feature-level attribution             │
│  Features: award amount, DoD flag, contract duration,       │
│  vendor experience, agency activity, DC location, Q4 flag   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  RAG LAYER                                                   │
│  ChromaDB vector store → 500 historical contracts indexed   │
│  SentenceTransformer embeddings (all-MiniLM-L6-v2)         │
│  Semantic retrieval → similar past awards surfaced          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  AGENT LAYER                                                 │
│  LangGraph multi-agent pipeline                             │
│  Groq + Llama 3.3-70B → strategy brief generation          │
│  Output: Competitive Assessment + 3 Actions + Risk Flags    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Dataset | 969 real federal contracts — USASpending.gov |
| Model | Random Forest Classifier |
| ROC-AUC | ~74% (data leakage identified and corrected) |
| Contracts indexed in ChromaDB | 500 |
| Inference speed | ~2 seconds via Groq API |
| Deployment | GCP Cloud Run |
| Total cost | $0 — fully open source |

---

## Engineering Decisions Worth Noting

**Real data only**
An early version used synthetic data. It was rebuilt entirely from USASpending.gov's public API — 969 contracts with verified award amounts, agency names, contract types, and durations. Simulated data does not hold up.

**Data leakage caught and corrected**
The first model produced a ROC-AUC of ~99.8% — a red flag. The root cause was a feature derived from the target variable leaking into the prediction. The target variable was redefined and the model was rebuilt. The honest result is ~74% ROC-AUC.

**Cloud-ready embeddings**
The original architecture used Ollama for embeddings — which fails on cloud servers because it connects to localhost:11434. Replaced with SentenceTransformer embeddings (all-MiniLM-L6-v2) which run natively in any environment.

**Containerized and deployed**
The app is containerized with Docker and deployed on GCP Cloud Run. Zero infrastructure management. Scales automatically.

---

## How to Use the App

1. Open the **[live app](https://federal-contract-app-646m5mi6fq-uc.a.run.app/)**
2. Configure a contract opportunity in the sidebar — agency, award amount, duration, vendor experience
3. Click **Run Win Rate Analysis**
4. The system returns a win probability score, SHAP waterfall chart, 5 similar historical contracts, and an AI-generated strategy brief with prioritized actions and risk flags

---

## Tech Stack

| Category | Tools |
|---|---|
| ML & Explainability | Random Forest · Scikit-learn · SHAP |
| Agentic AI | LangGraph · LangChain · ChromaDB · Groq + Llama 3.3-70B |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| App | Streamlit |
| Deployment | Docker · GCP Cloud Run |
| Data | USASpending.gov API · Pandas · NumPy |
| Dev | Python 3.11 · Jupyter · Git |

**100% open source. Zero cost.**

---

## Project Structure

```
Proposal-win-rate/
├── app.py                         ← Streamlit app — prediction, RAG, agent
├── src/                           ← Data pipeline and feature engineering
├── models/
│   ├── contract_model.pkl         ← Trained Random Forest model
│   └── shap_explainer.pkl         ← SHAP TreeExplainer
├── data/
│   └── processed/
│       └── contracts_features.csv ← 969 real USASpending.gov contracts
├── plots/                         ← EDA and model evaluation charts
├── Dockerfile                     ← GCP Cloud Run containerization
├── requirements.txt
└── README.md
```

---

## Author

**Sanjay Sarella**
M.S. Data Analytics — Oklahoma City University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sanjay%20Sarella-blue?logo=linkedin)](https://linkedin.com/in/sanjaysarella)
[![GitHub](https://img.shields.io/badge/GitHub-SanjaySarella-black?logo=github)](https://github.com/SanjaySarella)
