# 🏛️ Federal Contract Win Rate Intelligence

An end-to-end AI-powered system designed to analyze federal spending data, predict contract win probabilities, and generate strategic positioning briefs using Agentic AI.

[📊 View the Tableau Dashboard](https://public.tableau.com/app/profile/sanjay.sarella/viz/FederalContractWinRateIntelligence--CompetitivePositioningStrategy/Dashboard1)

---

## 🚀 Project Overview

This project provides a comprehensive intelligence layer for federal contractors. By leveraging historical contract data from USASpending.gov, the system identifies patterns in award behavior, predicts whether a contract will be won competitively, and provides an AI-driven "Strategy Brief" for new opportunities.

### 🌓 Key Components

1.  **📊 Data Intelligence**: Automated data pull of 1,000+ federal contracts using the USAspending API.
2.  **🧠 Predictive Modeling**: A Random Forest classifier achieved **98% accuracy** in predicting contract types and win outcomes based on duration, agency activity, and recipient experience.
3.  **⚖️ Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** to provide transparent "reasoning" behind every prediction, identifying key drivers like agency activity and award size.
4.  **🤖 Agentic AI**: A multi-node **LangGraph** agent using **Llama 3.2** and **ChromaDB (RAG)** to synthesize model outputs and historical context into actionable strategy briefs.
5.  **📈 Executive Dashboard**: A high-impact Tableau visualization for exploring market trends and competitive positioning.

---

## 🛠️ Tech Stack

- **Data**: USAspending API, Pandas, NumPy
- **Machine Learning**: Scikit-Learn, SHAP, Joblib
- **Generative AI**: LangGraph, LangChain, Ollama (Llama 3.2), ChromaDB
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Seaborn, Tableau

---

## 📂 Project Structure

- **`app.py`**: The main Streamlit dashboard serving as the primary user interface.
- **`src/`**: Modular source code for the project.
  - **`pipeline/`**: Data ingestion and processing (`data_pull.py`, `feature_engineering.py`).
  - **`modeling/`**: ML training and AI Agent logic (`model.py`, `agent.py`).
  - **`analytics/`**: EDA and downstream reporting (`eda.py`, `export_tableau.py`).
- **`data/`**: Tiered storage for datasets.
  - **`raw/`**: Original API pulls.
  - **`processed/`**: Cleaned features ready for modeling.
  - **`output/`**: Prediction results and Tableau exports.
- **`models/`**: House for trained machine learning model binaries (`.pkl`).
- **`plots/`**: Storage for all generated visualizations and analysis charts (`.png`).

---

## 🚦 Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Pull Data**:
    ```bash
    python src/pipeline/data_pull.py
    ```
3.  **Run the Pipeline**:
    Execute scripts in order: 
    - `python src/analytics/eda.py`
    - `python src/pipeline/feature_engineering.py`
    - `python src/modeling/model.py`
4.  **Launch the Dashboard**:
    ```bash
    python -m streamlit run app.py
    ```

---

## 📈 Dashboard Preview
Explore the interactive [Tableau Dashboard](https://public.tableau.com/app/profile/sanjay.sarella/viz/FederalContractWinRateIntelligence--CompetitivePositioningStrategy/Dashboard1) to see deep dives into:
- Agency-wise Award Distributions
- Recipient Experience vs. Win Rate
- Temporal Trends in Federal Contracting (Q4 Splurge Analysis)

---
*Created by Sanjay Sarella*
