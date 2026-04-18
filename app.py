import os
os.environ["HF_HUB_DISABLE_HTTPX"] = "1"
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Federal Contract Win Rate Intelligence",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Federal Contract Win Rate Intelligence")
st.markdown("*AI-powered competitive positioning for federal contract opportunities*")
st.divider()

feature_cols = [
    "log_award_amount", "is_dod", "contract_duration_days",
    "recipient_experience", "agency_activity",
    "is_dc", "award_year", "is_q4", "amount_bucket_encoded"
]

@st.cache_resource
def load_model():
    model = joblib.load("models/contract_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return model, explainer

@st.cache_resource
def get_embedding_function():
    return embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434/api/embeddings"
    )

@st.cache_resource
def load_chroma():
    client = chromadb.Client()
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name="contracts_app", embedding_function=ef
    )
    if collection.count() == 0:
        df = pd.read_csv("data/processed/contracts_features.csv")
        docs, ids, metas = [], [], []
        for i, row in df.iterrows():
            doc = (
                f"Agency: {row.get('Awarding Agency', 'Unknown')}. "
                f"Recipient: {row.get('Recipient Name', 'Unknown')}. "
                f"Amount: ${row.get('Award Amount', 0):,.0f}. "
                f"Type: {row.get('Contract Award Type', 'Unknown')}. "
                f"Duration: {row.get('contract_duration_days', 0)} days."
            )
            docs.append(doc)
            ids.append(str(i))
            metas.append({
                "agency": str(row.get("Awarding Agency", "Unknown")),
                "recipient": str(row.get("Recipient Name", "Unknown")),
                "amount": float(row.get("Award Amount", 0)),
                "contract_type": str(row.get("Contract Award Type", "Unknown")),
                "won_definitive": int(row.get("won_definitive", 0))
            })
            if len(docs) >= 500:
                break
        collection.add(documents=docs, ids=ids, metadatas=metas)
    return collection

@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

model, explainer = load_model()
collection = load_chroma()
llm = load_llm()

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("Contract Opportunity Details")

agency = st.sidebar.selectbox("Awarding Agency", [
    "Department of Defense",
    "Department of Health and Human Services",
    "Department of Homeland Security",
    "General Services Administration",
    "NASA",
    "Department of Veterans Affairs",
    "Department of State",
    "Department of Labor",
    "Other"
])

award_amount = st.sidebar.number_input(
    "Award Amount ($)", min_value=100000,
    max_value=50000000, value=5000000, step=100000
)

contract_duration = st.sidebar.slider(
    "Contract Duration (days)", 30, 3650, 365
)

recipient_experience = st.sidebar.slider(
    "Vendor Experience Score (prior contracts)", 1, 50, 5
)

agency_activity = st.sidebar.slider(
    "Agency Activity Level (contracts/year)", 1, 700, 100
)

is_dc = st.sidebar.checkbox("DC Area Performance", value=False)
is_q4 = st.sidebar.checkbox("Federal Q4 Award (July-Sept)", value=False)
award_year = st.sidebar.selectbox("Award Year", [2022, 2023, 2024], index=2)

is_dod = 1 if "Defense" in agency else 0

if award_amount < 1000000:
    amount_bucket = 0
elif award_amount < 5000000:
    amount_bucket = 1
elif award_amount < 15000000:
    amount_bucket = 2
else:
    amount_bucket = 3

# ── Analysis ──────────────────────────────────────────────────────
if st.sidebar.button("Run Win Rate Analysis", type="primary"):

    with st.spinner("Running prediction..."):
        features = pd.DataFrame([{
            "log_award_amount": np.log1p(award_amount),
            "is_dod": is_dod,
            "contract_duration_days": contract_duration,
            "recipient_experience": recipient_experience,
            "agency_activity": agency_activity,
            "is_dc": int(is_dc),
            "award_year": award_year,
            "is_q4": int(is_q4),
            "amount_bucket_encoded": amount_bucket
        }])

        prob = model.predict_proba(features)[0][1]
        sv = explainer.shap_values(features)
        if isinstance(sv, list):
            sv_class1 = sv[1][0]
        else:
            sv_class1 = sv[0, :, 1] if len(np.array(sv).shape) == 3 else sv[0]

        drivers = sorted(
            zip(feature_cols, sv_class1),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Probability", f"{prob:.1%}")
    col2.metric("Award Amount", f"${award_amount:,.0f}")
    col3.metric("Contract Duration", f"{contract_duration} days")
    col4.metric("Agency", agency.split(" of ")[-1] if " of " in agency else agency)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("SHAP Driver Analysis")
        fig, ax = plt.subplots(figsize=(6, 4))
        features_list = [d[0] for d in drivers]
        values_list = [d[1] for d in drivers]
        colors = ["#C00000" if v > 0 else "#2E75B6" for v in values_list]
        ax.barh(features_list, values_list, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top Factors Driving Prediction")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        st.subheader("Similar Historical Contracts")
        query = (
            f"Agency: {agency}. Amount: ${award_amount:,.0f}. "
            f"Duration: {contract_duration} days."
        )
        results = collection.query(query_texts=[query], n_results=5)
        similar_data = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            similar_data.append({
                "Agency": meta["agency"][:30],
                "Recipient": meta["recipient"][:25],
                "Amount": f"${meta['amount']:,.0f}",
                "Type": meta["contract_type"]
            })
        st.dataframe(pd.DataFrame(similar_data), use_container_width=True)

    st.divider()

    st.subheader("AI Strategy Brief")
    with st.spinner("Generating strategic recommendations..."):
        drivers_text = "\n".join([
            f"- {d[0]}: {d[1]:+.4f}" for d in drivers
        ])
        similar_text = "\n".join([
            f"- {s['Agency']} | {s['Recipient']} | {s['Amount']} | {s['Type']}"
            for s in similar_data[:3]
        ])
        prompt = f"""You are a senior federal business development strategist.

CONTRACT: {agency} | ${award_amount:,.0f} | {contract_duration} days
WIN PROBABILITY: {prob:.1%}
TOP SHAP DRIVERS:
{drivers_text}
SIMILAR CONTRACTS:
{similar_text}

Write a strategic brief with exactly this structure:
COMPETITIVE ASSESSMENT: [one sentence]
TOP 3 ACTIONS:
1. [action + expected outcome]
2. [action + expected outcome]
3. [action + expected outcome]
RISK FLAGS:
- [risk 1]
- [risk 2]
BOTTOM LINE: [one sentence a VP would act on]"""

        response = llm.invoke([HumanMessage(content=prompt)])
        st.markdown(response.content)

else:
    st.info("Configure the contract opportunity in the sidebar and click Run Win Rate Analysis.")
    st.markdown("""
    ### How This Works
    - **Prediction** — Random Forest model trained on 969 real federal contracts from USASpending.gov
    - **Explainability** — SHAP values identify which factors drive the win probability
    - **RAG Retrieval** — ChromaDB searches 500 historical contracts for similar past awards
    - **Strategy Agent** — LangGraph AI agent generates an actionable positioning brief
    - **Powered by** — Groq + Llama 3 for fast, free cloud inference
    """)