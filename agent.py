import json
import joblib
import numpy as np
import pandas as pd
import shap
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import chromadb
from chromadb.utils import embedding_functions
import warnings
warnings.filterwarnings("ignore")

# ── State definition ──────────────────────────────────────────────
class ContractState(TypedDict):
    contract_input: dict
    win_probability: Optional[float]
    shap_drivers: Optional[list]
    similar_contracts: Optional[list]
    analysis: Optional[str]
    strategy_brief: Optional[str]

# ── Load model and data ───────────────────────────────────────────
model = joblib.load("contract_model.pkl")
explainer = joblib.load("shap_explainer.pkl")
df = pd.read_csv("test_predictions.csv")
llm = OllamaLLM(model="llama3.2")

feature_cols = [
    "log_award_amount", "is_dod", "contract_duration_days",
    "recipient_experience", "agency_activity",
    "is_dc", "award_year", "is_q4", "amount_bucket_encoded"
]

# ── Set up ChromaDB ───────────────────────────────────────────────
chroma_client = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="contracts",
    embedding_function=ef
)

# Populate ChromaDB with historical contracts if empty
if collection.count() == 0:
    print("Loading historical contracts into ChromaDB...")
    raw_df = pd.read_csv("contracts_features.csv")
    docs, ids, metas = [], [], []
    for i, row in raw_df.iterrows():
        doc = (
            f"Agency: {row.get('Awarding Agency', 'Unknown')}. "
            f"Recipient: {row.get('Recipient Name', 'Unknown')}. "
            f"Award Amount: ${row.get('Award Amount', 0):,.0f}. "
            f"Contract Type: {row.get('Contract Award Type', 'Unknown')}. "
            f"Duration: {row.get('contract_duration_days', 0)} days. "
            f"DoD: {row.get('is_dod', 0)}. "
            f"DC Area: {row.get('is_dc', 0)}."
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
    print(f"Loaded {len(docs)} contracts into ChromaDB")

# ── Node 1: Prediction Node ───────────────────────────────────────
def prediction_node(state: ContractState) -> ContractState:
    print("\n[Node 1] Running prediction...")
    inp = state["contract_input"]
    features = pd.DataFrame([{
        "log_award_amount": np.log1p(inp["award_amount"]),
        "is_dod": inp["is_dod"],
        "contract_duration_days": inp["contract_duration_days"],
        "recipient_experience": inp["recipient_experience"],
        "agency_activity": inp["agency_activity"],
        "is_dc": inp["is_dc"],
        "award_year": inp["award_year"],
        "is_q4": inp["is_q4"],
        "amount_bucket_encoded": inp["amount_bucket_encoded"]
    }])

    prob = model.predict_proba(features)[0][1]

    # SHAP explanation
    sv = explainer.shap_values(features)
    if isinstance(sv, list):
        sv_class1 = sv[1][0]
    else:
        sv_class1 = sv[0, :, 1] if len(np.array(sv).shape) == 3 else sv[0]

    drivers = sorted(
        zip(feature_cols, sv_class1),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    state["win_probability"] = round(float(prob), 4)
    state["shap_drivers"] = [
        {"feature": f, "shap_value": round(float(v), 4)}
        for f, v in drivers
    ]
    print(f"  Win probability: {prob:.2%}")
    print(f"  Top driver: {drivers[0][0]}")
    return state

# ── Node 2: Retrieval Node ────────────────────────────────────────
def retrieval_node(state: ContractState) -> ContractState:
    print("\n[Node 2] Retrieving similar contracts...")
    inp = state["contract_input"]
    query = (
        f"Agency: {inp.get('agency', 'Federal Agency')}. "
        f"Award Amount: ${inp['award_amount']:,.0f}. "
        f"Duration: {inp['contract_duration_days']} days. "
        f"DoD: {inp['is_dod']}. DC Area: {inp['is_dc']}."
    )
    results = collection.query(query_texts=[query], n_results=5)
    similar = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        similar.append({
            "agency": meta["agency"],
            "recipient": meta["recipient"],
            "amount": meta["amount"],
            "contract_type": meta["contract_type"],
            "won_definitive": meta["won_definitive"]
        })
    state["similar_contracts"] = similar
    print(f"  Retrieved {len(similar)} similar contracts")
    return state

# ── Node 3: Analysis Node ─────────────────────────────────────────
def analysis_node(state: ContractState) -> ContractState:
    print("\n[Node 3] Generating competitive analysis...")
    inp = state["contract_input"]
    prob = state["win_probability"]
    drivers = state["shap_drivers"]
    similar = state["similar_contracts"]

    similar_text = "\n".join([
        f"- {s['agency']} | {s['recipient']} | "
        f"${s['amount']:,.0f} | {s['contract_type']}"
        for s in similar[:3]
    ])

    drivers_text = "\n".join([
        f"- {d['feature']}: SHAP={d['shap_value']:+.4f}"
        for d in drivers
    ])

    prompt = f"""You are a federal contracting strategy analyst.

CONTRACT OPPORTUNITY:
- Agency: {inp.get('agency', 'Federal Agency')}
- Award Amount: ${inp['award_amount']:,.0f}
- Duration: {inp['contract_duration_days']} days
- DoD Contract: {'Yes' if inp['is_dod'] else 'No'}
- DC Area Performance: {'Yes' if inp['is_dc'] else 'No'}
- Vendor Experience Score: {inp['recipient_experience']}

MODEL PREDICTION:
- Probability this is a Definitive Contract (open competition): {prob:.2%}

TOP SHAP DRIVERS (factors influencing prediction):
{drivers_text}

SIMILAR HISTORICAL CONTRACTS:
{similar_text}

Write a 3-paragraph competitive analysis covering:
1. What the prediction and SHAP drivers tell us about this opportunity
2. What the historical similar contracts reveal about agency behavior
3. Where the vendor's competitive position is strong or weak

Be specific, concise, and business-focused."""

    analysis = llm.invoke(prompt)
    state["analysis"] = analysis
    print("  Analysis complete")
    return state

# ── Node 4: Strategy Node ─────────────────────────────────────────
def strategy_node(state: ContractState) -> ContractState:
    print("\n[Node 4] Generating strategy brief...")
    inp = state["contract_input"]
    prob = state["win_probability"]
    analysis = state["analysis"]

    prompt = f"""You are a senior federal business development strategist.

Based on this analysis:
{analysis}

Win probability: {prob:.2%}
Agency: {inp.get('agency', 'Federal Agency')}
Award Amount: ${inp['award_amount']:,.0f}

Write a strategic positioning brief with exactly this structure:

COMPETITIVE ASSESSMENT: [one sentence on overall position]

TOP 3 RECOMMENDED ACTIONS:
1. [Specific action with expected outcome]
2. [Specific action with expected outcome]
3. [Specific action with expected outcome]

RISK FLAGS:
- [Key risk 1]
- [Key risk 2]

BOTTOM LINE: [One sentence summary a VP would act on]"""

    brief = llm.invoke(prompt)
    state["strategy_brief"] = brief
    print("  Strategy brief complete")
    return state

# ── Build LangGraph ───────────────────────────────────────────────
workflow = StateGraph(ContractState)
workflow.add_node("prediction", prediction_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("strategy", strategy_node)

workflow.set_entry_point("prediction")
workflow.add_edge("prediction", "retrieval")
workflow.add_edge("retrieval", "analysis")
workflow.add_edge("analysis", "strategy")
workflow.add_edge("strategy", END)

agent = workflow.compile()

# ── Test run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_contract = {
        "contract_input": {
            "agency": "Department of Defense",
            "award_amount": 15000000,
            "is_dod": 1,
            "contract_duration_days": 365,
            "recipient_experience": 8,
            "agency_activity": 630,
            "is_dc": 1,
            "award_year": 2024,
            "is_q4": 1,
            "amount_bucket_encoded": 2
        },
        "win_probability": None,
        "shap_drivers": None,
        "similar_contracts": None,
        "analysis": None,
        "strategy_brief": None
    }

    print("=" * 60)
    print("FEDERAL CONTRACT WIN RATE INTELLIGENCE AGENT")
    print("=" * 60)

    result = agent.invoke(test_contract)

    print("\n" + "=" * 60)
    print("FINAL STRATEGY BRIEF")
    print("=" * 60)
    print(result["strategy_brief"])
    print("\n" + "=" * 60)
    print(f"Win Probability: {result['win_probability']:.2%}")
    print("Top SHAP Drivers:")
    for d in result["shap_drivers"]:
        print(f"  {d['feature']}: {d['shap_value']:+.4f}")