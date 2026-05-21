import os, json, logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from groq import Groq

logger      = logging.getLogger(__name__)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

class State(TypedDict):
    inp:     Any
    pct:     int
    shap:    List[Dict]
    context: str
    points:  List[str]

def node_predict(s):
    return s

def node_retrieve(s):
    try:
        from backend.rag import query_rag
        s["context"] = query_rag(s["inp"], s["pct"])
    except Exception as e:
        logger.warning(f"RAG failed: {e}")
        s["context"] = ""
    return s

def node_analyze(s):
    return s

def node_strategy(s):
    inp, pct, shap = s["inp"], s["pct"], s["shap"]
    context = s.get("context", "")
    factors = "\n".join([
        f"- {f['name']}: {'increases' if f['value'] > 0 else 'decreases'} win probability (impact: {abs(f['value']):.2f})"
        for f in shap[:4]
    ])
    prompt = (
        "You are a federal procurement strategy advisor.\n\n"
        f"Contract: {inp.agency}, ${inp.amount:,}, {inp.duration} days\n"
        f"Vendor experience: {inp.experience} prior contracts\n"
        f"Win probability: {pct}%\n"
        f"Q4 submission: {inp.isq4} | DC location: {inp.isdc}\n\n"
        f"Key factors:\n{factors}\n\n"
        f"Historical context: {context[:400] if context else 'N/A'}\n\n"
        "Return ONLY a JSON array of exactly 4 strings. "
        "Each is one strategic recommendation, 1-2 sentences, direct, actionable. "
        "No markdown, no numbering. "
        'Example: ["point 1", "point 2", "point 3", "point 4"]'
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        text   = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        points = json.loads(text)
        if isinstance(points, list) and len(points) >= 2:
            s["points"] = points[:4]
            return s
    except Exception as e:
        logger.error(f"Groq failed: {e}")

    outlook = "favorable" if pct >= 60 else "moderate" if pct >= 40 else "challenging"
    s["points"] = [
        f"Win probability of {pct}% indicates a {outlook} position. {'Proceed with a full proposal effort.' if pct >= 50 else 'Evaluate bid/no-bid carefully before committing resources.'}",
        f"Vendor experience is the single strongest predictor. Your {inp.experience} prior contracts {'is a meaningful advantage.' if inp.experience >= 10 else 'is below the threshold where win rates improve significantly — teaming with an experienced prime is worth considering.'}",
        f"{'Q4 submissions historically reduce win rates by ~15% due to budget exhaustion. Shift to Q1-Q2 if the timeline allows.' if inp.isq4 else 'Non-Q4 timing is a genuine advantage — agencies have fuller budgets and fewer competing bids at this point in the fiscal year.'}",
        f"DoD contracts in the $1M-$5M range show the highest win probability for vendors with proven delivery records. {'Your amount falls in this range.' if 1_000_000 <= int(inp.amount) <= 5_000_000 else 'Adjust scope to target this range if the proposal allows.'}",
    ]
    return s

def _build():
    g = StateGraph(State)
    for name, fn in [("predict", node_predict), ("retrieve", node_retrieve),
                     ("analyze", node_analyze), ("strategy", node_strategy)]:
        g.add_node(name, fn)
    g.set_entry_point("predict")
    g.add_edge("predict",  "retrieve")
    g.add_edge("retrieve", "analyze")
    g.add_edge("analyze",  "strategy")
    g.add_edge("strategy", END)
    return g.compile()

pipeline = _build()

async def run_pipeline(contract_input, model_manager):
    pct, shap = model_manager.predict(contract_input)
    result = pipeline.invoke({"inp": contract_input, "pct": pct, "shap": shap, "context": "", "points": []})
    return {"pct": result["pct"], "features": result["shap"], "points": result["points"]}
