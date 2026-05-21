import os, logging
logger = logging.getLogger(__name__)
_collection = None

def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        base       = os.path.dirname(os.path.dirname(__file__))
        chroma_dir = os.path.join(base, "data", "chroma_db")
        os.makedirs(chroma_dir, exist_ok=True)
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=chroma_dir)
        _collection = client.get_or_create_collection("federal_contracts", embedding_function=ef)
        if _collection.count() == 0:
            _seed(base, _collection)
        logger.info(f"ChromaDB ready: {_collection.count()} contracts indexed")
    except Exception as e:
        logger.warning(f"ChromaDB init failed: {e}")
        _collection = None
    return _collection

def _seed(base, col):
    import pandas as pd
    csv = os.path.join(base, "data", "processed", "contracts_features.csv")
    if not os.path.exists(csv):
        return
    df = pd.read_csv(csv).head(500)
    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        text = (f"Agency: {row.get('agency_name','unknown')} | "
                f"Amount: {row.get('award_amount',0):.0f} | "
                f"Duration: {row.get('contract_duration_days',0)} days | "
                f"Win: {row.get('is_won',0)}")
        docs.append(text); ids.append(f"c_{i}"); metas.append({"i": int(i)})
    col.add(documents=docs, ids=ids, metadatas=metas)
    logger.info(f"Seeded {len(docs)} contracts into ChromaDB")

def query_rag(contract_input, pct):
    col = get_collection()
    if col is None or col.count() == 0:
        return ""
    try:
        q = f"{contract_input.agency} amount {contract_input.amount} duration {contract_input.duration} win {pct}%"
        r = col.query(query_texts=[q], n_results=3)
        return " | ".join(r.get("documents", [[]])[0][:3])
    except Exception as e:
        logger.warning(f"RAG query failed: {e}")
        return ""
