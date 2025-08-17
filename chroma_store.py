import ast 
import os 

import chromadb
from chromadb.config import Settings
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "./chroma_store"
COLLECTION = "ml_jobs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#----------------- Utility Functions------------------------# 
def safe_str(x):
    if isinstance(x, list):
        return ", ".join(str(v) for v in x)
    return str(x) if x is not None else None

def embedding_string(row):
    return (
        f"Seniority Level: {row.get('seniority_level')}\n"
        f"Internship: {row.get('internship')}\n"
        f"Degree Requirements: {row.get('degree_requirements')}\n"
        f"Programming Languages: {row.get('programming_languages')}\n"
        f"ML Specializations: {row.get('type_of_ml')}\n"
        f"Frameworks & Tools: {row.get('frameworks_tools')}\n"
        f"Industry Domain(s): {row.get('domain')}\n"
        f"Key Responsibilities: {row.get('key_responsibilities')}\n"
    ).strip()

def _load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return emb, device

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    return client.get_or_create_collection(COLLECTION)

def query_topk(collection, embedder, q, k = 5):
    """Return top-k matches as list of dicts with text/meta/distance."""
    if not (q or "").strip():
        return []
    q_emb = embedder.encode([q], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=max(1, k))
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "meta": meta, "distance": float(dist)})
    return out

def format_context(snippets, max_chars_per=1000):
    """Format snippets into a compact 'Context' block."""
    lines = []
    for i, s in enumerate(snippets, 1):
        level = s["meta"].get("seniority_level", "n/a")
        dom = s["meta"].get("domain", "n/a")
        ml = s["meta"].get("type_of_ml", "n/a")
        body = (s["text"] or "")[:max_chars_per]
        lines.append(f"[{i}] domain={dom}; level={level}; ml={ml}\n{body}")
    return "\n\n".join(lines)

#---------------- Ingestion (Run ONCE)--------------# 

def ingest_csv(csv_path="parsed_1000_ml_jobs_us.csv", batch=64):
    df = pd.read_csv(csv_path)

    list_columns = [
        "programming_languages", "type_of_ml", "libraries_and_tools",
        "cloud_platforms", "key_responsibilities", "domain"
    ]

    # Parse stringified lists
    for col in list_columns:
        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                s = x.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        v = ast.literal_eval(s)
                        return v if isinstance(v, list) else [v]
                    except Exception:
                        return [s]
                return [s] if s else []
            return []
        df[col] = df[col].apply(_to_list)

    mapping = {"AWS": "Amazon Web Services (AWS)", "GCP": "Google Cloud Platform (GCP)", "Azure": "Microsoft Azure"}
    df["cloud_platforms"] = df["cloud_platforms"].apply(lambda xs: [mapping.get(str(x).strip(), str(x).strip()) for x in xs])

    df["frameworks_tools"] = df.apply(lambda r: (r.get("libraries_and_tools", []) + r.get("cloud_platforms", [])), axis=1)

    ids, docs, metas = [], [], []
    for i, row in df.iterrows():
        ids.append(f"job-{i}")
        docs.append(embedding_string(row))
        metas.append({
            "seniority_level": safe_str(row.get("seniority_level")),
            "internship": safe_str(row.get("internship")),
            "degree_requirements": safe_str(row.get("degree_requirements")),
            "programming_languages": safe_str(row.get("programming_languages")),
            "type_of_ml": safe_str(row.get("type_of_ml")),
            "frameworks_tools": safe_str(row.get("frameworks_tools")),
            "domain": safe_str(row.get("domain")),
            "key_responsibilities": safe_str(row.get("key_responsibilities")),
            "source_row": int(i),
        })

    collection = get_collection()
    embedder, device = _load_embedder()

    print(f"Ingesting {len(docs)} docs: collection '{COLLECTION}' @ {CHROMA_DIR} (device={device})")
    for start in range(0, len(docs), batch):
        batch_ids = ids[start:start+batch]
        batch_docs = docs[start:start+batch]
        batch_meta = metas[start:start+batch]
        embs = embedder.encode(batch_docs, batch_size=batch, normalize_embeddings=True).tolist()
        collection.upsert(ids=batch_ids, documents=batch_docs, embeddings=embs, metadatas=batch_meta)

if __name__ == "__main__":
    # Run once to build the store
    ingest_csv()