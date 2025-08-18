"""
Script used to generate the Chroma vector database (DB) for the "ML Job Compass" app.

Author: Christina Joslin
Date: 8/18/2025
Purpose:
    - Normalize and ingest ML job data into a persistent Chroma collection
    - Provide helpers to load the embedder and query top-K matches
    - Format retrieved snippets into prompt-ready context blocks
Notes:
    - Run once with: python chroma_store.py
    - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (CPU/CUDA auto)
    - Persistent path: ./chroma_store
    - Collection name: ml_jobs
"""
#--------------- Load Libraries --------------------
import ast                                              # Safely parse Python literals (e.g., stringified lists) back into Python objects
import chromadb                                         # ChromaDB client (persistent vector database)
from chromadb.config import Settings                    # Client configuration (e.g., allow_reset, persistence path)
import pandas as pd                                     # Data manipulation and CSV I/O
import torch                                            # Detect CUDA vs CPU for embedding computation
from sentence_transformers import SentenceTransformer   # Sentence embedding model loader/encoder

#--------------- Configuration Constants --------------------
# Folder where Chroma will keep/lookup its persistent collection data.
CHROMA_DIR = "./chroma_store"
# Name of the collection storing ML job embeddings. 
COLLECTION = "ml_jobs"
# Sentence-transformers model used to embed documents and queries.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#----------------- Utility Functions------------------------# 
def safe_str(x):
    """
    Converts a value into a metadata-safe string.

    Returns:
      A comma-joined string for lists, a string for scalars, or None.
    """
    if isinstance(x, list):
        return ", ".join(str(v) for v in x)
    return str(x) if x is not None else None

def embedding_string(row):
    """
    Builds a compact, information-dense text block for a single job row.

    Returns:
      A single string representing the row to be embedded.
    """
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
    """
    Loads the sentence-transformers model on CUDA if available, else CPU.

    Returns:
      (embedder: SentenceTransformer, device: str)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return emb, device

def get_collection():
    """
    Opens or creates the persistent Chroma collection.

    Returns:
      The Chroma collection object.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    return client.get_or_create_collection(COLLECTION)

def query_topk(collection, embedder, q, k = 5):
    """
    Runs a semantic nearest-neighbor query over the collection.

    Returns:
      A list of dictionaries with keys: 'text', 'meta', and 'distance'.
    """
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
    """
    Formats retrieved snippets into a compact, prompt-ready context block.

    Returns:
      A newline-joined string of labeled snippets.
    """
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
# ingest_csv:
#   1) Read the CSV containing parsed ML job postings.
#   2) Convert stringified lists back into lists (robust to edge cases).
#   3) Expand cloud platform acronyms (AWS/GCP/Azure) to full names.
#   4) Create "frameworks_tools" = libraries_and_tools + cloud_platforms (per row).
#   5) Build three parallel arrays:
#        ids  -> stable unique ids per row (e.g., "job-42")
#        docs -> embedding strings (see embedding_string)
#        metas-> flattened, human-readable metadata to accompany each doc
#   6) Encode docs in batches and upsert into the persistent Chroma collection.

def ingest_csv(csv_path="parsed_1000_ml_jobs_us.csv", batch=64):
    """
    Ingests a CSV of ML job postings into a persistent Chroma collection.

    Returns:
      None (prints progress and upserts embeddings/metadata in batches).
    """
    df = pd.read_csv(csv_path)

    list_columns = [
        "programming_languages", "type_of_ml", "libraries_and_tools",
        "cloud_platforms", "key_responsibilities", "domain"
    ]

    # Parse stringified lists
    # _to_list is defensive: handles true lists, "[...]" strings, and scalars/missing.
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

    # Expand cloud acronyms to full names (consistency with front-end display).
    mapping = {"AWS": "Amazon Web Services (AWS)", "GCP": "Google Cloud Platform (GCP)", "Azure": "Microsoft Azure"}
    df["cloud_platforms"] = df["cloud_platforms"].apply(lambda xs: [mapping.get(str(x).strip(), str(x).strip()) for x in xs])

    # Merge frameworks/tools with cloud platforms so retrieval can match either class of skill.
    df["frameworks_tools"] = df.apply(lambda r: (r.get("libraries_and_tools", []) + r.get("cloud_platforms", [])), axis=1)

    # Build parallel arrays for Chroma upsert.
    # ids: unique per row; docs: the embedding strings; metas: flattened metadata
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

    # Initialize store + embedder (device reported for transparency).
    collection = get_collection()
    embedder, device = _load_embedder()

    # Batch encode and upsert embeddings (normalize for cosine similarity stability).
    print(f"Ingesting {len(docs)} docs: collection '{COLLECTION}' @ {CHROMA_DIR} (device={device})")
    for start in range(0, len(docs), batch):
        batch_ids = ids[start:start+batch]
        batch_docs = docs[start:start+batch]
        batch_meta = metas[start:start+batch]
        embs = embedder.encode(batch_docs, batch_size=batch, normalize_embeddings=True).tolist()
        collection.upsert(ids=batch_ids, documents=batch_docs, embeddings=embs, metadatas=batch_meta)

#--------------- Script Entry Point --------------------
if __name__ == "__main__":
    # Run once to build the store
    ingest_csv()