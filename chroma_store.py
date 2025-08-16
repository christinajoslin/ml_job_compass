import chromadb
from chromadb.config import Settings
import pandas as pd 
from sentence_transformers import SentenceTransformer 
import torch 
import ast 


CHROMA_DIR = "./chroma_store"
COLLECTION = "ml_jobs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


#--------------- Initial Preprocessing------------------------# 
df = pd.read_csv("parsed_1000_ml_jobs_us.csv")

# Columns that contain string representations of lists
list_columns = ['programming_languages', 'type_of_ml', 'libraries_and_tools', 'cloud_platforms', 'key_responsibilities', 'domain']

# Convert string representations of lists to actual lists
for col in list_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


# Spell out the acronyms in the cloud_platforms column 

mapping = {
    "AWS": "Amazon Web Services (AWS)",
    "GCP": "Google Cloud Platform (GCP)",
    "Azure": "Microsoft Azure"
}

s = df["cloud_platforms"].explode() 
s = s.astype(str).str.strip()
s = s.replace(mapping, regex=False) # exact swaps only 

# Reassemble list 
df["cloud_platforms"] = s.groupby(level=0).agg(list)

# Create a new column (frameworks and tools) that merges libraries and tools with cloud platforms 
df["frameworks_tools"] = df.apply(
    lambda r: (r.get("libraries_and_tools", []) + r.get("cloud_platforms", [])),
    axis=1
)

def embedding_string(row): 
    return(
        f"Seniority Level: {row.get("seniority_level")}\n"
        f"Internship: {row.get("internship")}\n"
        f"Degree Requirements: {row.get("degree_requirements")}\n"
        f"Programming Languages: {row.get("programming_languages")}\n"
        f"ML Specializations: {row.get("type_of_ml")}\n"
        f"Frameworks & Tools: {row.get("frameworks_tools")}\n"
        f"Industry Domain(s): {row.get("domain")}\n"
        f"Key Responsibilities: {row.get("key_responsibilities")}\n"
    ).strip()


# Converts columns with list values to a clean string 
def safe_str(x): 
    if isinstance(x,list): 
        return ", ".join(str(v) for v in x) 
    return str(x) if x is not None else None 

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
        "source_row": i,
    })


# Persistent Chroma client/collection 
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(COLLECTION)

# GPU-aware embedder *(Use CUDA) 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Embedding device {device}")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)


# Encode and upsert in batches 
BATCH = 64 
total = len(docs) 
for i in range(0, total,BATCH): 
    batch_ids = ids[i:i+BATCH]
    batch_docs = docs[i:i+BATCH]
    batch_meta = metas[i:i+BATCH]

embs = embedder.encode(batch_docs, batch_size=BATCH, normalize_embeddings=True)

collection.upsert(
    ids = batch_ids,
    documents=batch_docs,
    embeddings=embs,
    metadatas=batch_meta,
)

print(f"Upserted {total} items into collection '{COLLECTION}' at {CHROMA_DIR}")

