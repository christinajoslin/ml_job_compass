"""
Script to cluster ML job postings into topic groups and label them using BERTopic + GPT-4o.

Author: Christina Joslin
Date: 8/11/2025
Purpose:
    - Load parsed ML job postings from CSV.
    - Preprocess structured fields to create an embedding-ready text representation.
    - Embed postings using MiniLM.
    - Cluster using UMAP + HDBSCAN.
    - Use OpenAI's GPT-4o for interpretable, domain-specific topic labels.
    - Reassign outlier documents to nearest topics.
    - Save topic assignments and labeled topics to CSV for downstream dashboard use.

Key Components:
    - Embedding model: SentenceTransformer ('all-MiniLM-L6-v2')
    - Dimensionality reduction: UMAP (cosine metric)
    - Clustering: HDBSCAN
    - Topic labeling: OpenAI GPT-4o via BERTopic's `representation_model`
"""

# -------------------- Load Libraries --------------------
import os                                                    # OS-level operations (environment variables, file paths)
import re                                                    # Regex operations for cleaning text
import pandas as pd                                          # Data loading, cleaning, and manipulation
from bertopic import BERTopic                                # Topic modeling framework
from bertopic.representation import OpenAI                   # OpenAI-based representation model for topic labeling
from sentence_transformers import SentenceTransformer        # Pre-trained embedding models
from hdbscan import HDBSCAN                                  # Density-based clustering algorithm for topic discovery
from umap import UMAP                                        # Dimensionality reduction to make clustering easier
from sklearn.feature_extraction.text import CountVectorizer  # Text vectorization for keyword extraction
import openai                                                # OpenAI API calls
from cryptography.fernet import Fernet                       # Symmetric encryption/decryption for secure API key storage

# -------------------- Load and Decrypt API Key --------------------
# Read encryption key from config file
with open(".config.dat", 'rb') as key_file:
    key = key_file.read()

# Create Fernet cipher object
fernet = Fernet(key)

# Read encrypted API key from file
with open("gen_key.enc", "rb") as enc_file:
    encrypted_api_key = enc_file.read()

# Decrypt API key and store it in environment for LLM calls
decrypted_api_key = fernet.decrypt(encrypted_api_key).decode()
os.environ["OPENAI_API_KEY"] = decrypted_api_key

# -------------------- Load Data --------------------
df = pd.read_csv("parsed_1000_ml_jobs_us.csv")

# Columns used in embedding string to clean 
cols_to_clean_and_replace = [
    "type_of_ml",
    "programming_languages",
    "libraries_and_tools",
    "cloud_platforms"
]

# -------------------- Data Cleaning --------------------
def strip_brackets_quotes(text):
    """Remove surrounding brackets and single quotes from list-like strings."""
    return text.strip("[]").replace("'", "")

# Replace placeholder "['Not mentioned']" with blanks
df[cols_to_clean_and_replace] = df[cols_to_clean_and_replace].replace("['Not mentioned']", "")

# Apply cleaning function
df[cols_to_clean_and_replace] = df[cols_to_clean_and_replace].applymap(strip_brackets_quotes)

# -------------------- Build Embedding String --------------------
def make_embedding_string(row):
    """
    Combine key skill fields into one text string for embedding. 
    Use semicolons to preserve separations between categories. 

    """
    parts = [
        f"{row['type_of_ml']};",
        f"{row['programming_languages']};",
        f"{row['libraries_and_tools']};",
        f"{row['cloud_platforms']}"
    ]
    return " ".join(parts)

# Apply concatenation
df['embedding_string'] = df.apply(make_embedding_string, axis=1)

# Normalize spacing and remove redundant semicolons 
df['embedding_string'] = df['embedding_string'].str.replace(r';\s*;+', ' ', regex=True)

# -------------------- Embedding Model --------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------- OpenAI Representation Model --------------------
# Prompt definitions for consistent labeling 
system_prompt = (
    "You are an expert editor labeling ML job-topic clusters for a user-facing data dashboard."
    "Allowed labels (choose exactly one): language_and_vision, mlops_infrastructure, supervised_tabular_ml."
    "Definitions: language_and_vision = text/image/speech/multimodal modeling (e.g., Generative AI, Computer Vision, NLP); "
    "mlops_infrastructure = deployment, orchestration, monitoring, pipelines, serving, cloud; "
    "supervised_tabular_ml = regression/classification on tabular data (e.g., Time Series, Recommendations, Causal Inference)."
    "If modeling and operations are mixed, choose mlops_infrastructure only when deployment is primary."
    "Return exactly: topic: <label> (snake_case)."
)

user_prompt = """
You are given a topic to label.

Representative documents:
[DOCUMENTS]

Keywords:
[KEYWORDS]

Choose exactly one label from: language_and_vision, mlops_infrastructure, supervised_tabular_ml.

Rules:
- If modeling and operations are mixed, choose mlops_infrastructure only when deployment/production is primary.
- Output exactly one line in snake_case and nothing else:
topic: <label>
"""

# Initialize OpenAI clietn for BERTopic represetnation model 
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm_repr_model = OpenAI(
    client=client,
    model="gpt-4o",
    delay_in_seconds=5,         
    nr_docs=5,                     # Number of representative docs to send 
    doc_length=120,                # Max characters per representative doc
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    tokenizer="vectorizer"
)

# -------------------- BERTopic Models --------------------
# Clustering model (HDBSCAN) for grouping embeddings into topics
hdbscan_model = HDBSCAN(
    min_cluster_size=120,  # Minimum docs per cluster
    min_samples=5,         # Minimum density sample points
    prediction_data=True   # Store prediction info for outlier reassignment
)

# Dimensionality reduction (UMAP) before clustering
umap_model = UMAP(
    n_neighbors=51,
    min_dist=0.05,
    n_components=200,
    random_state=42,
    metric="cosine"
)

# Vectorizer for topic keywords
vectorizer_model = CountVectorizer(
    stop_words="english",   # Remove stop words
    ngram_range=(1, 2),     # Use unigrams + bigrams
    min_df=1,               # Keep even rare terms
    max_df=0.99              # Remove overly common terms
)

# Complete BERTopic model with all components
topic_model = BERTopic(
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model,
    umap_model=umap_model,
    nr_topics="auto",               # Let BERTopic decide number of topics
    vectorizer_model=vectorizer_model,
    representation_model=llm_repr_model,
    verbose=True
)

# -------------------- Fit Model --------------------
topics, probs = topic_model.fit_transform(df['embedding_string'])

# Save original topic names for later use 
info = topic_model.get_topic_info()
label_map = dict(zip(info.Topic, info.Name))

# -------------------- Reduce Outliers --------------------
# Reassign docs labeled as -1 (outliers) to nearest topics 
new_topics = topic_model.reduce_outliers(
    df['embedding_string'],
    topics,
    probabilities=probs,
    strategy="probabilities",
    threshold=0.0 # forces all outliers to be reassigned 
)

# Update topic assignments after outlier reassignment 
topic_model.update_topics(df['embedding_string'], topics=new_topics, representation_model=None)
topic_model.set_topic_labels(label_map) # Restore original names 

# -------------------- Save Labeled Topics --------------------
# Add topic IDs to dataframe 
df['topic'] = topic_model.topics_

# Create mapping from topic number to custom name 
topic_info = topic_model.get_topic_info()
topic_mapping = dict(zip(topic_info['Topic'], topic_info['CustomName']))
df['topic_name'] = df['topic'].map(topic_mapping)

# Save final dataframe with topic names 
df.to_csv("topics_1000_ml_jobs.csv", index=False)
