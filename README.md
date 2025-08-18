# 🧭 ML Job Compass 



## How to get the code to work 


## Directory 

```markdown 
├── .streamlit/
│   └── config.toml
├── chroma_store_/                       # Persistent Chroma DB artifacts `(created by ingest)
├── data_prep_/                          # One-off data prep & evaluation assets
│   ├── parse_job_info.py                # LLM parser for structured fields from raw job text
│   ├── parsed_test_ml_jobs_us.csv       # Sample parsed output (test subset)
│   ├── sentiment_similarity_test_v_true.png
│   ├── test_ml_jobs_us.csv              # Small raw test subset (CSV)
│   ├── test_ml_jobs_us.xlsx             # Same test subset (Excel)
│   └── Test_Parsing_Job_Info.ipynb      # Notebook for parsing/evaluation
├── .dockerignore
├── .gitignore
├── 1000_ml_jobs_us.csv                  # Raw jobs dataset (~1k postings)
├── app.py                               # Streamlit app: ML Job Compass UI
├── chroma_store.py                      # Build/query Chroma vector store (ingest + helpers)
├── docker-compose.yml                   # Dev stack (app + Ollama, etc.)
├── Dockerfile                           # App container image
├── parsed_1000_ml_jobs_us.csv           # Cleaned dataset consumed by the app
└── requirements.txt                     # Python dependencies
