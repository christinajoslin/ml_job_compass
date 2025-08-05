"""
Script to extract structured job posting information for machine learning roles using LangChain + Pydantic.

Author: Christina Joslin 
Date: 8/5/2025 
Purpose:
    - Load raw job postings from CSV.
    - Use a structured output parser with Pydantic to extract:
        • Seniority level (Early Career (0 to less than 2 years), Mid-Level (2 to less than 5 years), Senior (5 or more years), Not mentioned)
        • Internship (True/False)
        • Degree requirements
        • Programming languages
        • ML types (1-6 types from a standardized list (e.g., Computer Vision, Generative AI))
        • Libraries & tools (e.g., PyTorch, Docker, Kubernetes)
        • Cloud platforms (e.g., AWS, Azure, GCP)
        • Key responsibilities (2-4 short bullet points summarizing ML-related responsibilities)
        • Domain classification (1-2 industry domains from a standardized list where the ML work is applied) 
    - Retry parsing multiple times if schema validation fails.
    - Save the normalized structured data to CSV for downstream analysis such as clustering and dashboard visualization.
"""

# -------------------- Load Libraries --------------------
from langchain.output_parsers import PydanticOutputParser    # Parses LLM output into validated Pydantic objects
from langchain.prompts import PromptTemplate                 # Prompt templating for LLM calls
from pydantic import BaseModel, Field, ValidationError        # Data modeling and validation (Pydantic v2 syntax)
from typing import Literal, Annotated, Optional, Union        # Type hints and literal constraints
import json                                                   # Manual JSON parsing 
from langchain_openai import ChatOpenAI                       # OpenAI wrapper with LangChain
import time                                                   # Retry delay handling
import pandas as pd                                           # DataFrame operations
from tqdm import tqdm                                         # Progress bar for pandas `.apply`
import random                                                 # Randomized retry jitter                                  
import os                                                     # OS-level operations (env vars, file paths)
from cryptography.fernet import Fernet                        # Symmetric encryption/decryption for API keys

# Enable tqdm for pandas operations
tqdm.pandas()

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

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=512, 
    model_kwargs={"response_format": {"type": "json_object"}} # Enable JSON mode
)

# -------------------- Controlled Vocabulary --------------------
# Industry domain classifications
DomainType = Literal[
    # Health & Life Sciences
    "Healthcare", "Pharmaceuticals", "Biotechnology", "Environmental Science", "Agriculture",
    # Finance & Business
    "Finance", "Insurance", "E-commerce", "Retail", "Advertising", "Marketing", "Real Estate", "Consulting",
    # Tech & Platforms
    "Cybersecurity", "Telecommunications", "Software and Cloud", "Consumer Electronics", "Industrial Electronics",
    # Industry & Engineering
    "Manufacturing", "Energy", "Transportation", "Logistics", "Automotive", "Aerospace", "Space", "Construction",
    # Education & Government
    "Education", "Government", "Legal", "Defense",
    # Media & Communications
    "Entertainment", "Gaming", "Social Media", "Sports", "Publishing", "Design",
    # Services
    "Human Resources", "Travel and Hospitality", "Customer Service"
]

# Machine Learning type classifications
TypeOfML = Literal[
    # Core paradigms
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Self-Supervised Learning",
    "Transfer Learning", "Deep Learning",

    # Modeling & training techniques
    "Bayesian Modeling", "Statistical Modeling", "Optimization", "Causal Inference",
    "Federated Learning", "Multimodal Learning", "Meta-Learning", "Simulation-Based Learning",
    "Tree-Based Models", "Ensemble Learning",

    # Application areas
    "Embedded ML", "Pattern Recognition", "Predictive Modeling", "Time Series Analysis", "Anomaly Detection",
    "Graph Machine Learning", "Natural Language Processing", "Computer Vision", "Speech Recognition",
    "Generative AI", "Recommendation Systems", "Robotics", "Autonomous Systems",
    "Human-Computer Interaction", "MLOps"
]

# -------------------- Define Output Schema --------------------
class JobInfo(BaseModel):
    """
    Schema for extracting structured ML job posting data from text.
    """
    seniority_level: Literal["Early Career", "Mid-Level", "Senior", "Not mentioned"] = Field(
        description="Determine seniority strictly from stated years of experience; see detailed rules."
    )
    internship: bool = Field(description="True if internship, else False.")
    degree_requirements: str = Field(description="Degree(s) and fields as a single string.")
    programming_languages: Annotated[list[str], Field(min_length=1)] = Field(description="Programming languages.")
    type_of_ml: list[Union[TypeOfML, str]] = Field(min_length=1, max_length=6, description="List of ML types used.")
    libraries_and_tools: Annotated[list[str], Field(min_length=1)] = Field(description="ML libraries/tools, excluding cloud platforms.")
    cloud_platforms: Annotated[list[str], Field(min_length=1)] = Field(description="Cloud platforms used.")
    key_responsibilities: Annotated[list[str], Field(min_length=2, max_length=4)] = Field(description="2–4 ML-related responsibilities.")
    domain: list[Union[DomainType, str]] = Field(min_length=1, max_length=2, description="1–2 applicable industry domains.")

# -------------------- Build Prompt Template --------------------
parser = PydanticOutputParser(pydantic_object=JobInfo)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
You are a helpful assistant that extracts structured information from job descriptions for ML positions.

{format_instructions}

**Job Description Input**
{company_description}
{job_description}
""",
    partial_variables={"format_instructions": format_instructions},
    input_variables=["company_description", "job_description"]
)

# -------------------- Load Data --------------------
df = pd.read_csv("1000_ml_jobs_us.csv")
filtered_df = df[['company_description', 'job_description_text']]

# -------------------- Extraction Function --------------------
def extract_job_info(company_info, job_info, prompt, max_retries=7, delay=0.5):
    """
    Extracts structured job info from raw text via LLM, retrying if schema validation fails.
    """
    base_prompt_text = prompt.format(company_description=company_info, job_description=job_info)

    for attempt in range(1, max_retries + 1):
        try:
            # Add extra reminder if retrying
            prompt_text = base_prompt_text if attempt == 1 else base_prompt_text + \
                "\n**Reminder:** Output ONLY valid JSON matching the schema."

            # Call LLM
            response = llm.invoke(prompt_text)
            print(f"🧠 Attempt {attempt}; LLM response:\n{response}")

            # Parse JSON into Pydantic object
            json_string = response.content
            parsed_data = parser.parse(json_string).dict()
            print(f"✅ Parsed successfully on attempt {attempt}")
            return parsed_data

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay + random.uniform(0, 0.5))  # add jitter
            else:
                print("❗All retries exhausted.")
                return None

# -------------------- Apply Extraction to Dataset --------------------
filtered_df["parsed_job_info"] = filtered_df.progress_apply(
    lambda row: extract_job_info(row['company_description'], row['job_description_text'], prompt),
    axis=1
)


# -------------------- Flatten JSON and Save --------------------
new_df = pd.json_normalize(filtered_df['parsed_job_info'])
new_df.to_csv('parsed_1000_ml_jobs_us.csv', index=False)
