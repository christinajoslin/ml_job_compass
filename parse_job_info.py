"""




"""

import pandas as pd   # DataFrame operations 
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError # Updated import for Pydantic v2
from typing import Literal, Annotated # Import Annotated for list constraints
import json # Needed for manual JSON handling in extract_job_info
from langchain_openai import ChatOpenAI
import time 
from typing import Optional
from tqdm import tqdm 
tqdm.pandas() # enable progress_apply for pandas

jobs_df = pd.read_csv("1000_ml_jobs_us.csv")

# Add OpenAI API to .env 

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=512, 
    model_kwargs={"response_format": {"type": "json_object"}} # Enable JSON mode
)

# ML Fields by Domain 
DomainType = Literal[
    # Health & Life Sciences
    "Healthcare", "Pharmaceuticals", "Biotechnology", "Environmental Science", "Agriculture",
    # Finance & Business
    "Finance", "Insurance", "E-commerce", "Retail", "Advertising", "Marketing", "Real Estate",
    # Tech & Platforms 
    "Cybersecurity", "Telecommunications", "Software and Cloud", "Consumer Electronics", "Industrial Electronics",
    # Industry & Engineering
    "Manufacturing", "Energy", "Transportation", "Logistics", "Automotive", "Aerospace", "Construction",
    # Education & Government
    "Education", "Government", "Legal", "Defense",
    # Media & Communications
    "Entertainment", "Gaming", "Social Media", "Sports", "Publishing",
    # Services
    "Human Resources", "Travel and Hospitality", "Customer Service"
]

# ML by Type 
TypeOfML = Literal[
    # Core Paradigms 
    "Supervised Learning",
    "Unsupervised Learning",
    "Reinforcement Learning",
    "Self-Supervised Learning",
    "Transfer Learning",
    "Deep Learning",

    # Modeling & Training Techniques 
    "Bayesian Modeling",
    "Statistical Modeling",
    "Optimization",
    "Causal Inference",
    "Federated Learning",
    "Multimodal Learning",
    "Meta-Learning",
    "Simulation-Based Learning",
    "Tree-Based Models",
    "Ensemble Learning",

    # Application Areas
    "Embedded ML"
    "Time Series Analysis",
    "Anomaly Detection",
    "Graph Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Speech Recognition",
    "Generative AI",
    "Recommendation Systems",
    "Robotics",
    "Autonomous Systems",
    "Human-Computer Interaction",
    "MLOps"
]

# Define the output schema
class JobInfo(BaseModel):
    seniority_level: Literal["Early Career", "Mid-Level", "Senior", "Not mentioned"] = Field(
     description=(
        "Seniority level is determined **only** from the minimum years of experience explicitly stated "
        "in the job description, except for internships:\n"
        "- Early Career: minimum years is less than 2\n"
        "- Mid-Level: minimum years is at least 2 but less than 5\n"
        "- Senior: minimum years is 5 or more\n"
        "- Not mentioned: Use this if the job description does not specify years of experience\n\n"
        "Special rules:\n"
        "1. If the role is an internship, always assign 'Early Career'.\n"
        "2. If a range is given (e.g., '3–7 years'), use the **lower bound** (3) as the minimum.\n"
        "3. For 'at least X years' wording, treat X as the minimum.\n"
        "4. Do NOT infer seniority from job titles, responsibilities, or skills unless explicit years of experience are provided."
    )
    )
    internship: bool = Field(
        description="Whether the job is an internship or not (boolean only), 'true' if an internship and 'false' otherwise"
    )
    degree_requirements: str = Field(
         description="Degree requirement as a single string, combining degree level and fields of study (e.g., B.S. or M.S. (PhD preferred) in Data Science, Statistics, or a related field)."
    )
    programming_languages: Annotated[list[str], Field(min_length=1)] = Field(
        description="Programming languages relevant to this role (e.g., Python, Java, SQL, Scala)"
    )
    type_of_ml: Annotated[list[TypeOfML], Field(min_length=1, max_length=6)] = Field( # Updated for Pydantic v2
        description="List of 1–6 standardized ML types used in the role. Use title case. Select from a controlled vocabulary such as 'Natural Language Processing', 'Robotics', 'Time Series Analysis', etc."
    )
    libraries_and_tools: Annotated[list[str], Field(min_length=1)]= Field(
        description=(
        "ML libraries or tools mentioned (e.g., PyTorch, Docker, Kubernetes, Git, MLflow). Do not include cloud platforms such as AWS, Azure, or GCP in this list."
        "If no tools are mentioned, output a single-item list: ['Not mentioned']."
        )
    )
    cloud_platforms: Annotated[list[str], Field(min_length=1)]= Field(
        description="Cloud platforms mentioned (e.g., AWS, Azure, GCP). Use 'Not mentioned' if none."
    )
    key_responsibilities: Annotated[list[str], Field(min_length=2, max_length=4)] = Field(
        description="Short bullet list (2–4 items) summarizing main ML-related responsibilities."
    )
    domain: Annotated[list[DomainType], Field(min_length=1, max_length=2)] = Field( # Updated for Pydantic v2
    description=("One or two industry domains where the ML work is applied based on the company name (e.g., ['Healthcare'], or ['Retail', 'Advertising']). Use the most specific applicable categories.")
    )
# With Pydantic v2, use model_json_schema() for the schema
parser = PydanticOutputParser(pydantic_object=JobInfo) # Try the default pydantic_object argument for v2
format_instructions = parser.get_format_instructions() # string-version of job info schema

prompt = PromptTemplate(
    template="""
You are a helpful assistant that extracts structured information from job descriptions for machine learning (ML) related positions. This structured data will be used for downstream analysis such as clustering and dashboard visualizations of career opportunities in ML.

Given a job description, extract the following fields and output them as a JSON object that conforms to the JSON schema provided below.

{format_instructions}

Output only the JSON object that is an **instance** of the schema above, enclosed in ```json and ``` markers. Do NOT include any other text, explanations, or the prompt itself in your output.

**Job Description Input (for parsing below)**
{job_description}
""",
    partial_variables={"format_instructions": format_instructions}, # pre-filled formattinng instructions based on the job info schema
    input_variables=["job_description"]
)


def extract_job_info(job_info, prompt, max_retries=7, delay=1):
  ''' 
  Extracts structured job info using LLM and retries up to 7 times if parsing fails. 
  
  Parameters: 
  - job_info: raw job description txt 
  - prompt: a prompt template that includes {job_description}
  - max_retries: number of times to retry on failure
  - delay: delay (in seconds) between retries

  Returns: 
  - Parsed dict if successful, None otherwise 

  '''
  prompt_text = prompt.format(job_description=job_info)

  for attempt in range(1, max_retries + 1): 
    try:
      # With JSON mode enabled, the response should return an AIMessage object
      response = llm.invoke(prompt_text)
      print(f"🧠 Attempt {attempt}; LLM response:\n{response}")

      # Extract the string content from the AIMessage object
      json_string = response.content
      parsed_data = parser.parse(json_string).dict() 
      print(f"✅ Parsed successfully on attempt {attempt}")
      return parsed_data 
    except (json.JSONDecodeError, ValidationError, Exception) as e: 
      print(f"❌ Attempt {attempt} failed: {e}")
      if attempt < max_retries:
        time.sleep(delay)
      else: 
        print("❗All retries exhausted. Logging failed output.")
        return None