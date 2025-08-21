""" 
Script used to run the "ML Compass" Streamlit app. 

Author: Christina Joslin 
Date: 8/18/2025 
Purpose:
    - Explore top ML skills by industry domain
    - Personalize a skills checklist or a phased roadmap for a target role
    - Optionally condition guidance on an ML specialization and prep window
Notes:
    - Run locally using streamlit run app.py
    - Uses Ollama for LLM text generation (MODEL_ID + OLLAMA_BASE_URL env vars)
    - Uses a local Chroma vector store to pull job-snippet context for prompts
"""
#--------------- Load Libraries --------------------
import streamlit as st                                              # Core Streamlit UI framework for building the web app
from streamlit_extras.stylable_container import stylable_container  # Adds CSS-stylable containers to Streamlit
import ast                                                          # Safely parse Python literals (e.g., stringified lists) back into Python objects
import os                                                           # Interact with the operating system (env vars, paths, etc.)
import pandas as pd                                                 # Data manipulation and analysis (DataFrames, CSV I/O)
from collections import Counter                                     # Efficient frequency counting of iterable elements
import plotly.express as px                                         # High-level interactive plotting (scatter, bar, treemap, etc.)
from datetime import datetime                                       # Work with dates/times and format them for display
import requests                                                     # HTTP client for calling web APIs (e.g., Ollama)
from chroma_store import get_collection, _load_embedder, query_topk, format_context  # Vector DB helpers:
#   - get_collection: connect to the Chroma collection
#   - _load_embedder: load the embedding model (+ device)
#   - query_topk: retrieve top-K relevant text snippets
#   - format_context: format snippets into a prompt-ready context string
from requests.adapters import HTTPAdapter                           # Mount adapters on requests.Session for custom transport/retry behavior
from urllib3.util.retry import Retry                                # Define retry/backoff policy for transient HTTP errors
from dotenv import load_dotenv                                      # Load environment variables from a local .env file


#--------------- Environment Variables --------------------

# Load any KEY=VALUE pairs in local .env file into the process environment
load_dotenv(override=False)  # override=False ensures existing env vars (e.g., from Docker) are NOT overwritten 


#--------------- Defaults For Local Ollama --------------------

# If OLLAMA_BASE_URL is not set, fall back to local host default. 
# host.docker.internal resolves from inside Docker to the host machine.
DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"  

#--------------- Cached Ollama HTTP Client --------------------

# Ensures the HTTP Session (with retry/backoff) is only created once per session
@st.cache_resource 
def _ollama_client():
    """
    Builds and caches an HTTP client + config for talking to the Ollama server.

    Returns:
      (base_url: str, model_id: str, session: requests.Session)
    """
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    model_id = os.getenv("MODEL_ID")

    # Create a resilient HTTP session with basic retry/backoff.
    session = requests.Session()
    retry = Retry(
        total=3,                  # max 3 retries 
        backoff_factor=0.5,       # wait progressively longer between retries (0s, 0.5s, 1s, 2s...) 
        status_forcelist=[429, 502, 503, 504], # retry ONLY if these codes are returned 
        allowed_methods=["GET", "POST"],       # retry applies to these request types (GET = reads, POST = submissions)
    )

    # Mounts the adapter for both http and https 
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return base_url, model_id, session

#--------------- Ollama Single-Shot Text Generation --------------------

def ollama_generate(prompt, temperature = 0.3):
    """
    Calls Ollama's /api/generate endpoint and returns text. Expects MODEL_ID to be
    set. Temperature defaults to 0.3. 
    Returns:
        str: The generated text from the model (value of the "response" field in the JSON). 
              If missing, returns an empty string "".
    """
    base_url, model_id, session = _ollama_client()

    payload = {
        "model": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,          
    }
    resp = session.post(f"{base_url}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

#--------------- App Metadata (Author & Last Update Timestamp) --------------------

# Displayed in the hero header
AUTHOR = "Christina Joslin"
UPDATED = datetime(2025, 8, 18).strftime("%b %d, %Y") 

#--------------- Initial Preprocessing------------------------# 

# Load the CSV of parsed job postings 
df = pd.read_csv("parsed_1000_ml_jobs_us.csv")

# Columns that contain string representations of lists
list_columns = ['programming_languages', 'type_of_ml', 'libraries_and_tools', 'cloud_platforms', 'key_responsibilities', 'domain']

# Convert string representations of lists to actual lists
for col in list_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

t = df["libraries_and_tools"].explode() # expand list elements into multiple rows 
t = t.astype(str).str.strip() # convert to a string 

# Map tensorflow -> Tensorflow (case-insensitive exact match)
t = t.replace(to_replace=r'(?i)^tensorflow$', value='Tensorflow', regex=True)

# Map PyTorch (case-insensitive)
t = t.replace(to_replace=r'(?i)^pytorch$', value='PyTorch', regex=True)

# Map scikit-learn variants (case-insensitive)
t = t.replace(to_replace=r'(?i)^(scikit[\s-]?learn|sklearn|scikit)$', 
              value='scikit-learn', regex=True)

# Map Spark variants (case-insensitive)
t = t.replace(to_replace=r'(?i)^(spark(\s?ml|mllib|sql)?)$', 
              value='Apache Spark', regex=True)

t = t[~t.str.match(r'(?i)^cuda$')] # Drop CUDA (case-insensitive exact match)
# Group back; also deduplicate while preserving order
df["libraries_and_tools"] = t.groupby(level=0).agg(lambda x: list(dict.fromkeys([v for v in x if v])))


# Expand cloud platform acronyms to full names
cloud_mapping = {
    "AWS": "Amazon Web Services (AWS)",
    "GCP": "Google Cloud Platform (GCP)",
    "Azure": "Microsoft Azure"
}
s = df["cloud_platforms"].explode()  # expand list elements into multiple rows 
s = s.astype(str).str.strip() # convert to a string
s = s.replace(cloud_mapping, regex=False) # replace acronyms (exact swaps only) 
df["cloud_platforms"] = s.groupby(level=0).agg(list) # group back into a list per original row 

# Expand programming language variants to normalized names
lang_mapping = {
    "Node.js": "JavaScript",
    "Nodejs" : "JavaScript",
    "Javascript": "JavaScript",  # unify casing
    "Golang": "Go",
    "GoLang" : "Go",
    "GO": "Go",
    "Presto": "SQL",
    "Spark": "SQL",
    "Spark SQL" : "SQL", 
}
l = df["programming_languages"].explode()  # expand list elements
l = l.astype(str).str.strip()  # ensure string format
l = l.replace(lang_mapping, regex=False)  # apply normalization

# Then expand C/C++ into two items
l = l.apply(lambda x: ["C", "C++"] if x == "C/C++" else x)

# Explode again in case some entries became lists
l=l.explode()

# Group back into lists per original row
df["programming_languages"] = l.groupby(level=0).agg(lambda x: list(dict.fromkeys(x)))

# Create a new column (frameworks and tools) that merges libraries and tools with cloud platforms 
# Note: This powers the "Tools & Platforms" chart so cloud providers appear alongside libraries such as PyTorch 
df["frameworks_tools"] = df.apply(
    lambda r: (r.get("libraries_and_tools", []) + r.get("cloud_platforms", [])),
    axis=1
)


#--------------- Streamlit Page Setup --------------------

# Title, layout, favicon. Then hero header metadata is rendered wiht custom CSS 
st.set_page_config(page_title="ML Job Compass", layout="wide", page_icon="🧭")

TITLE   = "🧭ML Job Compass"  
SLOGAN  = "Get the machine learning job you want by learning the skills to get there."
AUTHOR  = AUTHOR            
UPDATED = UPDATED

#--------------- One-time CSS (Fonts, Palette, Hero, Cards) --------------------

# Customizes typogrpahy, colors, hero banner, and a soft "How to use" card.
# Note: unsafe_allow_html=True enables raw <style> injection 
st.markdown("""
<style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@700;800&display=swap');

/* Palette (teal family) */
:root {
  --accent: #5aae94;
  --accent-2: #4f947e;
  --accent-3: #3b6e5c;
  --ink: #0f172a;         /* slate-900 */
  --muted: #6b7280;       /* gray-500 */
  --panel: #f2f4f7;       /* light panel */
}

/* Reset default title margin */
.block-container h1 { margin: 0; }

/* Hero band */
.hero {
  background: linear-gradient(90deg, var(--accent), #72b6a7);
  border-radius: 16px;
  padding: 28px 28px 24px 28px;
  color: white;
  margin: 4px 0 18px 0;
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.hero-title {
  font-family: "Plus Jakarta Sans", Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif;
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: 40px;
  line-height: 1.1;
  margin-bottom: 6px;
}
.hero-slogan {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif;
  font-weight: 600;
  font-size: 18px;
  opacity: 0.95;
  margin-bottom: 10px;
}
.hero-meta {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif;
  font-weight: 400;
  font-size: 14px;
  opacity: 0.9;
}

/* Soft card for 'How to use' */
.card {
  background: var(--panel);
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 16px 18px;
  margin-top: 10px;
}
.card h3 {
  margin: 0 0 8px 0;
  font-family: Inter, system-ui;
  font-weight: 700;
  color: var(--ink);
}
.card p, .card li {
  color: var(--muted);
  font-family: Inter, system-ui;
  font-size: 15px;
}
</style>
""", unsafe_allow_html=True) 

#--------------- Hero Header (Title + Slogan + Meta) --------------------

# Renders the large banner with title, slogan, and author/timestamp
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">{TITLE}</div>
      <div class="hero-slogan">{SLOGAN}</div>
      <div class="hero-meta">By <b>{AUTHOR}</b> · Last updated {UPDATED}</div>
    </div>
    """,
    unsafe_allow_html=True
)

#--------------- "How to Use" Card --------------------

# Brief instructions to orient users: exploration vs. personalization tabs. 
st.markdown(
    """
    <div class="card">
      <h3>How to use</h3>
      <ul>
        <li><b>Exploration</b>: Explore required ML skills in current U.S. job postings (2025) by industry domain.</li>
        <li><b>Personalization</b>: Choose a target domain, career level, and optional ML specialization to get a tailored <i>skills checklist</i> or <i>phased roadmap</i>.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

#--------------- Domain List & Helpers --------------------

# Builds a frequency table of industry domains, then keep only domains with at least 10 postings.  
domain_counts = Counter(d for doms in df["domain"] for d in doms if d != "Not mentioned")
domains_list = [d for d, c in domain_counts.items() if c >= 10] # If the count exceeds 10 data points 

# Safe iterator over list-like cells, skipping blanks and "Not mentioned"
def safe_item(series): 
    for x in series: 
        for v in (x or []): 
            if v and v != "Not mentioned": 
                yield v 

# Turn a select column into a (Item, Count) dataframe for top-N charting 
def plotify_df(subset, column_name, top_n): 
    items = list(safe_item(subset[column_name]))  
    counts = Counter(items)
    top_items = counts.most_common(top_n)
    return pd.DataFrame(top_items, columns =["Item","Count"])

#--------------- Tabs & Tab Styling --------------------

# Create two main tabs: "Exploration" and "Personalization" with CSS tweaks for improved readability (i.e., larger font, padding)

if "section" not in st.session_state:
    st.session_state.section = "Exploration"  # default


tab_explore, tab_recs = st.tabs(["Exploration", "Personalization"])

st.markdown("""
<style>
/* Increase tab label font size */
div[data-testid="stTabs"] button[role="tab"] p {
    font-size: 1.15rem;      /* try 1.25rem, 18px, etc. */
}

/* Make active tab a bit bolder */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
    font-weight: 600;
}

/* (optional) add some breathing room */
div[data-testid="stTabs"] button[role="tab"] {
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

#========================================================
#                 EXPLORATION TAB
#========================================================

# Interactively explore top Programming Languages, Tools/Cloud, and ML Specializations 
# for either (a) a specific industry domain, or (b) across all domains. 
# 
with tab_explore: 

    #------ Intro copy for the exploration controls ------
    st.subheader("Explore Top ML Job Skills by Industry Domain")
    st.markdown(
    """
    Use the slider controls below to identify the top (up to 10) programming languages, tools & cloud platforms, and ML specializations by **industry domain for all career levels**.  
    """
    )
    
    #------ Domain scope toggle ------
    # If checked: aggregate across ALL domains (disable domain picker).
    # If unchecked: allow user to pick one domain.
    show_overall = st.toggle(
    "Show for all industry domains",
    value=False
    )

    # Styled container for domain picker (disabled if "all domains" is on).
    with stylable_container(
        key="lang_controls_card_1",
        css_styles="""
        {
            background: #5aae94ff;                 /* light teal */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            color: #ffff
        }
        """
    ):
        st.markdown('<strong>Step 1. Select an industry domain.</strong>', unsafe_allow_html=True)
        selected_domain = st.selectbox("", domains_list, index=0, disabled=show_overall)
    
    #---------------- Domain subset selection ----------------------

    # Build the dataframe subset based on the toggle above, and set labels for chart titles 
    if show_overall: 
        subset = df.copy() 
        scope_label="All Industry Domains"
        root_label="All Industry Domains"
    else: 
        subset = df[df["domain"].apply(lambda doms: selected_domain in doms)]
        scope_label = f"Industry Domain: {selected_domain}"
        root_label=selected_domain 
    
    st.divider() 
    st.subheader(scope_label)

    #------------- Visualizations for the current scope ----------------------
    # Figure 1: Top Programming Languages (dot plot with stems)
    # Figure 2: Top Tools & Cloud Platforms (horizontal bar chart)
    # Figure 3: Top ML Specializations (treemap)

    # ------------------ Figure 1: Programming Languages ---------------------
    with stylable_container(
        key="lang_controls_card",
        css_styles="""
        {
            background: #F2F4F7;                 /* light grey */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        """
    ):
        st.markdown('<center><strong>Step 2.</strong> Use slider to select up to top 10 programming languages.</center>', unsafe_allow_html=True)
        top_n_languages = st.slider("", 3, 10, 5, key = "nlanguages")
    
    plot_df_languages = plotify_df(subset, "programming_languages", top_n_languages)
    lang_df = plot_df_languages.sort_values("Count", ascending=False).copy()


    # Build a stemmed dot plot where each language is a trace; stems match dot 
    fig1 = px.scatter(
    lang_df, x="Count", y="Item",
    color="Item",
    text=lang_df["Count"],
    color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Show dots + labels (counts as text)
    fig1.update_traces(
    mode="markers+text",
    marker=dict(size=18, line=dict(width=1, color="white")),
    textposition="middle right",
    )

    # Add “stems” back to zero for visual clarity; color stems per language trace.
    for tr in fig1.data:
        xval = float(tr.x[0])
        tr.update(error_x=dict(
            type="data", array=[0], arrayminus=[xval],
            symmetric=False, color=tr.marker.color, thickness=6, width=0
        ),
        hovertemplate=f"<b>{tr.name}</b><br>Count: {int(xval)}<extra></extra>")

    # Keep order = sorted by count; put Top #1 at top
    fig1.update_yaxes(categoryorder="array", categoryarray=lang_df["Item"].tolist())

    fig1.update_layout(
    title=f"Top {len(lang_df)} Programming Languages",
    xaxis_title="Count of postings",
    yaxis_title="Programming languages",
    showlegend=False, margin=dict(l=0, r=0, t=60, b=0)
    )   

    # Reverse y for descending
    fig1.update_yaxes(autorange="reversed")

    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ------------------- Figure 2: Tools & Cloud Platforms ------------------
    with stylable_container(
    key="tools_controls_card",
    css_styles="""
    {
            background: #F2F4F7;                 /* light grey */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    """
    ):
        st.markdown('<center><strong>Step 3.</strong> Use slider to select up to top 10 tools & cloud platforms.</center>',unsafe_allow_html=True)
        top_n_tools = st.slider("", 3, 10,5, key = "ntools")
    plot_df_tools = plotify_df(subset, "frameworks_tools",top_n_tools)

    # Horizontal bars, colored by item, with value labels and simple hover text.
    fig2 = px.bar(
    plot_df_tools.sort_values("Count", ascending=False),
    x="Count", y="Item", orientation="h",
    color="Item",
    color_discrete_sequence = px.colors.qualitative.Set3,
    text="Count"
    )
    fig2.update_traces(
     hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
    )
    fig2.update_layout(
    title=f"Top {len(plot_df_tools)} Tools & Cloud Platforms",
    xaxis_title="Count of postings", yaxis_title="Tools & platforms",
    showlegend=False, margin=dict(l=0,r=0,t=60,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider() 

# -------------------- Figure 3: ML Specializations -----------------------
    with stylable_container(
        key="ml_concepts_controls_card",
        css_styles="""
        {
            background: #F2F4F7;                 /* light grey */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        """
        ):   
        st.markdown('<center><strong>Step 4.</strong> Use slider to select up to top 10 machine learning specializations.</center>',unsafe_allow_html=True)
        top_n_concepts = st.slider("", 3, 10, 5, key = "nconcepts")
    plot_df_concepts = plotify_df(subset, "type_of_ml", top_n_concepts) 

    # Treemap with a root label (domain or "All") and rectangles sized by counts.
    fig3 = px.treemap(
    plot_df_concepts.sort_values("Count", ascending=False), 
    path=[px.Constant(root_label), "Item"], values="Count",
    color="Count",
    color_continuous_scale=px.colors.sequential.Mint
    )

    fig3.update_traces(
    textinfo="label+value", 
    hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    texttemplate="%{label}<br>%{value}",
    )

    fig3.update_layout(
    title=f"Top {len(plot_df_concepts)} Machine Learning Specializations",
    margin=dict(l=0, r=0, t=60, b=0),
    coloraxis_showscale=False
    )

    st.plotly_chart(fig3, use_container_width=True)


#========================================================
#                PERSONALIZATION TAB
#========================================================
# Collect user preferences (domain, career level, specialization, output style),
# retrieve relevant job-snippet context from a Chroma vector store,
# generate a tailored checklist or roadmap using the Ollama LLM,
# and render the result.

with tab_recs: 
    st.subheader("Personalized Skill Requirements")
    st.markdown(
        "Based on your **target industry domain, career level, and machine learning specialization (optional)**, "
        "a personalized skills checklist or phased roadmap for your desired ML job will be generated."
    )

      #------ Four-column control layout ------
    col1, col2, col3, col4 = st.columns([2, 2, 1.6, 1.6])

    # Step 1: Target Industry Domain (defaults to previously selected domain from "Exploration" tab)
    with col1: 
        with stylable_container(
        key="industrydomain_step1",
        css_styles="""
        {
            background: #5aae94ff;                 /* light teal */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            color: #ffff                         /* white text*/ 
        }
        """
        ): 
            st.markdown('<center><strong>Step 1. Industry Domain</strong></center>',unsafe_allow_html=True)
            st.markdown("") 
            st.markdown("")
        rec_domain = st.selectbox(
            "Select Target Industry Domain",
            domains_list,
            index = domains_list.index(selected_domain) if selected_domain in domains_list else 0,
            key = "rec_domain"
        )
    # Step 2: Career Level (+ optional Internship)
    with col2: 
        with stylable_container(
        key="careerlevel_step2",
        css_styles="""
        {
            background: #4f947eff;                 /* slightly darker teal */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            color: #ffff                         /* white text*/ 
        }
        """
        ): 
            st.markdown('<center><strong>Step 2. Career Level</strong></center>',unsafe_allow_html=True) 
            st.markdown("")
            st.markdown("")
         # Choose target seniority for the *future* role (assumptions encoded in the prompt later).
        target_level=st.radio(
            "Select Target Career Level",
            ["Early Career", "Mid-Level", "Senior"],
            horizontal=True,
            key="rec_experience",
            help=(
            "Select the level of the **future ML role** you want to pursue, NOT your current level.\n\n"
            "- **Early Career (0 to <2 years of industry experience)**: Includes titles such as "
            "*ML Intern*, *Associate AI Scientist*, and *ML Engineer*. Typical tasks include "
            "implementing models and supporting senior staff on projects. "
            "(If you want internship-specific guidance, also select the **Internship** option).\n"

            "- **Mid-Level (2 to <5 years of industry experience)**: Includes titles such as "
            "*AI Scientist II/III* and *Lead ML Engineer*. Typical tasks include independently training/tuning models, "
            "collaborating with software engineers for deployment, and mentoring junior members.\n"

            "- **Senior (5+ years of industry experience):** Includes titles such as *Senior ML Engineer*, "
            "*Principal AI Scientist*, or *Executive-track ML roles*. Typical tasks include leading ML projects, "
            "driving research or architecture decisions, "
             "and managing/mentoring teams.\n\n"
            "⚠️ Job titles and expectations vary by company, so treat these as general ML career guidelines."
            )
        )

        # Internship toggle appears only for Early Career targeting.
        targetting_intern = st.checkbox(
            "Internship",
            key="targetting_intern",
            disabled=(target_level != "Early Career"),
            help="Check this if you are seeking an internship position."
        )
        if target_level != "Early Career": 
            targetting_intern = False 
    # Step 3: Optional ML Specialization
    with col3: 
        APPLICATION_AREAS = ["Natural Language Processing (NLP)",
                         "Computer Vision",
                         "Speech & Audio Machine Learning",
                         "Multimodal Learning",
                         "Ensemble Learning",
                         "Causal Inference",
                         "Time Series Forecasting",
                         "Anomaly Detection",
                         "Recommender System",
                         "Graph Machine Learning",
                         "Reinforcement Learning",
                         "Embedded and Edge Machine Learning",
                         "Machine Learning Operations (MLOps)",
                         ""]
        with stylable_container(
        key="mlspecialization_step3",
        css_styles="""
        {
            background: #45816dff;                 /* darker teal */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            color: #ffff                         /* white text*/ 
        }
        """
        ): 
            st.markdown('<center><strong>Step 3. ML Specialization</strong></center>',unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
        # Specialization picker (optional); left blank = model will infer a relevant one.
        type_of_ml = st.selectbox(
        "Select Target ML Specialization (optional)",
        APPLICATION_AREAS,
        index=None, # no preselection
        placeholder="Choose a specialization",
        key="rec_ml_types",
        help="Select one specialization relevant to your machine learning interests or leave blank.")
    # Step 4: Output Style (Checklist vs Roadmap) + Prep Window (for Roadmap)
    with col4: 
        with stylable_container(
        key="outputstyle_step4",
        css_styles="""
        {
            background: #3b6e5cff;                 /* darkest teal */
            border: 1px solid #E5E7EB;           /* subtle border */
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            color: #ffff                         /* white text*/ 
        }
        """
        ): 
            st.markdown('<center><strong>Step 4. Output Style</strong></center>',unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
        # Choose between a compact skills checklist or a time-phased roadmap.
        output_style = st.radio(
            "Select Output Style",
            ["Skills checklist", "Roadmap (timeline)"],
            horizontal=False,
            key="rec_output_style",
            help=(
            "**Skills checklist**: 3–5 required skills, each with what to learn, why it matters, "
            "a concrete mini-project idea, and 1–2 named resources.\n\n"
            "**Roadmap (timeline)**: Phased plan over your chosen prep window with weekly/monthly goals, "
            "project ideas and resources."
            )
        )
        # For roadmaps, let the user define the prep window in months.
        fixed_months=None 
        if output_style == "Roadmap (timeline)": 
            fixed_months = st.slider(
                "Skills preparation window (months)",
                min_value = 1, max_value = 24, value = 6, step=1, 
                help="How many months you want to allocate for preparation."
            )
    # Primary action: trigger guidance generation
    get_recs = st.button("Generate output", type="primary", key="rec_go")

    #--------------- Guidance Generation Pipeline --------------------
    # When clicked:
    # 1) Build human-readable labels (role, specialization clause).
    # 2) Retrieve top-K relevant snippets from the vector store (domain/level/spec).
    # 3) Compose a strict, domain- and seniority-aware prompt with formatting rules.
    # 4) Call Ollama to generate either a checklist or a roadmap.
    # 5) Render markdown into the reserved slot.

    if get_recs: 
        plan_slot = st.empty() # location where the answer will be rendered 

        with st.spinner("Generating output..."): 

            # Build the identity pieces 
            role_label = "Internship" if targetting_intern else f"{target_level} full-time role"
            spec_clause = f" with a focus on {type_of_ml}" if type_of_ml else ""
            ml_clause = f"(i.e., {type_of_ml})" if type_of_ml else ""


            #------- Retrieve vector content (top-k) ------------------
            # Connect to the Chroma collection and the embedder, then query with a
            # semantic string composed of domain + target role + specialization.
            collection = get_collection() 
            embedder, device = _load_embedder()

            # Build the semantic query 
            semantic_query = " | ".join(
                p for p in [rec_domain, role_label, (type_of_ml or "").strip()] if p
            )

            snippets = query_topk(collection, embedder, semantic_query)

             #--------- Assemble the Prompt ----------------------------
            # Start with a concise, practical "career coach" persona and explicit task.
            headline = (
            f"You are a concise, practical machine learning (ML) career coach. "
            f"Generate the most specific, technically grounded guidance possible for preparing for a "
            f"{role_label} in the {rec_domain} domain{spec_clause}."
            )

            user_prompt_parts = [headline]

            # Append the context (if any). I instruct the model to use only concrete details,
            # not to quote or reveal metadata from the job snippets directly. 
            if snippets:
                ctx = format_context(snippets)
                user_prompt_parts.append(
                "Use the following ML position descriptions as background signal ONLY. "
                "Do NOT copy phrasing or expose metadata (e.g., seniority level, industry domain)."
                "Incorporate ONLY concrete, task-relevant details (e.g., skills, tools)."
                )
                user_prompt_parts.append("Context (background only, NOT to be quoted):\n" + ctx)

        
            # Output-style guidance (Checklist vs Roadmap)
            if output_style == "Skills checklist":
                user_prompt_parts.append(
                    "\n Produce a **skills checklist** with 3 to 5 items. "
                    "For each item, include: (1) what to learn, (2) why it matters **for this domain and/or ML specialization**, "
                    "(3) one concrete mini-project and (4) 1 to 2 named resources "
                    "(e.g., recommended certifications, courses, official docs)."
                )
            else:
                # Roadmap variant with either a fixed or estimated prep window.
                if fixed_months:
                    user_prompt_parts.append(
                        f"\n Create a **phased roadmap** to prepare in ~{fixed_months} months."
                    )
                else:
                    user_prompt_parts.append(
                        "\n Create a **phased roadmap** and estimate a realistic total prep time."
                    )
                user_prompt_parts.append(
                    "Break down into phases (e.g., Foundations, Core Projects, Portfolio and Interview Prep). "
                    "Specify weekly/monthly goals, concrete deliverables, and named resources (certifications, courses, official docs). "
                    f"Tailor everything to machine learning in the chosen domain and, if provided, the specialization {ml_clause}."
                )

            # Precision & formatting constraints + seniority assumptions
            user_prompt_parts.append(
                "Hard rules: avoid generic advice; be domain-specific AND machine learning-specific; prefer actionable steps over descriptions; "
                "Name resources explicitly. Format with headings and bullet points."
                "Seniority assumptions: "
                "If the target is Early Career (from Student/Intern), assume familiarity with Python syntax, Jupyter/Colab, basic NumPy/Pandas, "
                 "intro ML concepts (linear/logistic regression, overfitting, train/test split), and exposure to one ML library (scikit-learn or PyTorch basics). "
                "Do not re-teach these. "
                "If the target is Mid-Level (from Early Career), assume proficiency with Python, NumPy/Pandas, Git, Linux/bash, virtual envs, "
                "basic statistics/ML (train/val/test, cross-validation), PyTorch training loops, and basic cloud usage (S3/GCS, IAM basics). "
                "Do not re-teach these. "
                "If the target is Senior (from Mid-Level), assume mastery of model training/fine-tuning in PyTorch, code review and unit/integration testing, "
                "CI/CD (GitHub Actions/GitLab CI), containerization (Docker), basic Kubernetes, cost-aware cloud deployment, "
                "and common CV stacks (ResNet/EfficientNet/ViT/YOLO/Seg models). "
                "Do not re-teach these. "
                f"Focus ONLY on the delta (TARGET ROLE: {role_label}) required to advance to the target level."
            )
            # If no specialization selected, ask the model to infer a suitable one and disclose it.
            if not type_of_ml:
                user_prompt_parts.append(
                    "If no ML specialization was provided, pick the most relevant one for this domain and state that assumption in one short line at the top."
                )
            # Join the prompt parts into one instruction string.
            full_prompt = "\n".join(user_prompt_parts)

            # Call Ollama. Any exception is surfaced as a Streamlit error.
            try: 
                answer = ollama_generate(full_prompt)
            except Exception as e: 
                st.error(f"Oops - couldn't get recommendations: {e}")
                st.stop()
        # Render the generated markdown into the reserved slot.
        with plan_slot.container(): 
            st.markdown(answer)




