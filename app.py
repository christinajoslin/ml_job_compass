# Run as streamlit run app.py

import streamlit as st 
from streamlit_extras.stylable_container import stylable_container 
import ast 
import pandas as pd 
from collections import Counter
import plotly.express as px 
from datetime import datetime 
from openai import OpenAI
from cryptography.fernet import Fernet  

@st.cache_resource 
def _openai_client(): 
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
    return OpenAI(api_key = decrypted_api_key)

AUTHOR = "Christina Joslin"
UPDATED = datetime.now().strftime("%b %d, %Y")

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


st.set_page_config(page_title="ML Job Insights", layout="wide")
st.title("Machine Learning Job Insights")
st.markdown(
    f"<div style='font-size:0.95rem;color:#6B7280'>"
    f"By <b>{AUTHOR}</b> · Last updated {UPDATED}"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    **What this is:** An interactive view of skills in 2025 U.S. Machine Learning (ML) job postings with a personalized upskilling coach.

    **How to use it**
    - **Explore:** Pick a domain to see the top programming languages, tools & cloud platforms, and ML concepts in current postings.
    - **Recommendations:** Choose your target domain and the experience level of the role you want (or toggle Internship) to get essential skills and practice project ideas.
    """,
)
st.divider()

#---------- Count of Domains-------------# 
domain_counts = Counter(d for doms in df["domain"] for d in doms if d != "Not mentioned")
domains_list = [d for d, c in domain_counts.items() if c >= 10] # If the count exceeds 10 data points 

def safe_item(series): 
    for x in series: 
        for v in (x or []): 
            if v and v != "Not mentioned": 
                yield v 

def plotify_df(subset, column_name, top_n): 
    items = list(safe_item(subset[column_name]))  
    counts = Counter(items)
    top_items = counts.most_common(top_n)
    return pd.DataFrame(top_items, columns =["Item","Count"])
# --- Additional buttons for exploration -----# 
tab_explore, tab_recs = st.tabs(["Explore", "Recommendations"])
# put this once (before/after tabs — either is fine)
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

#------------- EXPLORE TAB ------------------# 
with tab_explore: 

    st.subheader("Explore Top ML Job Skills by Industry Domain")
    st.markdown(
    """
    Use the controls below to the top skill listed in machine learning (ML) position descriptions by **industry domain**.  
    The charts show:  
    (1) Top programming languages  
    (2) Top tools & cloud platforms  
    (3) Most-mentioned ML specializations
    """
    )
    selected_domain = st.selectbox("Select a domain", domains_list, index=0)
    st.divider() 
    st.subheader(f"Industry Domain: {selected_domain}")

    #----------------Aggregate toolsplatforms for the selected domain----------------------# 

    subset = df[df["domain"].apply(lambda doms: selected_domain in doms)]

    #------------- Show Visualizations for the Given Domain ----------------------# 

    # ------------------ Figure 1 -------------------------# 
    
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
        top_n_languages = st.slider("Top N programming languages", 3, 10, 5)
    
    plot_df_languages = plotify_df(subset, "programming_languages", top_n_languages)
    lang_df = plot_df_languages.sort_values("Count", ascending=False).copy()


    # one trace per language so stems can match the dot color
    fig1 = px.scatter(
    lang_df, x="Count", y="Item",
    color="Item",
    text=lang_df["Count"],
    color_discrete_sequence=px.colors.qualitative.Set2
    )

    # dots + labels
    fig1.update_traces(
    mode="markers+text",
    marker=dict(size=18, line=dict(width=1, color="white")),
    textposition="middle right",
    )

    # per-trace stems back to zero, colored to match each dot
    for tr in fig1.data:
        xval = float(tr.x[0])
        tr.update(error_x=dict(
            type="data", array=[0], arrayminus=[xval],
            symmetric=False, color=tr.marker.color, thickness=6, width=0
        ),
        hovertemplate=f"<b>{tr.name}</b><br>Count: {int(xval)}<extra></extra>")

    # keep explicit order (Top #1 at top)
    fig1.update_yaxes(categoryorder="array", categoryarray=lang_df["Item"].tolist())

    fig1.update_layout(
    title=f"Top {len(lang_df)} Programming Languages",
    xaxis_title="Count of postings",
    yaxis_title="Programming languages",
    showlegend=False, margin=dict(l=0, r=0, t=60, b=0)
    )   

    fig1.update_yaxes(autorange="reversed")

    st.plotly_chart(fig1, use_container_width=True)

    st.divider()


    # ------------------- Figure 2 --------------------------# 
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
        top_n_tools = st.slider("Top N tools & cloud platforms", 3, 10,5)
    plot_df_tools = plotify_df(subset, "frameworks_tools",top_n_tools)
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

# -------------------- Figure 3 -----------------------# 
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
        top_n_concepts = st.slider("Top N machine learning concepts", 3, 10, 5)
    plot_df_concepts = plotify_df(subset, "type_of_ml", top_n_concepts) 
    fig3 = px.treemap(
    plot_df_concepts.sort_values("Count", ascending=False), 
    path=[px.Constant(f"{selected_domain}"), "Item"], values="Count",
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

#---------- Recommendations Tab ----------------# 
with tab_recs: 
    st.subheader("Personalized skill recommendations")
    st.markdown(
        "Based on your **target industry domain**, **target machine learning specialization** (optional), "
        "and the **experience level of the role you want to pursue** (not your current level), "
        "we’ll suggest the core skills and practice projects to help you upskill and become competitive "
        "for these positions."
    )
    col1, col2, col3, col4 = st.columns([2, 2, 1.6,1.6])

    with col1: 
        # defaults to the same domain picked previously 
        rec_domain = st.selectbox(
            "Industry Domain",
            domains_list,
            index = domains_list.index(selected_domain) if selected_domain in domains_list else 0,
            key = "rec_domain",
            help="Pick the industry domain you're aiming to work in."
        )
    with col2: 
        target_level=st.radio(
            "Target experience level",
            ["Early Career", "Mid-Level", "Senior"],
            horizontal=True,
            key="rec_experience",
            help="Select the level of the **future** role you want to land, NOT your current level."
        )

        
        targetting_intern = st.checkbox(
            "Target internship",
            key="targetting_intern",
            disabled=(target_level != "Early Career"),
            help="Check this if you're applying for an internship position."
        )
        if target_level != "Early Career": 
            targetting_intern = False 
    
            
    with col3: 
        detail = st.radio(
            "Detail level",
            ["Essential (5 skills)", "Extended (10 skills)"],
            index=0,
            horizontal=True,
            key="rec_detail",
            help="Choose a concise core list or a longer plan."
        )
    
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
    with col4: 
        type_of_ml = st.selectbox(
        "Machine Learning Specialization (optional)",
        APPLICATION_AREAS,
        index=None, # no preselection
        placeholder="Choose a specialization",
        key="rec_ml_types",
        help="Select at most one specialization relevant to your machine learning interests or leave blank.")

    top_n_recs = 5 if detail.startswith("Essential") else 10 

    get_recs = st.button("Get recomendations", type="primary", key="rec_go")


    if get_recs: 
        plan_slot = st.empty() # location where the answer will be rendered 

        with st.spinner("Generating recommendations..."): 
            client = _openai_client() 

            # Build a simple prompt from the user's choices
            target = "Internship" if targetting_intern else f"a {target_level} full-time role"


            user_prompt_parts = [
                f"Target industry domain: {rec_domain}" 
            ]
            
            if type_of_ml == "": # only if the user selected a category then you can add this to the user prompt
                user_prompt_parts.append(f"Target machine learning specialization: {type_of_ml}")

            user_prompt_parts.append(f"Target role: {target}")
            user_prompt_parts.append(f"In bullet points, list {top_n_recs} essential skills to learn next and 1 mini-project for each. "
            "Keep it under 150 words.")

            user_prompt = "\n".join(user_prompt_parts)
            
            try: 
                resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                {"role": "system", "content": "You are a concise career coach."},
                {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                )
                answer=resp.choices[0].message.content
            except Exception as e: 
                st.error(f"Oops - couldn't get recommendations: {e}")
                st.stop()
        
        with plan_slot.container(): 
            st.markdown("### Suggested plan")
            st.markdown(answer)