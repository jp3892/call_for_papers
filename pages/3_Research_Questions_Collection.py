import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
RQ_PATH = "data/research_questions_cleaned.csv"
CFP_PATH = "data/cfps_map_subset.csv"
EMBED_PATH = "data/rq_embeddings_3.npy"

# === Load Model and Cache ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data


def load_data_cleaned(path):
    valid_rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", maxsplit=1)
            if len(parts) == 2 and re.match(r"\d{4}-\d{4}-\w+-\w+", parts[0]):
                valid_rows.append(parts)
    return pd.DataFrame(valid_rows[1:], columns=["unique_id", "research_questions"])

@st.cache_data
def load_data():
    rq_df = load_data_cleaned(RQ_PATH)
    cfp_df = pd.read_csv(CFP_PATH)
    df = pd.merge(rq_df, cfp_df, on="unique_id", how="left")
    df['research_questions'] = df['research_questions'].fillna("")
    return df

df = load_data()



# === Load or Generate Embeddings ===
@st.cache_data
def get_embeddings(questions):
    if os.path.exists(EMBED_PATH):
        return np.load(EMBED_PATH)
    model = load_model()
    embeddings = model.encode(questions, show_progress_bar=True)
    np.save(EMBED_PATH, embeddings)
    return embeddings

embedded_df = df[df['research_questions'].str.strip() != ""].reset_index(drop=True)
embeddings = get_embeddings(embedded_df['research_questions'].tolist())


# === Page Title and Intro ===
st.set_page_config(page_title="Research Questions Collection", layout="wide")
st.title("Research Questions Collection")

st.markdown("""
<div style="border-radius: 12px; background: #fff7e6; padding: 1.5rem; border-left: 6px solid #f4b400;
box-shadow: 0 2px 5px rgba(0,0,0,0.05); font-size: 1.05rem;">
<p><strong>Call-for-papers often pose relevant, pressing research questions that shape the outcomes of panels, articles, and chapters.</strong> 
<p>I used Gemini's 2.0 Flash LLM to extract research questions from the full dataset.</p>
<p>You can either <strong>type your own topic or question</strong> below for semantic search, or <strong>browse by field</strong> using the filter.</p>
</div>
<p>
""", unsafe_allow_html=True)

# === Search Input ===
search_query = st.text_input("Enter a research topic or question (or leave blank to browse):")

# === Controls Below Input ===
col1, col2 = st.columns([2, 2])

with col1:
    if not search_query:
        # === Show Category Filter (only in browse mode) ===
        all_categories = sorted(set(cat.strip() for cats in df['categories'].dropna() for cat in cats.split(',')))
        selected_categories = st.multiselect("Browse by Field", all_categories)
    else:
        selected_categories = []

with col2:
    # === Sort Option ===
    sort_option = st.radio("Sort results by:", ["Similarity", "Date"] if search_query else ["Date"], horizontal=True)

# === Semantic Search or Browse ===
if search_query:
    model = load_model()
    query_vec = model.encode([search_query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    
    # Create a copy of the embedded_df to avoid modifying cached df
    embedded_df = embedded_df.copy()
    embedded_df['similarity'] = sims

    results = embedded_df.sort_values("similarity", ascending=False)
else:
    results = df.copy()
    ...

    if selected_categories:
        results = results[results['categories'].apply(
            lambda x: any(cat.strip() in x.split(',') for cat in selected_categories) if pd.notna(x) else False)]

# === Sort Final Results ===
if sort_option == "Date":
    results = results.sort_values("date", ascending=False)
elif sort_option == "Similarity" and "similarity" in results.columns:
    results = results.sort_values("similarity", ascending=False)

# === Display Results ===
st.markdown(f"### Showing {len(results)} results")

for _, row in results.iterrows():
    st.markdown(f"""
**{row['research_questions']}**  
{row['date']} | *View Count:* {row['view_count']}  
[View CfP]({row['url']})  
*Categories*: {row['categories']}  
*Universities*: {row['universities']}  
""" + (f"*Similarity*: `{row['similarity']:.2f}`" if search_query else ""))
    st.markdown("---")
