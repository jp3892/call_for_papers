import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer

# === Paths ===
RQ_PATH = "data/research_questions_cleaned.csv"
CFP_PATH = "data/cfps_map_subset.csv"
EMBED_PATH = "data/rq_embeddings_final.npy"

# === Load and filter research questions ===
@st.cache_data
def load_data():
    df = pd.read_csv(RQ_PATH)
    df = df[df['unique_id'].str.match(r'\d{4}-\d{4}-\w+-\w+')].copy()
    df['research_questions'] = df['research_questions'].fillna("")

    cfp_df = pd.read_csv(CFP_PATH)
    df = pd.merge(df, cfp_df, on="unique_id", how="left")
    return df

df = load_data()

# === Load and normalize embeddings ===
@st.cache_data
def get_embeddings():
    emb = np.load(EMBED_PATH)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms  # Normalize to unit vectors
    return emb

embeddings = get_embeddings()

# Optional: Debug average norm
# st.write(f"Average norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")

# === Consistency check ===
if len(df) != len(embeddings):
    st.error(f"❌ Mismatch: {len(df)} entries vs {len(embeddings)} embeddings")
    st.stop()

# === Load model ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# === Page setup ===
st.set_page_config(page_title="Research Questions Collection", layout="wide")
st.title("Research Questions Collection")

st.markdown("""
<div style="border-radius: 12px; background: #fff7e6; padding: 1.5rem; border-left: 6px solid #f4b400;
box-shadow: 0 2px 5px rgba(0,0,0,0.05); font-size: 1.05rem;">
<p><strong>Call-for-papers often pose relevant, pressing research questions that shape the outcomes of panels, articles, and chapters.</strong></p>
<p>I used Gemini's 2.0 Flash LLM to extract research questions from the full dataset.</p>
<p>You can either <strong>type your own topic or question</strong> below for semantic search, or <strong>browse by field</strong> using the filter.</p>
</div>
""", unsafe_allow_html=True)

# === User input ===
search_query = st.text_input("Enter a research topic or question (or leave blank to browse):")

# === Controls ===
col1, col2 = st.columns([2, 2])

with col1:
    if not search_query:
        all_categories = sorted(set(
            cat.strip() for cats in df['categories'].dropna() for cat in cats.split(',')
        ))
        selected_categories = st.multiselect("Browse by Field", all_categories)
    else:
        selected_categories = []

with col2:
    sort_option = st.radio("Sort results by:", ["Similarity", "Date"] if search_query else ["Date"], horizontal=True)

# === Semantic Search or Browse ===
if search_query:
    query_vec = model.encode([search_query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)  # Normalize query

    # Use fast dot product since all vectors are normalized
    sims = np.dot(embeddings, query_vec[0])

    embedded_df = df.copy()
    embedded_df['similarity'] = sims

    # Filter based on similarity threshold
    if sort_option == "Similarity":
        results = embedded_df[embedded_df['similarity'] >= 0.20]
        results = results.sort_values("similarity", ascending=False)
    else:  # sort_option == "Date"
        results = embedded_df[embedded_df['similarity'] >= 0.30]
        results = results.sort_values("date", ascending=False)

else:
    results = df.copy()
    if selected_categories:
        results = results[results['categories'].apply(
            lambda x: any(cat.strip() in x.split(',') for cat in selected_categories) if pd.notna(x) else False
        )]

# === Sort final results ===
if sort_option == "Date":
    results = results.sort_values("date", ascending=False)

# === Display results with pagination ===
st.markdown(f"### Showing {len(results)} results")

PAGE_SIZE = 10
max_page = max(1, (len(results) - 1) // PAGE_SIZE + 1)
page = st.number_input("Page", min_value=1, max_value=max_page, step=1)

start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE

for _, row in results.iloc[start:end].iterrows():
    st.markdown(f"""
**{row['research_questions']}**  
{row['date']} | *View Count:* {row['view_count']}  
[View CfP]({row['url']})  
*Categories*: {row['categories']}  
*Universities*: {row['universities']}  
""" + (f"*Similarity*: `{row['similarity']:.2f}`" if search_query else ""))
    st.markdown("---")
