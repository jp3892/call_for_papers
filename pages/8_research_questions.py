import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
RQ_PATH = "data/research_questions_output_test.csv"
CFP_PATH = "data/cfps_map_subset.csv"
EMBED_PATH = "data/rq_embeddings.npy"

# === Load Model and Cache ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    rq_df = pd.read_csv(RQ_PATH)
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

embeddings = get_embeddings(df['research_questions'].tolist())

# === Sidebar Filters ===
st.sidebar.header("📂 Filter Options")

# Filter: Categories (comma-separated)
all_categories = sorted(set(cat.strip() for cats in df['categories'].dropna() for cat in cats.split(',')))
selected_categories = st.sidebar.multiselect("Category", all_categories)

# Filter: Universities (semicolon-separated)
all_universities = sorted(set(u.strip() for unis in df['universities'].dropna() for u in unis.split(';')))
selected_universities = st.sidebar.multiselect("University", all_universities)

# Filter: Associations (semicolon-separated)
#all_assocs = sorted(set(a.strip() for assocs in df['associations'].dropna() for a in assocs.split(';')))
#selected_associations = st.sidebar.multiselect("Association", all_assocs)

# View Count Filter
view_min, view_max = int(df['view_count'].min()), int(df['view_count'].max())
view_range = st.sidebar.slider("View Count", view_min, view_max, (view_min, view_max))

# Sort Option
sort_by = st.sidebar.radio("Sort by", ["Date", "View Count"])

# === Semantic Search Input ===
st.title("🔬 Research Questions Explorer")
search_query = st.text_input("🔍 Enter a research topic or question to explore:")

# === Semantic Search ===
if search_query:
    model = load_model()
    query_vec = model.encode([search_query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    df['similarity'] = sims
    results = df.sort_values("similarity", ascending=False).head(50)
else:
    results = df.copy()

# === Apply Filters ===
if selected_categories:
    results = results[results['categories'].apply(
        lambda x: any(cat.strip() in x.split(',') for cat in selected_categories) if pd.notna(x) else False)]

if selected_universities:
    results = results[results['universities'].apply(
        lambda x: any(u.strip() in x.split(';') for u in selected_universities) if pd.notna(x) else False)]

'''if selected_associations:
    results = results[results['associations'].apply(
        lambda x: any(a.strip() in x.split(';') for a in selected_associations) if pd.notna(x) else False)]'''

results = results[(results['view_count'] >= view_range[0]) & (results['view_count'] <= view_range[1])]

if sort_by == "Date":
    results = results.sort_values("date", ascending=False)
else:
    results = results.sort_values("view_count", ascending=False)

# === Display Results ===
st.markdown(f"### Showing {len(results)} results")

for _, row in results.iterrows():
    st.markdown(f"""
** {row['research_questions']}**
- {row['date']} | 👁️ {row['view_count']}
- [View CfP]({row['url']})
- *Categories*: {row['categories']}
- *Universities*: {row['universities']}

""" + (f"-  *Similarity*: `{row['similarity']:.2f}`" if search_query else ""))
    st.markdown("---")
