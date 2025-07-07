import streamlit as st
import pandas as pd
import re


st.set_page_config(page_title="Explore by Association", layout="wide")
st.title("Associations")

# === Load data ===
@st.cache_data

def load_data():
    df = pd.read_csv("data/cfps_associations_subset.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["associations"] = df["associations"].fillna("").astype(str).str.split(";")
    return df

cfp_df = load_data()

# === Association count bar chart ===
all_associations = cfp_df.explode("associations")
associations_counts = all_associations["associations"].value_counts().drop("", errors="ignore")
top_n = st.slider("Show top N Associations by CfP count:", min_value=5, max_value=50, value=20)

st.bar_chart(associations_counts.head(top_n))

# === Association search ===
unique_associations = sorted(set(a for sub in cfp_df["associations"] for a in sub if a.strip()))
selected_association = st.selectbox("Search for an association:", options=unique_associations)

if selected_association:
    st.markdown(f"### CfPs linked to: *{selected_association}*")

    filtered_df = cfp_df[cfp_df["associations"].apply(lambda js: any(re.fullmatch(re.escape(selected_association), j.strip(), flags=re.I) for j in js))].copy()

    # Optional filters
    with st.expander("Optional Filters"):
        date_range = st.date_input("Filter by Date Range:", [])
        if date_range and len(date_range) == 2:
            filtered_df = filtered_df[filtered_df["date"].between(date_range[0], date_range[1])]

        selected_cat = st.multiselect("Filter by Category:", options=sorted(cfp_df["categories"].dropna().explode().unique()))
        if selected_cat:
            filtered_df = filtered_df[filtered_df["categories"].apply(lambda x: any(cat in str(x) for cat in selected_cat))]

    st.markdown(f"### Showing {len(filtered_df)} results")
    for _, row in filtered_df.iterrows():
        st.markdown(f"""
        **[{row['title']}]({row['url']})**  
        Date: {row['date'].date() if pd.notnull(row['date']) else 'Unknown'}  
        Categories: `{row['categories']}`  
        View Count: `{int(row['view_count']) if pd.notnull(row['view_count']) else 'N/A'}`
        ---
        """)
