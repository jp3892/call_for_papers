import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Journal Explorer", layout="wide")
st.title("Journal Explorer")
st.markdown(f"**under development**")
# === Load data ===
def save_for_later():
    @st.cache_data

    def load_data():
        df = pd.read_csv("data/cfps_journals_subset.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["journals"] = df["journals"].fillna("").astype(str).str.split(";")
        return df

    cfp_df = load_data()

    # === Journal count bar chart ===
    all_journals = cfp_df.explode("journals")
    journal_counts = all_journals["journals"].value_counts().drop("", errors="ignore")
    top_n = st.slider("Show top N journals by CfP count:", min_value=5, max_value=50, value=20)

    st.bar_chart(journal_counts.head(top_n))

    # === Journal search ===
    unique_journals = sorted(set(j for sub in cfp_df["journals"] for j in sub if j.strip()))
    selected_journal = st.selectbox("Search for a journal:", options=unique_journals)

    if selected_journal:
        st.markdown(f"### CfPs linked to: *{selected_journal}*")

        filtered_df = cfp_df[cfp_df["journals"].apply(lambda js: any(re.fullmatch(re.escape(selected_journal), j.strip(), flags=re.I) for j in js))].copy()

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
