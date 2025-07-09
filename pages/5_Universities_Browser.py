import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re

st.set_page_config(layout="wide")
st.title("University Explorer")

# === Load Data ===
@st.cache_data

def load_data():
    cfp_df = pd.read_csv("data/cfps_map_subset.csv")
    cfp_df = cfp_df.dropna(subset=["universities"])
    cfp_df["date"] = pd.to_datetime(cfp_df["date"], errors="coerce")

    return cfp_df

cfp_df = load_data()

# === View Options ===
view_option = st.radio("Choose how to explore:", ["By University", "University Overview"], horizontal=True)

# ------------------
# Option 1: BY UNIVERSITY
# ------------------
# Normalize university names from semi-colon-separated strings
def extract_university_list(series):
    return (
        series.dropna()
        .astype(str)
        .str.split(";")              # Split on semi colon
        .explode()                   # Flatten lists
        .str.strip()                 # Remove extra whitespace
        .dropna()
        .unique()
    )

if view_option == "By University":
    universities = sorted(extract_university_list(cfp_df["universities"]))

    selected_university = st.multiselect(
        "Search for a university:",
        options=universities,
        max_selections=1
    )

    if selected_university:
        selected_univ = selected_university[0]

        # Match CfPs where university string contains selected university (case-insensitive)
        filtered_df = cfp_df[
            cfp_df["universities"].astype(str).str.contains(fr'\b{re.escape(selected_univ)}\b', case=False, na=False)
        ]

        st.markdown(f"### CfPs at {selected_univ} ({len(filtered_df)} total)")

        # Optional filters
        with st.expander("🔍 Optional Filters"):
            date_range = st.date_input("Filter by Date Range:", [])
            if date_range and len(date_range) == 2:
                filtered_df = filtered_df[filtered_df["date"].between(date_range[0], date_range[1])]

            selected_cat = st.multiselect("Filter by Category:", options=sorted(cfp_df["categories"].dropna().explode().unique()))
            if selected_cat:
                filtered_df = filtered_df[filtered_df["categories"].apply(lambda x: any(cat in str(x) for cat in selected_cat))]

        for _, row in filtered_df.iterrows():
            st.markdown(f"""
**[{row['title']}]({row['url']})**  
Date: {row['date'].date() if pd.notnull(row['date']) else 'Unknown'}  
Categories: `{row['categories']}`  
View Count: `{int(row['view_count']) if pd.notnull(row['view_count']) else 'N/A'}`
---
""")
    else:
        st.stop()


# ------------------
# Option 2: UNIVERSITY OVERVIEW
# ------------------
else:
    st.subheader("Overview")

    # Explode universities if comma-separated lists
    cfp_df["universities"] = cfp_df["universities"].astype(str)
    exploded = cfp_df.assign(university=cfp_df["universities"].str.split(";")).explode("universities")

    # Clean up individual university names (remove extra spaces)
    exploded["university"] = exploded["university"].str.strip()

    agg_df = (
        exploded.groupby("university")
        .agg(
            num_cfps=("url", "count"),
            avg_views=("view_count", "mean")
        )
        .sort_values("num_cfps", ascending=False)
        .reset_index()
    )   


    st.dataframe(agg_df, use_container_width=True)

    st.bar_chart(agg_df.set_index("universities").head(20)["num_cfps"])
