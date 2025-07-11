import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import altair as alt
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
    # === Optional category filter ===
    st.markdown("### Optional Filters")
    all_categories = sorted(set(cat.strip() for cats in cfp_df["categories"].dropna() for cat in cats.split(",")))
    selected_category = st.selectbox("Filter by Category:", ["All"] + all_categories)

    # Filter the base DataFrame
    filtered_df = cfp_df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["categories"].fillna("").str.contains(fr"\b{re.escape(selected_category)}\b", case=False)]


    # Clean and explode by semicolon (not comma!)
    cfp_df["universities"] = cfp_df["universities"].astype(str)
    exploded = filtered_df.assign(university=filtered_df["universities"].str.split(";")).explode("university")


    # Clean up whitespace
    exploded["university"] = exploded["university"].str.strip()

    # Drop any blank entries caused by bad separators
    exploded = exploded[exploded["university"].notna() & (exploded["university"] != "")]

    # Remove junk: empty strings or literal "nan" values
    exploded = exploded[
        exploded["university"].notna() &
        (exploded["university"].str.lower() != "nan") &
        (exploded["university"] != "")
    ]

    # Aggregate
    agg_df = (
        exploded.groupby("university")
        .agg(
            num_cfps=("url", "count"),
            avg_views=("view_count", "mean")
        )
        .sort_values("num_cfps", ascending=False)
        .reset_index()
    )

    # Show results
    st.dataframe(agg_df, use_container_width=True)


    # Take top 20 universities by CfP count
    top20 = agg_df.sort_values("num_cfps", ascending=False).head(20)

    # Create horizontal bar chart with descending order and varying color
    bar_chart = alt.Chart(top20).mark_bar().encode(
        x=alt.X("num_cfps:Q", title="Number of CfPs"),
        y=alt.Y("university:N", sort='-x', title="University"),
        color=alt.Color("num_cfps:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["university", "num_cfps", "avg_views"]
    ).properties(width=700, height=500)

    st.altair_chart(bar_chart, use_container_width=True)

