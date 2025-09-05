import streamlit as st
import pandas as pd
import re
import plotly.express as px

st.set_page_config(page_title="Journal Explorer", layout="wide")
st.title("Journal Explorer")


# Load cleaned data
df = pd.read_csv("data/journals_clean_2.csv")
main_df = pd.read_csv("data/cfps_map_subset.csv")

@st.cache_data
def load_data():
    df = pd.read_csv("data/journals_clean_2.csv")
    main_df = pd.read_csv("data/cfps_map_subset.csv")

    # Merge to bring in dates and categories
    merged = df.merge(main_df[["unique_id", "date", "categories", "title", "url"]], on="unique_id", how="left")
    merged["year"] = pd.to_datetime(merged["date"], errors="coerce").dt.year
    return merged, main_df

merged, main_df = load_data()


# ========================
# 1. Top Contributing Journals
# ========================
st.header("Top Contributing Journals")

# Category filter
all_categories = (
    merged["categories"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .unique()
)
selected_cats = st.multiselect("Filter by Category", sorted(all_categories))

filtered = merged.copy()
if selected_cats:
    filtered = filtered[
        filtered["categories"].apply(
            lambda x: any(cat in str(x) for cat in selected_cats)
        )
    ]

# Journal counts
journal_counts = (
    filtered["journal_name"]
    .value_counts()
    .reset_index()
)
journal_counts.columns = ["journal_name", "count"]


# Ensure count is numeric
journal_counts["count"] = pd.to_numeric(journal_counts["count"], errors="coerce")
journal_counts = journal_counts.dropna(subset=["count"])
journal_counts["count"] = journal_counts["count"].astype(int)

# Show toggle
show_all = st.checkbox("Show all journals with ≥5 contributions", value=False)

if show_all:
    top_journals = journal_counts[journal_counts["count"] >= 5]
else:
    top_journals = journal_counts.head(10)

# Buttons for navigation
for _, row in top_journals.iterrows():
    journal = row["journal_name"]
    count = row["count"]

    if st.button(f"{journal} ({count})", key=journal):
        st.session_state["selected_journal"] = journal
        st.switch_page("pages/9_Journal_Details.py")

# ========================
# 2. Journal Contributions Over Time
# ========================
st.header("Journal Contributions Over Time")

if "year" in merged.columns and merged["year"].notna().any():
    yearly_total = merged.groupby("year").size().reset_index(name="total_journal_cfps")

    all_cfps = main_df.copy()
    all_cfps["year"] = pd.to_datetime(all_cfps["date"], errors="coerce").dt.year
    yearly_all = all_cfps.groupby("year").size().reset_index(name="total_all_cfps")

    merged_counts = pd.merge(yearly_all, yearly_total, on="year", how="left").fillna(0)

    fig = px.line(
        merged_counts,
        x="year",
        y=["total_all_cfps", "total_journal_cfps"],
        labels={"value": "Number of CfPs", "year": "Year", "variable": "Type"},
        title="CfPs Over Time (All vs. Journal Submissions)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================
# 3. Explore Journals
# ========================
st.header("Explore Journals")

col1, col2, col3, col4 = st.columns(4)

with col1:
    search_query = st.text_input("Search journal name")

with col2:
    year_filter = st.selectbox("Filter by year", options=["All"] + sorted(merged["year"].dropna().unique().tolist()))

with col3:
    category_filter = st.selectbox(
        "Filter by category",
        options=["All"] + sorted(set(cat for cats in merged["categories"].dropna().str.split(",") for cat in cats)),
    )

with col4:
    special_only = st.checkbox("Show only special issues")

filtered_df = merged.copy()

if search_query:
    filtered_df = filtered_df[filtered_df["journal_name"].str.contains(search_query, case=False, na=False)]

if year_filter != "All":
    filtered_df = filtered_df[filtered_df["year"] == year_filter]

if category_filter != "All":
    filtered_df = filtered_df[filtered_df["categories"].fillna("").str.contains(category_filter, case=False)]

if special_only:
    filtered_df = filtered_df[filtered_df["special_issue"] == True]

st.dataframe(
    filtered_df[["journal_name", "title", "url", "year", "categories", "special_issue"]].sort_values(
        by="year", ascending=False
    ),
    use_container_width=True,
)
