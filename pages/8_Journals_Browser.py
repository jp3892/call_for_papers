import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Journal Explorer", layout="wide")
st.title("Journal Explorer")


# Load cleaned data
df = pd.read_csv("data/journals_clean.csv")
main_df = pd.read_csv("data/cfps_map_subset.csv")

# Merge to bring in dates and categories
merged = df.merge(main_df[["unique_id", "date", "categories"]], on="unique_id", how="left")
merged["year"] = pd.to_datetime(merged["date"], errors="coerce").dt.year

# === General Info ===
st.subheader("General Information")
st.write(f"Unique journals: {df['journal_name'].nunique():,}")
st.write("Top contributing journals:")
st.write(df["journal_name"].value_counts().head(10))

# === Contributions over time ===
st.subheader("Journal Contributions Over Time")
journal_year_counts = merged.groupby(["year", "journal_name"]).size().reset_index(name="count")
st.line_chart(journal_year_counts.pivot(index="year", columns="journal_name", values="count").fillna(0))

# === Search & Browsing ===
st.subheader("Explore Journals")

# Journal dropdown
journal_options = ["All"] + sorted(df["journal_name"].unique())
selected_journal = st.selectbox("Select a Journal", journal_options)

# Topics dropdown
topics = set()
df["key_topics"].dropna().apply(lambda x: topics.update([t.strip() for t in x.split(",")]))
topic_options = ["All"] + sorted(topics)
selected_topic = st.selectbox("Select a Topic", topic_options)

# Categories dropdown
categories = set()
merged["categories"].dropna().apply(lambda x: categories.update([c.strip() for c in x.split(",")]))
category_options = ["All"] + sorted(categories)
selected_category = st.selectbox("Select a Category", category_options)

# Apply filters
filtered = merged.copy()
if selected_journal != "All":
    filtered = filtered[filtered["journal_name"] == selected_journal]
if selected_topic != "All":
    filtered = filtered[filtered["key_topics"].str.contains(selected_topic, na=False)]
if selected_category != "All":
    filtered = filtered[filtered["categories"].str.contains(selected_category, na=False)]

st.write(filtered)
