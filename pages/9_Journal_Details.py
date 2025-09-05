import streamlit as st
import pandas as pd
import plotly.express as px
import itertools
from collections import Counter

# === Load Data (same as in browser page) ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/journals_clean_2.csv")
    main_df = pd.read_csv("data/cfps_map_subset.csv")
    merged = df.merge(
        main_df[["unique_id", "date", "categories", "title", "url"]],
        on="unique_id", how="left"
    )
    merged["year"] = pd.to_datetime(merged["date"], errors="coerce").dt.year
    return merged

merged = load_data()

# === Check session_state ===
if "selected_journal" not in st.session_state:
    st.error("No journal selected. Please go back to the browser page.")
    st.stop()

journal_name = st.session_state["selected_journal"]
st.title(f"Journal Details: {journal_name}")

# === Filter data for this journal ===
journal_df = merged[merged["journal_name"] == journal_name]

# === Show CfPs Table ===
st.subheader("Associated CfPs")
cfp_table = journal_df[["title", "url", "year", "categories"]].sort_values(by="year", ascending=False).copy()
cfp_table["url"] = cfp_table["url"].apply(lambda x: f"[Link]({x})" if pd.notna(x) else "")
st.markdown(cfp_table.to_markdown(index=False), unsafe_allow_html=True)


# === Categories Distribution ===
st.subheader("Categories Distribution")
if "categories" in journal_df.columns:
    cat_counts = (
        journal_df["categories"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts(normalize=True) * 100
    ).reset_index()
    cat_counts.columns = ["category", "percentage"]

    # Group <5% into "Other"
    major = cat_counts[cat_counts["percentage"] >= 5]
    other = cat_counts[cat_counts["percentage"] < 5]["percentage"].sum()
    if other > 0:
        major = pd.concat([major, pd.DataFrame([{"category": "Other", "percentage": other}])])

    fig = px.pie(major, values="percentage", names="category", title=f"Categories for {journal_name}")
    st.plotly_chart(fig, use_container_width=True)


# === Special Issue Breakdown ===
if "special_issue" in journal_df.columns:
    st.subheader("Special Issue Breakdown")
    special_counts = journal_df["special_issue"].value_counts().reset_index()
    special_counts.columns = ["special_issue", "count"]

    if not special_counts.empty:
        fig2 = px.bar(
            special_counts,
            x="special_issue",
            y="count",
            title=f"Special Issues in {journal_name}",
            labels={"special_issue": "Special Issue", "count": "Number of CfPs"},
        )
        st.plotly_chart(fig2, use_container_width=True)

# === Topics Distribution ===
st.subheader("Key Topics in CfPs")

if "key_topics" in journal_df.columns:
    all_topics = (
        journal_df["key_topics"]
        .dropna()
        .apply(lambda x: [t.strip() for t in str(x).split(",") if t.strip()])
    )

    flat_topics = list(itertools.chain.from_iterable(all_topics))
    topic_counts = Counter(flat_topics)

    if topic_counts:
        topic_df = pd.DataFrame(topic_counts.items(), columns=["topic", "count"]).sort_values(by="count", ascending=False)

        fig3 = px.bar(
            topic_df.head(20),
            x="topic",
            y="count",
            title=f"Top Topics for {journal_name}",
        )
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Show full topic list"):
            st.dataframe(topic_df, use_container_width=True)
    else:
        st.info("No topics detected for this journal.")

