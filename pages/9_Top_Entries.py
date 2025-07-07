import streamlit as st
import pandas as pd
import altair as alt

# === Load Data ===
@st.cache_data
def load_data():
    df_full = pd.read_csv("data/cfps_map_subset.csv", parse_dates=["date"])
    df_top25 = pd.read_csv("data/top25_entries.csv")
    return df_full, df_top25

df, top25 = load_data()

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("View Count Overview and Top Entries")

# === Part 1: View Count Summary ===
st.header("Overall Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Views", f"{df['view_count'].sum():,}")
col2.metric("Average Views", f"{df['view_count'].mean():,.1f}")
col3.metric("Entry Count", len(df))

# === Views Over Time (by Month) ===
df['year_month'] = df['date'].dt.to_period('M').astype(str)
views_over_time = df.groupby('year_month')['view_count'].sum().reset_index()

st.subheader("Views Over Time")
line_chart = alt.Chart(views_over_time).mark_line(point=True).encode(
    x=alt.X("year_month:T", title="Month"),
    y=alt.Y("view_count", title="Total Views"),
    tooltip=["year_month", "view_count"]
).properties(width=800, height=300)

st.altair_chart(line_chart, use_container_width=True)

# === Part 2: Top 25 Most Viewed ===
st.header("Top 25 Most Viewed CfPs")

for _, row in top25.iterrows():
    st.markdown(f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; margin-bottom:1rem;">
    <h4>{row['title']}</h4>
    <p><strong>Date:</strong> {row['date']} | <strong>👁️ Views:</strong> {row['view_count']}</p>
    <p><strong>Categories:</strong> {row['categories']}</p>
    <p><strong>Organization(s):</strong> {row.get('universities') or row.get('associations')}</p>
    <p><strong>ID:</strong> {row['unique_id']}</p>
    <p style="margin-top:0.5rem;">{row['content'][:500]}{'...' if len(row['content']) > 500 else ''}</p>
    </div>
    """, unsafe_allow_html=True)
