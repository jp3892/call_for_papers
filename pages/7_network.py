import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os

# === Paths ===
DATA_PATH = "data/cfps_map_subset.csv"

st.set_page_config(page_title="University-CfPs Network", layout="wide")
st.title("🎓 University–CfPs Network Graph")

st.write("Files in /data:")
st.write(os.listdir("data") if os.path.exists("data") else "No data folder")


# === Load data ===
@st.cache_data

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["universities"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df

df = load_data()

# === Sidebar filters ===
st.sidebar.header("Filters")

category_options = sorted(set(cat.strip() for cats in df["categories"].dropna() for cat in cats.split(",")))
selected_category = st.sidebar.selectbox("Filter by Category:", ["All"] + category_options)

min_year = int(df["date"].dt.year.min())
max_year = int(df["date"].dt.year.max())
selected_year_range = st.sidebar.slider("Year Range:", min_year, max_year, (min_year, max_year))

view_threshold = st.sidebar.number_input("Minimum View Count:", value=0, min_value=0, step=1)

min_universities = st.sidebar.number_input("Minimum Universities per CfP:", value=1, min_value=1)

max_nodes = st.sidebar.number_input("Maximum CfPs in Network (for speed):", value=500, min_value=10, step=10)

# === Apply filters ===
if selected_category != "All":
    df = df[df["categories"].str.contains(fr"\b{selected_category}\b", case=False, na=False)]

df = df[
    (df["date"].dt.year >= selected_year_range[0]) &
    (df["date"].dt.year <= selected_year_range[1]) &
    (df["view_count"].fillna(0) >= view_threshold)
]

# === Explode universities ===
df["universities"] = df["universities"].fillna("")
df_exploded = df.assign(university=df["universities"].str.split(",")).explode("university")
df_exploded["university"] = df_exploded["university"].str.strip()

# === Filter by min_universities ===
univ_counts = df_exploded.groupby("unique_id")["university"].nunique()
valid_ids = univ_counts[univ_counts >= min_universities].index
df_exploded = df_exploded[df_exploded["unique_id"].isin(valid_ids)]

# === Limit size ===
cfp_ids = df_exploded["unique_id"].drop_duplicates().head(max_nodes)
df_exploded = df_exploded[df_exploded["unique_id"].isin(cfp_ids)]

# === Create Graph ===
G = nx.Graph()

for _, row in df_exploded.iterrows():
    cfp_node = f"cfp_{row['unique_id']}"
    uni_node = f"uni_{row['university']}"

    G.add_node(cfp_node, label=row["title"], type="cfp", url=row.get("url", "#"))
    G.add_node(uni_node, label=row["university"], type="uni")
    G.add_edge(cfp_node, uni_node)

# === Visualize with Pyvis ===
net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000")

for node, data in G.nodes(data=True):
    if data["type"] == "cfp":
        net.add_node(node, label=data["label"][:80], title=data["label"], shape="dot", color="#1f78b4", size=10, href=data.get("url"))
    else:
        net.add_node(node, label=data["label"], title=data["label"], shape="box", color="#33a02c")

for source, target in G.edges():
    net.add_edge(source, target)

net.set_options('''
  var options = {
    "nodes": {
      "font": {"size": 14}
    },
    "edges": {
      "color": {"inherit": true},
      "smooth": false
    },
    "physics": {
      "barnesHut": {
        "gravitationalConstant": -30000,
        "centralGravity": 0.3,
        "springLength": 100
      },
      "minVelocity": 0.75
    }
  }
''')

import tempfile

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    net.save_graph(tmp_file.name)
    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html_content = f.read()

components.html(html_content, height=750, scrolling=True)
