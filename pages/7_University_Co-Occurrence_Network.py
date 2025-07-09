import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import itertools
import tempfile

# === Paths ===
DATA_PATH = "data/cfps_map_subset.csv"

st.set_page_config(page_title="University-CfPs Network", layout="wide")
st.title("University Co-Occurrence Network")

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

min_cooccurrence = st.sidebar.number_input("Minimum Co-Occurrences to Show Edge:", value=10, min_value=2)

# === Apply filters ===
if selected_category != "All":
    df = df[df["categories"].str.contains(fr"\b{selected_category}\b", case=False, na=False)]

df = df[
    (df["date"].dt.year >= selected_year_range[0]) &
    (df["date"].dt.year <= selected_year_range[1]) &
    (df["view_count"].fillna(0) >= view_threshold)
]

# === Build Co-Occurrence Graph ===
G = nx.Graph()

for unis in df["universities"].fillna("").str.split(";"):
    cleaned_unis = sorted(set(u.strip() for u in unis if u.strip() and u.strip().lower() != "nan"))
    for u1, u2 in itertools.combinations(cleaned_unis, 2):
        if G.has_edge(u1, u2):
            G[u1][u2]["weight"] += 1
        else:
            G.add_edge(u1, u2, weight=1)

# === Filter edges by co-occurrence threshold ===
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_cooccurrence]
G.remove_edges_from(edges_to_remove)

# Remove isolated nodes
dangling_nodes = list(nx.isolates(G))
G.remove_nodes_from(dangling_nodes)

# === Visualize with Pyvis ===
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#000000")

for node in G.nodes():
    net.add_node(node, label=node, shape="dot", color="#1f78b4")

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d["weight"], title=f"Co-Occurrences: {d['weight']}")

net.set_options('''
  var options = {
    "nodes": {
      "font": {"size": 14},
      "scaling": {"min": 10, "max": 30}
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

st.subheader("Network Metrics")

# Basic stats
st.markdown(f"- **Total Universities:** {G.number_of_nodes()}")
st.markdown(f"- **Total Co-Occurrences (Edges):** {G.number_of_edges()}")

# Degree centrality
degree_centrality = nx.degree_centrality(G)
top_universities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

st.markdown("#### Top 10 Most Connected Universities")
for i, (node, centrality) in enumerate(top_universities, 1):
    uni_label = node.replace("uni_", "")
    st.markdown(f"{i}. **{uni_label}** – Centrality: `{centrality:.3f}`")


with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    net.save_graph(tmp_file.name)
    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html_content = f.read()

components.html(html_content, height=750, scrolling=True)

