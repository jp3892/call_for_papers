import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import itertools
import tempfile

# === Paths ===
DATA_PATH = "data/cfps_map_subset.csv"

st.set_page_config(page_title="University-CfPs Network", layout="wide")
st.title("University Co-Occurrence Network")

st.markdown("""
    <div style="border-radius: 12px; background: #fff7e6; padding: 1.5rem; border-left: 6px solid #f4b400;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <p style="margin:0; font-size: 1.1rem;">
    <strong> Universities do not work in isolation. This network map shows a glimpse of how vital it is for universities to speak and connect with each other.
    </strong> </p>
    </p> A co-occurrence means that a single CfP is associated with two distinct universities. For example, a panel organized in conjunction by scholars from different universities.
    <p>
    <p style="margin:0; font-size: 1.05rem;">
    Feel free to play with the filters! I have set the minimun co-occurrence defualt at 10.
    </p>
    <p> 
    <p> </p>
    <p>Edge thickness is correlated with the number of co-occurrences. 
    <p>
</div>
<p>
""", unsafe_allow_html=True) 

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

all_universities = sorted(set(
    u.strip() for sublist in df["universities"].dropna().str.split(";") for u in sublist if u.strip()
))
selected_university = st.sidebar.selectbox("Focus on a Specific University:", ["All"] + all_universities)

min_cooccurrence = st.sidebar.number_input("Minimum Co-Occurrences to Show Edge:", value=10, min_value=2)

# === Apply filters ===
if selected_category != "All":
    df = df[df["categories"].str.contains(fr"\b{selected_category}\b", case=False, na=False)]

df = df[
    (df["date"].dt.year >= selected_year_range[0]) &
    (df["date"].dt.year <= selected_year_range[1]) 
]

# === Build Co-Occurrence Graph ===
G = nx.Graph()

# ... build graph from universities ...
 

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
G.remove_nodes_from(list(nx.isolates(G)))

# === Apply University Filter ===
if selected_university != "All" and selected_university in G.nodes:
    neighbors = list(G.neighbors(selected_university))
    G = nx.Graph()
    original_edges = list(G.edges(data=True)) 
    for neighbor in neighbors:
        weight = next(
            (d["weight"] for u, v, d in original_edges if 
             (u == selected_university and v == neighbor) or 
             (v == selected_university and u == neighbor)),
            None
        )
        if weight:
            G.add_edge(selected_university, neighbor, weight=weight)


# === Remove any isolated nodes again
G.remove_nodes_from(list(nx.isolates(G)))

# === Compute top 10 universities by centrality (excluding unwanted) ===
degree_centrality = nx.degree_centrality(G)
unwanted_universities = {"National University"}

sorted_unis = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
filtered_unis = [(name, score) for name, score in sorted_unis if name not in unwanted_universities]
top_universities = filtered_unis[:10]
top_university_names = {name for name, _ in top_universities}

# === Visualize with Pyvis ===
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#000000")


for node in G.nodes():
    if node == selected_university:
        node_color = "#e41a1c"  # Red for selected university
    elif node in top_university_names:
        node_color = "#ff7f0e"  # Orange for top centrality
    else:
        node_color = "#1f78b4"  # Blue for others
    net.add_node(node, label=node, shape="dot", color=node_color)

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

# === Metrics Display ===
st.subheader("Network Metrics")

st.markdown(f"- **Total Universities:** {G.number_of_nodes()}")
st.markdown(f"- **Total Co-Occurrences (Edges):** {G.number_of_edges()}")

st.markdown("#### Top 10 Most Connected Universities")
for i, (node, centrality) in enumerate(top_universities, 1):
    uni_label = node.replace("uni_", "")
    st.markdown(f"{i}. **{uni_label}** – Centrality: `{centrality:.3f}`")

st.markdown("""
🟥 Selected University  
🟧 Top 10 most central universities  
🟦 All others
""")

# === Render graph ===
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    net.save_graph(tmp_file.name)
    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html_content = f.read()

components.html(html_content, height=750, scrolling=True)
