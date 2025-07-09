import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# === Paths ===
CFP_PATH = "data/cfps_map_subset.csv"
LOCATIONS_PATH = "data/geocoded_locations.csv"

st.title("Location Explorer")

# === Load data ===
@st.cache_data
def load_data():
    cfp_df = pd.read_csv(CFP_PATH)
    locations_df = pd.read_csv(LOCATIONS_PATH)
    locations_df = locations_df.dropna(subset=["lat", "lon", "location"])
    return cfp_df, locations_df

cfp_df, locations_df = load_data()

# === Filter UI ===
with st.sidebar:
    st.header("Filter CfPs")

    categories = sorted(set(cat.strip() for cats in cfp_df["categories"].dropna() for cat in cats.split(",")))
    selected_category = st.selectbox("Filter by Category (optional):", ["All"] + categories)

    universities = sorted(
    set(uni.strip() for unis in cfp_df["universities"].fillna("") for uni in unis.split(";"))
    )

    selected_university = st.selectbox("Filter by University (optional):", ["All"] + universities)

    sort_by = st.selectbox("Sort CfPs by:", ["view_count", "date", "title"])

# === Create map ===
m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(m)
for _, row in locations_df.iterrows():
    location_name = row["location"]
    lat = row["lat"]
    lon = row["lon"]

    # Find CfPs matching this location
    mask = cfp_df["locations"].fillna("").str.contains(fr"\b{location_name}\b", case=False, na=False)

    if selected_category != "All":
        mask &= cfp_df["categories"].fillna("").str.contains(fr"\b{selected_category}\b", case=False)

    if selected_university != "All":
        mask &= cfp_df["universities"].fillna("").str.contains(fr"\b{selected_university}\b", case=False)

    filtered_cfps = cfp_df[mask].copy()

    if not filtered_cfps.empty:
        if sort_by in filtered_cfps.columns:
            filtered_cfps = filtered_cfps.sort_values(by=sort_by, ascending=False)
        else:
            filtered_cfps = filtered_cfps.sort_values(by="date", ascending=False)

        top_cfps = filtered_cfps.head(10)

        html = f"<b>{location_name}</b><br><ul>"
        for _, r in top_cfps.iterrows():
            title = r['title']
            url = r.get('url', '#')
            html += f"<li><a href='{url}' target='_blank'>{title}</a></li>"
        html += "</ul>"

        iframe = folium.IFrame(html=html, width=300, height=200)
        popup = folium.Popup(iframe, max_width=300)
        folium.Marker(
            [lat, lon],
            popup=popup,
            tooltip=location_name,
            icon=folium.Icon(icon="fa-map-pin", prefix="fa", color="green")
        ).add_to(marker_cluster)


# === Show map ===
st.subheader("Hover over a marker to explore CfPs")
st_data = st_folium(m, width=1000, height=600)
