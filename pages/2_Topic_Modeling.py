import streamlit as st
import pandas as pd
import json
import os
import re
from little_mallet_wrapper import load_topic_keys, load_topic_distributions
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

st.set_page_config(page_title="Topic Modeling Explorer", layout="wide")
st.title("Topic Modeling Explorer")

# === Step 1: Set base path ===
BASE_PATH = "topic_model_output_slimmed/Topic_modeling_jsons"
CFP_DATA_PATH = "data/cfps_topic_explorer_subset.csv"

# === Cache the full CfP dataset ===
@st.cache_data
def load_all_cfps(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df.reset_index(drop=True)

full_cfp_df = load_all_cfps(CFP_DATA_PATH)

# === Detect available topic models ===
topic_files = [f for f in os.listdir(BASE_PATH) if f.startswith("topics_") and f.endswith(".json")]
categories = [f.replace("topics_", "").replace(".json", "") for f in topic_files]

if not categories:
    st.warning("⚠️ No topic files found.")
    st.stop()

# === Step 3: User selects category ===
st.markdown("## Exploration Mode")
mode = st.radio("Choose how to explore:", ["By Field", "By Topic"])
selected_category = None
selected_topic_label = None

if mode == "By Field":
    selected_category = st.selectbox("Choose a field to explore:", sorted(categories))
elif mode == "By Topic":
    topic_labels = set()
    for f in topic_files:
        path = os.path.join(BASE_PATH, f)
        with open(path, "r") as file:
            data = json.load(file)
            topic_labels.update(t["label"] for t in data if t["label"] != "NA")
    selected_topic_label = st.selectbox("Choose a topic label to compare:", sorted(topic_labels))

if selected_category:
    file_path = os.path.join(BASE_PATH, f"topics_{selected_category}.json")
    try:
        with open(file_path, "r") as f:
            topics = json.load(f)
        topic_df = pd.DataFrame(topics)
        st.subheader(f"Topics *{selected_category}*")
        st.dataframe(topic_df)
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {e}")

    selected_topic = st.selectbox("Explore topic details:", [f"{t['topic_id']}: {t['label']}" for t in topics])
    topic_id = int(selected_topic.split(":")[0])
    topic_label = next(t['label'] for t in topics if t['topic_id'] == topic_id)
    top_words = next(t['top_words'] for t in topics if t['topic_id'] == topic_id)

    st.markdown(f"### Topic {topic_id}: *{topic_label}*")
    st.write("Top words:", ", ".join(top_words))

    cfp_df = full_cfp_df.copy()
    st.write("Selected category:", selected_category)
    st.write("Sample categories in data:", cfp_df['categories'].dropna().unique()[:10])

    cfp_df = cfp_df[cfp_df['categories'].str.lower().str.contains(selected_category.lower(), na=False)]


    model_dir = f"topic_model_output_slimmed/cfps20_{selected_category}"
    topic_dist_path = os.path.join(model_dir, "mallet.topic_distributions.20")

    try:
        topic_distributions = load_topic_distributions(topic_dist_path)
    except FileNotFoundError:
        st.error(f"❌ Couldn't find topic distributions at {topic_dist_path}")
        st.stop()

    topic_df_values = pd.DataFrame(topic_distributions)
    topic_df_values.columns = [f"Topic {i}" for i in range(len(topic_df_values.columns))]
    st.write("cfp_df rows:", len(cfp_df))
    st.write("topic_df_values rows:", len(topic_df_values))

    cfp_df = pd.concat([cfp_df, topic_df_values], axis=1)


    cfp_df['month'] = cfp_df['date'].dt.to_period("M").dt.to_timestamp()
    topic_col = f"Topic {topic_id}"

    def short_title_list(titles):
        return "<br>".join(titles[:5]) + ("<br>..." if len(titles) > 5 else "")

    grouped = (
        cfp_df.groupby('month')
        .agg(
            avg_weight=(topic_col, 'mean'),
            titles=('title', list),
            urls=('url', list),
            ids=('unique_id', list),
            scores=(topic_col, list)
        )
        .reset_index()
    )

    grouped['hover_titles'] = grouped['titles'].apply(short_title_list)
    st.subheader("📈 Topic Frequency Over Time")

    fig = px.line(
        grouped,
        x="month",
        y="avg_weight",
        markers=True,
        title=f"Frequency of Topic {topic_id}: {topic_label}",
        hover_data={"hover_titles": True, "avg_weight": ':.3f'},
    )

    fig.update_traces(
        hovertemplate="<b>Month</b>: %{x|%B %Y}<br>" +
                      "<b>Avg Weight</b>: %{y:.3f}<br>" +
                      "<b>Sample CfPs</b>:<br>%{customdata[0]}"
    )

    st.plotly_chart(fig, use_container_width=True)

    month_options = grouped['month'].dt.strftime('%Y-%m')
    selected_month_str = st.selectbox("📅 Select a month to see full CfPs", month_options)
    selected_ts = pd.to_datetime(selected_month_str)
    month_df = cfp_df[cfp_df['month'] == selected_ts]

    st.markdown(f"### 📄 CfPs for {selected_month_str} ({len(month_df)} total)")

    for _, row in month_df.iterrows():
        st.markdown(f"""
    **[{row['title']}]({row['url']})**  
    Topic Score: `{row[topic_col]:.4f}`  
    ID: `{row['unique_id']}`
    ---""")

    if not month_df.empty:
        download_df = month_df[["title", "url", "unique_id", "date", "categories", topic_col]]
        download_csv = download_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download these CfPs as CSV",
            data=download_csv,
            file_name=f"{selected_category}_topic_{topic_id}_{selected_month_str}.csv",
            mime="text/csv"
        )

    word_weights_path = os.path.join(model_dir, "mallet.word_weights.top50")
    word_freq = {}
    try:
        with open(word_weights_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                t_index, word, weight_str = parts
                if t_index == str(topic_id):
                    try:
                        word_freq[word] = float(weight_str)
                    except ValueError:
                        continue
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        st.subheader("Word Cloud")
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        ax_wc.set_title(f"Topic {topic_id}: {topic_label}")
        st.pyplot(fig_wc)
    except FileNotFoundError:
        st.error(f"Could not find word weights at: {word_weights_path}")

# === Cross-category Topic Exploration ===
if selected_topic_label:
    st.markdown(f"### Cross-field Exploration of Topic: *{selected_topic_label}*")
    matching_entries = []
    for f in topic_files:
        cat = f.replace("topics_", "").replace(".json", "")
        with open(os.path.join(BASE_PATH, f)) as file:
            data = json.load(file)
            for topic in data:
                if topic["label"] == selected_topic_label:
                    matching_entries.append((cat, topic["topic_id"], topic["label"], topic["top_words"]))

    if not matching_entries:
        st.warning("No matching topics found across fields.")
        st.stop()

    for cat, tid, label, words in matching_entries:
        st.markdown(f"#### 📚 Field: *{cat}*, Topic {tid}")
        st.write("Top words:", ", ".join(words))

        subset_df = full_cfp_df[full_cfp_df["categories"].str.contains(fr"\\b{re.escape(cat)}\\b", case=False, na=False)].copy()
        subset_df = subset_df.reset_index(drop=True)

        model_dir = f"topic_model_output_slimmed/cfps20_{cat}"
        dist_path = os.path.join(model_dir, "mallet.topic_distributions.20")

        try:
            topic_distributions = load_topic_distributions(dist_path)
        except FileNotFoundError:
            st.warning(f"⚠️ Couldn't load distributions for {cat}")
            continue

        topic_df = pd.DataFrame(topic_distributions)
        topic_df.columns = [f"Topic {i}" for i in range(len(topic_df.columns))]
        subset_df = pd.concat([subset_df.reset_index(drop=True), topic_df], axis=1)

        subset_df["month"] = subset_df["date"].dt.to_period("M").dt.to_timestamp()
        topic_col = f"Topic {tid}"

        grouped = (
            subset_df.groupby("month")
            .agg(
                avg_weight=(topic_col, "mean"),
                titles=("title", list),
                urls=("url", list),
            )
            .reset_index()
        )
        grouped["hover_titles"] = grouped["titles"].apply(short_title_list)

        fig = px.line(
            grouped,
            x="month",
            y="avg_weight",
            markers=True,
            title=f"Topic {tid} in *{cat}*",
            hover_data={"hover_titles": True, "avg_weight": ':.3f'},
        )

        fig.update_traces(
            hovertemplate="<b>Month</b>: %{x|%B %Y}<br>" +
                          "<b>Avg Weight</b>: %{y:.3f}<br>" +
                          "<b>Sample CfPs</b>:<br>%{customdata[0]}"
        )

        st.plotly_chart(fig, use_container_width=True)

