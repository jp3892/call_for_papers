

import streamlit as st
import pandas as pd
import json
import os
import re
from little_mallet_wrapper import load_topic_keys, load_topic_distributions
import matplotlib.pyplot as plt

st.title("Topic Modeling Explorer")


# === Step 1: Set base path ===
BASE_PATH = "/Users/juanpabloalbornoz/Documents/CfPs Dataset and Documentation/Python notebooks/Website/Topic_modeling" 
CFP_DATA_PATH = "/Users/juanpabloalbornoz/Documents/CfPs Dataset and Documentation/Python notebooks/cfps_deduplicated_by_title_and_content.csv"

# === Detect available topic models ===

# === Step 2: List available topic files ===
topic_files = [f for f in os.listdir(BASE_PATH) if f.startswith("topics_") and f.endswith(".json")]
topic_list = {
    "Memory and Trauma", "Gender Studies"
    }

# Extract categories from filenames like: topics_theory.json → "theory"
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
    # === Step 4: Load and show topics ===
    file_path = os.path.join(BASE_PATH, f"topics_{selected_category}.json")

    try:
        with open(file_path, "r") as f:
            topics = json.load(f)
        topic_df = pd.DataFrame(topics)
        
        st.subheader(f"Topics *{selected_category}*")
        st.dataframe(topic_df)
        


    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {e}")


    # === User picks topic ===
    selected_topic = st.selectbox(
        "Explore topic details:",
        [f"{t['topic_id']}: {t['label']}" for t in topics]
    )
    topic_id = int(selected_topic.split(":")[0])
    topic_label = next(t['label'] for t in topics if t['topic_id'] == topic_id)
    top_words = next(t['top_words'] for t in topics if t['topic_id'] == topic_id)

    st.markdown(f"### Topic {topic_id}: *{topic_label}*")
    st.write("Top words:", ", ".join(top_words))

    # === Load CfP data ===
    cfp_df = pd.read_csv(CFP_DATA_PATH)
    cfp_df['date'] = pd.to_datetime(cfp_df['date'], errors='coerce')
    cfp_df = cfp_df.dropna(subset=['date'])
    cfp_df = cfp_df[cfp_df['categories'].str.contains(fr'\b{re.escape(selected_category)}\b', case=False, na=False)]
    cfp_df = cfp_df.reset_index(drop=True)

    # === Load topic distributions ===
    # === Load topic distributions for the selected category ===
    model_dir = f"/Users/juanpabloalbornoz/Documents/CfPs Dataset and Documentation/DATA/topic_model_output_2/cfps20_{selected_category}"
    topic_dist_path = os.path.join(model_dir, "mallet.topic_distributions.20")

    try:
        topic_distributions = load_topic_distributions(topic_dist_path)
    except FileNotFoundError:
        st.error(f"❌ Couldn't find topic distributions at {topic_dist_path}")
        st.stop()


    topic_df_values = pd.DataFrame(topic_distributions)
    topic_df_values.columns = [f"Topic {i}" for i in range(len(topic_df_values.columns))]

    # === Merge with CfP metadata ===
    cfp_df = pd.concat([cfp_df, topic_df_values], axis=1)

    # === Create time series ===
    import plotly.express as px

    # === Create a 'month' column if not already created ===
    cfp_df['month'] = cfp_df['date'].dt.to_period("M").dt.to_timestamp()

    # Get dynamic topic column name
    topic_col = f"Topic {topic_id}"

    # === Group for interactive plot ===
    hover_limit = 5

    def short_title_list(titles):
        return "<br>".join(titles[:hover_limit]) + ("<br>..." if len(titles) > hover_limit else "")

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

    # === Plot interactive line chart ===
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

    # === Dropdown to explore CfPs for a selected month ===
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
    # Prepare download CSV
    if not month_df.empty:
        download_df = month_df[["title", "url", "unique_id", "date", "categories", topic_col]]
        download_csv = download_df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Download these CfPs as CSV",
            data=download_csv,
            file_name=f"{selected_category}_topic_{topic_id}_{selected_month_str}.csv",
            mime="text/csv"
        )


    from wordcloud import WordCloud

    # === Optional UI toggle ===
    #if st.checkbox("Show word cloud for this topic"):

    # Dynamic path to word weights
    word_weights_path = os.path.join(model_dir, "mallet.word_weights.20")

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

            # Generate and display word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white'
        ).generate_from_frequencies(word_freq)

        st.subheader("Word Cloud")
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        ax_wc.set_title(f"Topic {topic_id}: {topic_label}")
        st.pyplot(fig_wc)

    except FileNotFoundError:
        st.error(f"Could not find word weights at: {word_weights_path}")

    import re
    import string

    st.subheader("Top Documents for This Topic")

    num_results = st.slider("Number of top documents to show:", min_value=1, max_value=20, value=5)

    # === Step 1: Reload and clean CfP content ===
    cfp_df_filtered = cfp_df[cfp_df['categories'].str.contains(fr'\b{re.escape(selected_category)}\b', case=False, na=False)].copy()
    cfp_df_filtered = cfp_df_filtered.reset_index(drop=True)

    import nltk
    from nltk.corpus import stopwords
    import little_mallet_wrapper

    custom_stopwords = {
        "call", "paper", "papers", "conference", "symposium", "proposal", "submit",
        "submission", "panel", "participants", "topics", "deadline", "french", "paris",
        "press", "mso", "université", "france", "font", "univ", "eds", "york", "routledge",
        "family", "cambridge", "oxford", "london", "sorbonne", "francophone", "association",
        "annual", "january", "session", "college", "samla", "theme", "invites", "mail",
        "aspect", "must", "author", "articles", "publication", "published", "name", "review",
        "submitted", "mla", "title", "format", "authors", "issue", "pages", "page", "essays",
        "collection", "book", "volume", "edited", "series", "chapters", "editors", "chapter",
        "essay", "due", "editor", "books", "contributors", "contributor", "nemla", "www",
        "convention", "seminar", "acla", "present", "march", "september", "https", "roundtable",
        "cfplist", "cfp", "http", "mailing", "received", "athttp", "calls", "listcfp", "est",
        "edufull", "jennifer", "eduor", "higginbotham", "higginbj", "edt", "edumore", "tue",
        "mon", "pamla", "categories", "organization", "november", "gmail", "collections",
        "identityfilm", "org", "keynote", "speaker", "speakers", "cea", "web", "presentations",
        "areas", "los", "welcomes", "las", "que", "del", "departments", "proposals",
        "submissions", "panels", "cfp", "write", "www", "fri", "wed", "feb", "thu", "jan",
        "nov", "sat", "sun", "oct", "erika", "com", "edu/cfp/or", "lin", "fax", "mar", "elin",
        "state", "language", "modern", "fiction", "writers", "boston", "held", "brief",
        "northeast", "works", "requirements", "interested", "creative", "student",
        "composition", "special", "presenters", "san", "topic", "rmmla", "join", "technical",
        "via", "see", "louisiana", "general", "project", "friday", "accepted", "organising",
        "people", "net", "receive", "following", "global", "details", "issues", "explore",
        "chairs", "alternative", "together", "full", "peer", "reviews", "reviewed", "style",
        "guidelines", "chair", "southwestpca", "graduate", "albuquerque", "pca/acacreative",
        "contact", "registration", "hotel", "swpaca", "regency", "hyatt", "encouraged",
        "december", "etc", "swtxpca", "welcome", "please", "send", "abstracts", "address",
        "may", "april", "february", "june", "july", "october"
    }

    words_to_remove = set(stopwords.words("english")).union(custom_stopwords)


    # Clean text (same as used during training)
    def clean_text(text):
        text = str(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\d+", "", text)
        text = text.lower()
        tokens = text.split()
        return " ".join([t for t in tokens if t not in words_to_remove and len(t) > 2])

    training_data = [clean_text(text) for text in cfp_df_filtered['content']]
    original_texts = cfp_df_filtered['content'].tolist()
    cfp_dict = dict(zip(training_data, original_texts))

    # === Step 2: Load topic model outputs again ===
    topics_path = os.path.join(model_dir, f"mallet.topic_keys.20")
    dists_path = os.path.join(model_dir, f"mallet.topic_distributions.20")
    topics = little_mallet_wrapper.load_topic_keys(topics_path)
    topic_distributions = little_mallet_wrapper.load_topic_distributions(dists_path)


    # === Step 3: Get top documents ===
    top_docs = little_mallet_wrapper.get_top_docs(training_data, topic_distributions, topic_id, n=num_results)

    for prob, doc in top_docs:
        idx = training_data.index(doc)
        row = cfp_df_filtered.iloc[idx]

        title = row.get("title", "Untitled")
        unique_id = row.get("unique_id", "Unknown ID")
        content = row.get("content", "")
        categories = row.get("categories", "None")

        # Bold top words
        for word in topics[topic_id]:
            content = re.sub(rf"\b({re.escape(word)})\b", r"**\1**", content, flags=re.IGNORECASE)

        with st.expander(f"📌 {title} — Score: {round(prob, 4)}"):
            st.markdown(f"""
    **ID:** `{unique_id}`  
    **Categories:** {categories}  
    **Topic:** *{topic_label}*

    {content}
    """)
import plotly.express as px

# === Cache the full CfP dataset ===
@st.cache_data
def load_all_cfps(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df.reset_index(drop=True)

full_cfp_df = load_all_cfps(CFP_DATA_PATH)

# === Cross-category Topic Exploration ===
if selected_topic_label:

    st.markdown(f"### Cross-field Exploration of Topic: *{selected_topic_label}*")

    matching_entries = []

    # Find all topic instances across fields
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

    # === Create plots and metadata for each field ===
    for cat, tid, label, words in matching_entries:
        st.markdown(f"#### 📚 Field: *{cat}*, Topic {tid}")
        st.write("Top words:", ", ".join(words))

        # Filter CfPs for this category
        subset_df = full_cfp_df[full_cfp_df["categories"].str.contains(fr"\b{re.escape(cat)}\b", case=False, na=False)].copy()
        subset_df = subset_df.reset_index(drop=True)

        # Load topic distribution
        model_dir = f"/Users/juanpabloalbornoz/Documents/CfPs Dataset and Documentation/DATA/topic_model_output_2/cfps20_{cat}"
        dist_path = os.path.join(model_dir, "mallet.topic_distributions.20")

        try:
            topic_distributions = load_topic_distributions(dist_path)
        except FileNotFoundError:
            st.warning(f"⚠️ Couldn't load distributions for {cat}")
            continue

        topic_df = pd.DataFrame(topic_distributions)
        topic_df.columns = [f"Topic {i}" for i in range(len(topic_df.columns))]

        # Add to dataframe
        subset_df = pd.concat([subset_df.reset_index(drop=True), topic_df], axis=1)

        # Generate monthly average
        subset_df["month"] = subset_df["date"].dt.to_period("M").dt.to_timestamp()
        topic_col = f"Topic {tid}"

        # Prepare hover data
        def short_title_list(titles):
            return "<br>".join(titles[:5]) + ("<br>..." if len(titles) > 5 else "")

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

        # Plot
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
