# app.py
import streamlit as st

st.set_page_config(
    page_title="The Call-For-Papers Dataset Companion: An Interactive Tool for Research, Trend-, and Network Analysis",
    layout="wide",
)

st.title("Humanities Call-for-Papers Dataset Explorer")
st.markdown("""
### What you can do here:
- Explore topic modeling results across time and fields.
- Browse CfPs via interactive maps, tables, and filters.
- Search and browse a growing collection of extracted research questions from 1995 to the present.
""")
st.warning("For documentation about this project, please check the **About** page")
            
