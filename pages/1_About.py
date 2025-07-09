import streamlit as st

st.title("The Call-For-Papers Dataset Companion")
st.markdown("""

<div style="border-radius: 12px; background: #fff7e6; padding: 1.5rem; border-left: 6px solid #f4b400;
             box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <p style="margin:0; font-size: 1.1rem;">
    <strong>A call-for-papers is often the first building block of new ideas,</strong>
    connecting scholars around the world that are thinking about critical questions of the present.
    </p>
    <br>
    <p style="margin:0; font-size: 1.05rem;">
    This app invites users to explore a large Call-For-Papers dataset in the humanities, 
    allowing them to discover and analyze new or old trends, connections, and research questions.
    </p>
</div>

</p>
For more information about this dataset, data cleaning, curation, limitations, and applications, please refer to the following companion paper:
            
["The Overlooked Genre of Calls-for-Papers: New Grounds to Study Shifts and Connections in Academic Discourse."](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.278)

Or feel free to contact me at

ja827@cornell.edu

[Juan-Pablo Albornoz](https://english.cornell.edu/juan-pablo-albornoz-0), Cornell University
            
**Special thanks to** 

- The whole team and cohort at the [Summer Graduate Fellowship in Digital Humanities at Cornell University](https://www.library.cornell.edu/about/staff/central-departments/digital-scholarship/colab-programs/summer-dh/)
- Lindsay Thomas, Cornell University
- [Journal of Open Humanities Data](https://openhumanitiesdata.metajnl.com/)
- The folks at the [University of Pennsylvania CfP Website](https://call-for-papers.sas.upenn.edu/)



            
### For documentation about the pages in this app, see below. 
 
This app uses slimmed versions of the main dataset. To access the full dataset, please refer to the [Harvard Dataverse Repository.](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DFSMBN) 

This app was created for research purposes only.
I used ChatGpT and Copilot to help correct and write some of the code used in this app.

### Top Entries
- I extracted the view count of the CfPs in the University of Pennsylvania CfP Website. View count extraction was performed in June, 2025. This will be updated with some regularity. 

### Research Questions Collection
- I used Gemini's Flash 2.0 LLM to extract research questions from the dataset. 
- The exact prompt I used was: "Read the data and extract any explicit research questions you find from each entry. Leave blank if you do not find any research questions."
- The LLM picked both direct and indirect questions.
- Important note: LLMs may hallucinate. You can check the URL of the original CfP to corroborate accuracy. 
- Search function was made by calculating similarity scores through embeddings via [SBERT](https://sbert.net/).
            
### Topic modeling 
- The topic modeling was trained using [Little_Mallet_Wrapper](https://github.com/maria-antoniak/little-mallet-wrapper), developed by Maria Antoniak. The base code was written following Melanie Walsh's [topic modeling tutorial](https://melaniewalsh.github.io/Intro-Cultural-Analytics/welcome.html).
- The topic modeling page allows the user to browse the results by field and topic.
    - **Field:** every Call-for-Paper from the [University of Pennsylvania Call-for-Papers Website](https://call-for-papers.sas.upenn.edu/) is tagged with one or more category tags. The tagging is made by the indiviudal CfP uploader, who selects the tags from a pre-set list of 41 available category tags. Each category tag is here taken as a field. 
    - **Topic:** The topic labelling was done by me, after analyzing the results for each field. 
- **Topic numbers:**
    - With the whole dataset, I experimented with 10, 15, 20, 35, and 41 topics. I found the 35-topic list to present the most robust results. 
    - I trained each individual field to produce 20 topics. I used a common stop-word list plus a custom stop-word list. The custom stop-word list was made to have the model avoid words that are specific to the call-for-paper genre but that I did not see fit to include, such as "call", "conference", or "panel".
- **CfP Discourse Topic:** 
    - After analyzing the results, I decided to label one of the topics "CfP Discourse". This is a topic that tries to reflect the typical discourse, rather than the content, present in the CfPs.
- **NA Topics:**
    - Because I made the decision to train all fields with 20 topics, some of the results do not give a clear, useful topic. I therefore labelled these NA.



### Map
- I used NER to extract university names from the whole dataset. Locations were then extracted from university info and organizer emails using [GeoPy](https://geopy.readthedocs.io/en/stable/)
- A location means that the CfP includes a university name in that location.            
### Universities
- This page simply aggregates the CfP counts for individual unviersity and allows any researcher to browse CfPs by university.
### Journals
- Extracting journal information from the CfP database was a half-success, as Regex and NER produced many false positives and negatives. The best experiment resulted from using [Open Alex's API](https://openalex.org/) to source journal names. This limitation must be taken into account. 
### Associations
- This page allows the user to browse CfPs by Association (e.g. MLA)
### Network
- I used [Pyvis](https://pyvis.readthedocs.io/en/latest/) to create an interactive ntewrok visualization.
- Each node is a university and the edges represent the number of call-for-papers in which they are associated. For a better experience, I recommend filtering by category, university, or using a high co-ocurrence number.

### Limitations and Possible Enhancements
- This list is by no means exhaustive and I absolutely welcome further feedback.
    - **Topic modeling:** one big mistake was that I did not filter out the field title during training. This mistakes implies that, for example, in the category of "Victorian" the model detects "victorian" as an important word in many topics. However, given the time cit takes to manually label all the topics, I did not retrain.
    - 20 might not be the best topic number for each category. A better training would try to figure out the optimal number of topics for each category.
    - BERTopic might be worth a try
    - All information extracted through NER or regex might be incomplete, or false positives might be present. 
    - A search engine that would allow researchers to look for specific topics could represent a nice addition to this app. 

### External Links:
- Python packages documentation
    - [Pandas](https://pandas.pydata.org/docs/)
    - [Matplotlib](https://matplotlib.org/stable/index.html)
    - [Plotly](https://plotly.com/python/)
    - [NLTK](https://www.nltk.org/)
    - [Little Mallet Wrapper](https://github.com/maria-antoniak/little-mallet-wrapper)
    - [SciPy](https://docs.scipy.org/doc/scipy/)
    - [Folium](https://python-visualization.github.io/folium/latest/)
    - [Pyvis](https://pyvis.readthedocs.io/en/latest/index.html)
    - [Sentence-Transformers](https://sbert.net/)
    - [Scikit-learn](https://scikit-learn.org/stable/)
    - [Altair](https://altair-viz.github.io/)
 - Python Scripts, Topic Modeling Full data, Full Dataset:
    - [Harvard Dataverse Repository.](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DFSMBN)**Note: not yet updated**
    - [GitHub Repo](https://github.com/jp3892/call_for_papers)
    - [University of Pennsylvania CfP Website](https://call-for-papers.sas.upenn.edu/)

**Copyright notice** I do not own any of the content extracted from the UPenn CfP Website. If you have any queries, or want something taken down, don't hesitate to contact me. This app is designed for research purposes. 

Juan-Pablo Albornoz, Cornell University
    
""", unsafe_allow_html=True)
