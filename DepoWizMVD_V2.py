import streamlit as st
from DepoWizTOC_V4 import app1
from DepoWizQNA_V3 import app2

# Create a dictionary with page names as keys and corresponding functions as values
pages = {
    'Generate TOC': app1,
    'Question Answering': app2
}

# Add a radio button in the sidebar to switch between pages
page_selection = st.sidebar.radio('Select a page', list(pages.keys()))

# Call the selected page function to display the page content
pages[page_selection]()
