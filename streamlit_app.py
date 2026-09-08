import streamlit as st

lab1_page = st.Page("Lab1.py", title="Lab 1 - Build a Streamlit Document ‘Q&A’ app", icon="1️⃣")
lab2_page = st.Page("Lab2.py", title="Lab 2 - Document Summarizer", icon="2️⃣", default=True)
lab3_page = st.Page("Lab3.py", title="Lab 3 - Streaming Chatbot with Memory", icon="3️⃣")

pg = st.navigation([lab1_page, lab2_page, lab3_page])
pg.run()
