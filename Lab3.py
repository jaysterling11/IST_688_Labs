import streamlit as st
from openai import OpenAI
import numpy as np

st.title("MY Lab 3 question answering chatbot")

openAI_model = st.sidebar.selectbox(
    "Which Model?",
    ("mini", "regular")
)

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

if 'client' not in st.session_state:
    openai_api_key = st.secrets["openai_api_key"]
    st.session_state.client = OpenAI(api_key=openai_api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your chatbot. How can I assist you today?"
        }
    ]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    stream = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )