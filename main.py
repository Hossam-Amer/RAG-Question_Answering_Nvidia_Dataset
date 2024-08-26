import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
st.header("Nvidia Q/A:",divider='gray')

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke(question)
    st.header("Answer")
    st.write(response["result"])