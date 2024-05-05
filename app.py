from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os 
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-pro")

def get_gemini_responce(question):
    responce=model.generate_content(question)
    return responce.text

st.set_page_config(page_title="Q&A LLM")

st.header("LLM uasing google api")

input=st.text_input("Input: ",key="input")

submit=st.button("Ask the question")


if submit:
    response=get_gemini_responce(input)
    st.subheader("The Responce is")
    st.write(response)

    