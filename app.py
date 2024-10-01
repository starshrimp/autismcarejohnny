__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import logging
import streamlit as st
import openai 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Context:

{context}

---

Answer the question based on the above context: {question}
"""
load_dotenv()

#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = st.secrets["OPENAI_API_KEY"]
# Function to generate response
def generate_response(query_text):
    # Prepare the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results.", [], ""

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response from the model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )
    response_text = llm.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    os.write(1,b'Something was executed.\n')   
    return response_text, sources, prompt

# Streamlit App
st.set_page_config(page_title="ASD Johnny", page_icon="🤖")

# Title
st.title("ASD Support Chatbot")
st.write("This AI-powered chatbot helps support parents of children with mild Autism Spectrum Disorder (ASD).")

# User input
query_text = st.text_input("Ask me anything about ASD:")

# Button to trigger response
if st.button("Send"):
    if query_text:
        with st.spinner("Generating response..."):
            response, sources, prompt = generate_response(query_text)
            st.error(f"**Proof of Concept.**")
            st.info(query_text)
            st.success(f"**AutiBot:** {response}")
            # Display sources if available
            if sources:
                st.write("**Sources:**")
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.write("No sources available.")
            st.info(f"**The following prompt was used:** \n \n{prompt}")


            
    else:
        st.warning("Please enter a question or statement to get started.")

# Footer
st.markdown("---")
st.markdown("Created by Adrian Altermatt & Sarah Meyer")
