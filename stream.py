import streamlit as st
import time
import numpy as np
import plotly.express as px
import requests, json
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

# ---- Setup & Custom CSS for Apple-inspired UI ----
st.set_page_config(
    page_title="RAG & Intelligent Search Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
    .header { text-align: center; margin-top: 40px; }
    .token, .embedding { display: inline-block; margin: 5px; padding: 10px 15px; border-radius: 8px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .token:hover, .embedding:hover { transform: scale(1.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header and Navigation ----
st.markdown(
    "<h1 class='header'>RAG & Intelligent Search Demo App</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='header'>From pure LLM responses to augmented, context-aware answers</p>",
    unsafe_allow_html=True,
)

# ---- Sidebar: Mode Selection ----
demo_mode = st.sidebar.radio(
    "Select Demo Mode", ("LLM Only", "RAG + Intelligent Search")
)


# ---- Function: Call LLM Endpoint with System & User Prompts ----
def get_llm_response(system_prompt, user_prompt, temperature=0.0):
    endpoint = (
        "http://localhost:11434/api/generate"  # Replace with your LLM endpoint URL
    )
    payload = {
        "model": "llama3.1:latest",
        "system": system_prompt,
        "prompt": user_prompt,
        "temperature": temperature,
        "stream": False,
    }
    response = requests.post(
        endpoint, headers={"Content-Type": "application/json"}, data=json.dumps(payload)
    )
    res_json = response.json()
    return res_json.get("response", "No response received")


# ---- Function: RAG Chain (Using LangChain) ----
@st.cache_data(show_spinner=False)
def load_documents(markdown_dir="./markdown_files"):
    documents = []
    for filename in os.listdir(markdown_dir):
        if filename.endswith(".md"):
            loader = TextLoader(os.path.join(markdown_dir, filename))
            documents.extend(loader.load())
    return documents


@st.cache_data(show_spinner=False)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        split_docs.extend(
            [Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks]
        )
    return split_docs


@st.cache_resource(show_spinner=False)
def create_vector_store(split_docs):
    embedding_model = OllamaEmbeddings(model="llama3.1")
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    return vector_store


# Load and prepare documents for RAG only if needed
if demo_mode == "RAG + Intelligent Search":
    with st.spinner("Loading and processing documents..."):
        documents = load_documents()
        split_docs = split_documents(documents)
        vector_store = create_vector_store(split_docs)

# ---- Explanation of Prompts ----
st.markdown("## Understanding Prompts")
st.markdown(
    """
A **prompt** is the input we provide to the language model, and it consists of two key parts:

1. **System Prompt:**  
   This part defines the context, tone, and behavior of the model. It tells the model how it should behave.  
   *Example:* "You are a helpful IT operations assistant. Provide concise and factual recommendations based on system logs."

2. **User Prompt:**  
   This is the actual question or task provided by the user. It specifies what information or action is required.  
   *Example:* "Summarize the last 10 critical errors in the Kubernetes logs and suggest potential fixes."

Together, these prompts ensure the model’s response is both context-aware and directly relevant to your query.
"""
)

with st.expander("See a Sample Prompt Breakdown"):
    st.markdown("### Sample Prompt Breakdown")
    sample_system = "You are a helpful IT operations assistant. Provide concise and factual recommendations based on system logs."
    sample_user = "Summarize the last 10 critical errors in the Kubernetes logs and suggest potential fixes."
    st.markdown(
        f"""
    <div style="background-color: #f0f0f5; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
        <strong>System Prompt:</strong><br>
        <em>{sample_system}</em>
    </div>
    <div style="background-color: #f0f0f5; padding: 15px; border-radius: 8px;">
        <strong>User Prompt:</strong><br>
        <em>{sample_user}</em>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    **How It Works:**
    - The **system prompt** sets the stage by guiding the model's behavior.
    - The **user prompt** provides the direct question.
    
    Combined, they ensure the model understands both what to do and what is being asked.
    """
    )

# ---- Main Interface: System and User Prompts ----
st.markdown("### Enter your Prompts")
default_system_prompt = "You are a helpful IT operations assistant. Provide concise and factual recommendations based on system logs."
default_user_prompt = "Summarize the last 10 critical errors in the Kubernetes logs and suggest potential fixes."

system_prompt = st.text_area("System Prompt:", default_system_prompt, height=100)
user_prompt = st.text_input("User Prompt:", default_user_prompt)


# ---- Visualization: How LLMs Work (Tokenization, Self-Attention & Embeddings) ----
st.markdown("---")
if user_prompt:
    st.markdown("#### Tokenization")
    tokens = user_prompt.split()
    token_html = " ".join(
        [f"<span class='token' title='Token: {t}'>{t}</span>" for t in tokens]
    )
    st.markdown(token_html, unsafe_allow_html=True)
    time.sleep(0.5)

    st.markdown("#### Self-Attention Visualization")
    attention_matrix = np.random.rand(len(tokens), len(tokens))
    fig_attention = px.imshow(
        attention_matrix,
        labels=dict(x="Tokens", y="Tokens", color="Attention Weight"),
        x=tokens,
        y=tokens,
        color_continuous_scale="Reds",
    )
    fig_attention.update_layout(title="Simulated Self-Attention Weights")
    st.plotly_chart(fig_attention, use_container_width=True)
    time.sleep(0.5)

    embedding_model = OllamaEmbeddings(model="llama3.1")
    with st.spinner("Generating real embeddings..."):
        real_embeddings = embedding_model.embed_documents(tokens)
        embeddings_matrix = np.array(real_embeddings)
    st.markdown("#### Embeddings Matrix")
    st.write(embeddings_matrix)

st.markdown("---")

# ---- Send Request Button & Main Q&A Execution ----
if st.button("Send Request to LLM"):
    if system_prompt and user_prompt:
        if demo_mode == "LLM Only":
            with st.spinner("Generating response using pure LLM..."):
                response_text = get_llm_response(
                    system_prompt, user_prompt, temperature=0.2
                )
            st.markdown("### LLM Response:")
            highlighted_response = f"""
            <div style="
                background-color: #e0f7fa;
                border-left: 4px solid #00796b;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                font-size: 1.1em;
                line-height: 1.5;">
                {response_text}
            </div>
            """
            st.markdown(highlighted_response, unsafe_allow_html=True)
        elif demo_mode == "RAG + Intelligent Search":
            st.markdown("### Retrieving Contextual Data...")
            retriever = vector_store.as_retriever()
            results = retriever.get_relevant_documents(user_prompt)
            context = "\n".join([doc.page_content for doc in results[:3]])
            st.markdown("#### Retrieved Context:")
            st.markdown(context)
            rag_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_prompt}"
            with st.spinner("Generating response using RAG..."):
                rag_response = get_llm_response(
                    system_prompt, rag_prompt, temperature=0.2
                )
            st.markdown("### RAG Response:")
            st.markdown(f"> {rag_response}")
