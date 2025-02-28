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


# ---- Special Token Highlighting ----
def highlight_prompt(prompt):
    special_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
    for token in special_tokens:
        prompt = prompt.replace(
            token, f"<span style='color:blue; font-weight:bold;'>{token}</span>"
        )
    return prompt


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


# ---- Apple-Level UI Setup ----
st.set_page_config(page_title="RAG & Intelligent Search Demo", layout="wide")

st.markdown(
    """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
    .header { text-align: center; margin-top: 40px; font-size: 2em; font-weight: bold; }
    .token, .embedding { display: inline-block; margin: 5px; padding: 10px 15px; border-radius: 8px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .token:hover, .embedding:hover { transform: scale(1.05); }
    .split-container { display: flex; justify-content: space-between; }
    .system-prompt { background: #e3f2fd; padding: 10px; border-radius: 5px; }
    .user-prompt { background: #fbe9e7; padding: 10px; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown(
    "<h1 class='header'>RAG & Intelligent Search Demo</h1>", unsafe_allow_html=True
)

# ---- Sidebar: Mode Selection ----
demo_mode = st.sidebar.radio(
    "Select Demo Mode", ("LLM Only", "RAG + Intelligent Search")
)

# Load and prepare documents for RAG only if needed
if demo_mode == "RAG + Intelligent Search":
    with st.spinner("Loading and processing documents..."):
        documents = load_documents()
        split_docs = split_documents(documents)
        vector_store = create_vector_store(split_docs)

# ---- Model Selection & Controls ----
model_options = ["Llama 3.1", "Llama 3.3", "GPT-4"]
selected_model = st.sidebar.selectbox("Select Model:", model_options)
temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.2, 0.1)
mode = st.sidebar.radio(
    "Function Calling Mode:", ["Zero-Shot", "User Message", "System Message"]
)

st.markdown("### Enter Your Prompts")
system_prompt = st.text_area(
    "System Prompt:", "You are a helpful IT assistant.", height=100
)
user_prompt = st.text_area(
    "User Prompt:",
    "Summarize the last 10 critical errors in the Kubernetes logs.",
    height=100,
)

# Display explanation with special tokens
st.markdown("### Prompt Breakdown")
special_prompt_display = (
    "<div class='system-prompt'><b>System Prompt:</b><br>"
    "<|start_header_id|>system<|end_header_id|>\n"
    + system_prompt
    + "\n<|eot_id|></div>"
    "<br><div class='user-prompt'><b>User Prompt:</b><br>"
    "<|start_header_id|>user<|end_header_id|>\n" + user_prompt + "\n<|eot_id|></div>"
)
st.markdown(
    f"<div>{highlight_prompt(special_prompt_display)}</div>", unsafe_allow_html=True
)


# # ---- Explanation of Prompts ----
# with st.expander("See a Sample Prompt Breakdown"):
#     st.markdown("## Understanding Prompts")
#     st.markdown(
#         """
#     A **prompt** is the input we provide to the language model, and it consists of two key parts:

#     1. **System Prompt:**
#     This part defines the context, tone, and behavior of the model. It tells the model how it should behave.
#     *Example:* "You are a helpful IT operations assistant. Provide concise and factual recommendations based on system logs."

#     2. **User Prompt:**
#     This is the actual question or task provided by the user. It specifies what information or action is required.
#     *Example:* "Summarize the last 10 critical errors in the Kubernetes logs and suggest potential fixes."

#     Together, these prompts ensure the model’s response is both context-aware and directly relevant to your query.
#     """
#     )
#     st.markdown("### Sample Prompt Breakdown")
#     sample_system = "You are a helpful IT operations assistant. Provide concise and factual recommendations based on system logs."
#     sample_user = "Summarize the last 10 critical errors in the Kubernetes logs and suggest potential fixes."
#     st.markdown(
#         f"""
#     <div style="background-color: #f0f0f5; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
#         <strong>System Prompt:</strong><br>
#         <em>{sample_system}</em>
#     </div>
#     <div style="background-color: #f0f0f5; padding: 15px; border-radius: 8px;">
#         <strong>User Prompt:</strong><br>
#         <em>{sample_user}</em>
#     </div>
#     """,
#         unsafe_allow_html=True,
#     )

# # Highlight special tokens
# st.markdown("### Tokenized Prompt Preview")
# st.markdown(f"<div>{highlight_prompt(user_prompt)}</div>", unsafe_allow_html=True)


# # ---- Visualization: How LLMs Work (Tokenization, Self-Attention & Embeddings) ----
# st.markdown("---")
# if user_prompt:
#     st.markdown("#### Tokenization")
#     tokens = user_prompt.split()
#     token_html = " ".join(
#         [f"<span class='token' title='Token: {t}'>{t}</span>" for t in tokens]
#     )
#     st.markdown(token_html, unsafe_allow_html=True)
#     time.sleep(0.5)

#     st.markdown("#### Self-Attention Visualization")
#     attention_matrix = np.random.rand(len(tokens), len(tokens))
#     fig_attention = px.imshow(
#         attention_matrix,
#         labels=dict(x="Tokens", y="Tokens", color="Attention Weight"),
#         x=tokens,
#         y=tokens,
#         color_continuous_scale="Reds",
#     )
#     fig_attention.update_layout(title="Simulated Self-Attention Weights")
#     st.plotly_chart(fig_attention, use_container_width=True)
#     time.sleep(0.5)

#     embedding_model = OllamaEmbeddings(model=selected_Model)
#     with st.spinner("Generating real embeddings..."):
#         real_embeddings = embedding_model.embed_documents(tokens)
#         embeddings_matrix = np.array(real_embeddings)
#     st.markdown("#### Embeddings Matrix")
#     st.write(embeddings_matrix)

# st.markdown("---")

# # ---- Send Request Button & Main Q&A Execution ----
# if st.button("Send Request to LLM"):
#     if system_prompt and user_prompt:
#         if demo_mode == "LLM Only":
#             with st.spinner("Generating response using pure LLM..."):
#                 response_text = get_llm_response(
#                     system_prompt, user_prompt, temperature=0.2
#                 )
#             st.markdown("### LLM Response:")
#             highlighted_response = f"""
#             <div style="
#                 background-color: #e0f7fa;
#                 border-left: 4px solid #00796b;
#                 padding: 20px;
#                 margin: 20px 0;
#                 border-radius: 8px;
#                 font-size: 1.1em;
#                 line-height: 1.5;">
#                 {response_text}
#             </div>
#             """
#             st.markdown(highlighted_response, unsafe_allow_html=True)
#         elif demo_mode == "RAG + Intelligent Search":
#             st.markdown("### Retrieving Contextual Data...")
#             retriever = vector_store.as_retriever()
#             results = retriever.get_relevant_documents(user_prompt)
#             context = "\n".join([doc.page_content for doc in results[:3]])
#             st.markdown("#### Retrieved Context:")
#             st.markdown(context)
#             rag_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_prompt}"
#             with st.spinner("Generating response using RAG..."):
#                 rag_response = get_llm_response(
#                     system_prompt, rag_prompt, temperature=0.2
#                 )
#             st.markdown("### RAG Response:")
#             st.markdown(f"> {rag_response}")
