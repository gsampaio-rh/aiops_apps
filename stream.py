import streamlit as st
import time
import numpy as np
import plotly.express as px
import requests, json
import os
import random
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


# ---- Apple-Level UI Setup ----
st.set_page_config(page_title="RAG & Intelligent Search Demo", layout="wide")

st.markdown(
    """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
    .header { text-align: center; margin-top: 40px; font-size: 2em; font-weight: bold; }
    .token, .embedding { display: inline-block; margin: 5px; padding: 10px 15px; border-radius: 8px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .token:hover, .embedding:hover { transform: scale(1.05); }
    .token-box { display: inline-block; padding: 5px 8px; margin: 3px; border-radius: 5px; font-size: 1em; font-weight: bold; }
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

# ---- Model Selection & Controls ----
model_options = ["llama3.1:latest", "llama 3.3", "deepseek"]
selected_model = st.sidebar.selectbox("Select Model:", model_options)
temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.2, 0.1)
demo_mode = st.sidebar.radio(
    "Select Demo Mode", ("LLM Only", "RAG + Intelligent Search")
)

# ---- Generate Random Colors for Tokens ----
def random_pastel_color():
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    return f"rgb({r}, {g}, {b})"

# ---- Tokenizer Method ----
def tokenize_text(text):
    """
    Simple tokenizer method that splits text into tokens while preserving special tokens.
    """
    special_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
    tokens = []
    words = text.split()
    for word in words:
        if word in special_tokens:
            tokens.append((word, True))  # Mark special tokens distinctly
        else:
            tokens.append((word, False))
    return tokens


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


st.markdown("### Enter Your Prompts")

system_prompt = st.text_area(
    "System Prompt:", "You are a helpful IT assistant.", height=100
)
user_prompt = st.text_area(
    "User Prompt:",
    "Summarize the last 10 critical errors in the Kubernetes logs.",
    height=100,
)

# ---- Explanation of Prompts ----
with st.expander("See a Sample Prompt Breakdown"):
    st.markdown("### Understanding Prompts")
    st.markdown(
        """
        A **prompt** guides the AI's response and consists of two main parts:

        1. **System Prompt:** Defines the model’s behavior and context.
        *Example:* "You are a helpful IT assistant."

        2. **User Prompt:** Specifies the user’s request.
        *Example:* "Summarize the last 10 critical errors in the Kubernetes logs."

        The AI also uses **special tokens** to structure communication:
        - `<|start_header_id|>` and `<|end_header_id|>` define sections.
        - `<|eot_id|>` marks the end of a message.
        """
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

# ---- Visualization: How LLMs Work (Tokenization, Self-Attention & Embeddings) ----


# st.markdown("---")
if user_prompt:
    # ---- Tokenization and Display ----
    st.markdown("### Tokenized View")

    system_tokens = tokenize_text(system_prompt)
    user_tokens = tokenize_text(user_prompt)
    tokens = system_tokens + user_tokens

    # Render tokens with different colors for visualization
    system_token_html = "".join(
        [
            f"<span class='token-box' style='background-color: {random_pastel_color()};'>{token}</span>"
            for token, _ in system_tokens
        ]
    )
    user_token_html = "".join(
        [
            f"<span class='token-box' style='background-color: {random_pastel_color()};'>{token}</span>"
            for token, _ in user_tokens
        ]
    )

    # Display Tokens in a Container
    st.markdown(
        f"<div class='system-prompt'><b>System Tokens:</b><br>{system_token_html}</div><br>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='user-prompt'><b>User Tokens:</b><br>{user_token_html}</div>",
        unsafe_allow_html=True,
    )

    # ---- Character and Token Counts ----
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Characters", value=len(system_prompt) + len(user_prompt))
    with col2:
        st.metric(label="Tokens", value=len(system_tokens) + len(user_tokens))

    st.markdown("#### Self-Attention Visualization")
    with st.expander("Self-Attention Visualization"):
        attention_matrix = np.random.rand(len(tokens), len(tokens))
        fig_attention = px.imshow(
            attention_matrix,
            labels=dict(x="Tokens", y="Tokens", color="Attention Weight"),
            x=[t[0] for t in tokens],
            y=[t[0] for t in tokens],
            color_continuous_scale="Reds",
        )
        fig_attention.update_layout(title="Simulated Self-Attention Weights")
        st.plotly_chart(fig_attention, use_container_width=True)
        time.sleep(0.5)

    embedding_model = OllamaEmbeddings(model=selected_model)
    with st.spinner("Generating real embeddings..."):
        plain_tokens = [token[0] for token in tokens]  # Extract only the words
        real_embeddings = embedding_model.embed_documents(plain_tokens)

        embeddings_matrix = np.array(real_embeddings)
    st.markdown("#### Embeddings Matrix")
    with st.expander("Embeddings Matrix"):
        st.write(embeddings_matrix)

st.markdown("---")
# ---- Send Request Button & Main Q&A Execution ----
if st.button("Send Request to LLM"):
    if system_prompt and user_prompt:
        if demo_mode == "LLM Only":
            with st.spinner("Generating response using pure LLM..."):
                response_text = get_llm_response(
                    system_prompt, user_prompt, temperature=temperature
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
                    system_prompt, rag_prompt, temperature=temperature
                )
            st.markdown("### RAG Response:")
            st.markdown(f"> {rag_response}")
