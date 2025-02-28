import streamlit as st
import time
import numpy as np
import plotly.express as px
import requests, json
import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
import seaborn as sns
import pandas as pd


# ---- Simple Embedding Matrix Visualization ----
def visualize_embedding_matrix(retrieved_docs, vector_store):
    """
    Displays the vector embeddings stored in FAISS as a heatmap.
    Highlights retrieved document vectors separately.
    """
    with st.spinner("📊 Generating embedding matrix visualization..."):
        # Extract stored document vectors from FAISS
        stored_vectors = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)

        # Convert to Pandas DataFrame for visualization
        df_embeddings = pd.DataFrame(stored_vectors)

        # Assign names for readability
        df_embeddings.index = [f"Doc {i+1}" for i in range(len(stored_vectors))]

        # Highlight retrieved documents in a different color
        retrieved_indices = []
        for doc in retrieved_docs:
            try:
                idx = vector_store.index.search(
                    np.array(doc.embedding).reshape(1, -1), 1
                )[1][0][0]
                retrieved_indices.append(idx)
            except:
                continue

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df_embeddings, cmap="coolwarm", cbar=True, ax=ax)

        # Annotate retrieved documents
        for idx in retrieved_indices:
            ax.add_patch(
                plt.Rectangle(
                    (0, idx),
                    len(df_embeddings.columns),
                    1,
                    fill=False,
                    edgecolor="yellow",
                    lw=2,
                )
            )

        plt.title("📊 Vector Store Embedding Matrix")
        st.pyplot(fig)


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

# ---- File Upload for RAG ----

# ---- File Upload for RAG ----
if demo_mode == "RAG + Intelligent Search":
    uploaded_file = st.file_uploader("Upload a Markdown file for RAG:", type=["md"])

    if uploaded_file is not None:
        file_path = f"./data/temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📥 Processing document into vector store..."):
            # Load Markdown file
            loader = TextLoader(file_path)
            documents = loader.load()

            # ✅ **Improved Chunking Strategy**
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # ✅ Smaller chunks
                chunk_overlap=100,  # ✅ Overlap ensures context continuity
            )
            docs = text_splitter.split_documents(documents)

            # ✅ **Store Document Metadata (Title, Section Names)**
            formatted_docs = [
                Document(
                    page_content=chunk.page_content,
                    metadata={"source": uploaded_file.name, "chunk_id": i},
                )
                for i, chunk in enumerate(docs)
            ]

            # ✅ **Use Embeddings for Better Retrieval**
            embedding_model = OllamaEmbeddings(model=selected_model)
            vector_store = FAISS.from_documents(formatted_docs, embedding_model)

            st.success(
                f"✅ Uploaded and processed `{uploaded_file.name}` successfully!"
            )


# ---- Node-Based Retrieval Visualization ----
def visualize_retrieval(query, retrieved_docs):
    G = nx.DiGraph()

    # Add query node
    G.add_node("User Query", color="red", size=1200)

    # Add retrieved documents as nodes
    for idx, doc in enumerate(retrieved_docs):
        G.add_node(f"Doc {idx+1}", color="blue", size=1000)
        G.add_edge("User Query", f"Doc {idx+1}")

    # Plot the graph
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
    plt.figure(figsize=(6, 4))

    # Extract colors & sizes
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=10,
        edge_color="gray",
    )

    # Show visualization
    st.pyplot(plt)


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
            if uploaded_file is not None:
                # ✅ **Modify Retrieval Logic**
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )  # ✅ Retrieve only top 8 most relevant chunks
                results = retriever.get_relevant_documents(user_prompt)
                context = "\n".join([doc.page_content for doc in results])

                st.markdown("#### 🔍 Retrieval Process")
                with st.expander("Node-Based Retrieval Visualization"):
                    visualize_retrieval(user_prompt, results)

                with st.expander("📊 Vector Store Representation"):
                    visualize_embedding_matrix(results, vector_store)

                with st.expander("📄 Retrieved Context"):
                    retrieved_container = (
                        st.container()
                    )  # Use container instead of nested expanders

                    for i, doc in enumerate(results):
                        with retrieved_container:
                            st.markdown(f"**🔹 Document {i+1}**")
                            st.info(doc.page_content[:500])  # Display first 500 chars

                rag_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_prompt}"
                with st.expander("Full Prompt"):
                    st.write(rag_prompt)
                with st.spinner("Generating response using RAG..."):
                    rag_response = get_llm_response(
                        system_prompt, rag_prompt, temperature=temperature
                    )
                st.markdown("### RAG Response:")
                highlighted_response = f"""
                <div style="
                    background-color: #e0f7fa;
                    border-left: 4px solid #00796b;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    font-size: 1.1em;
                    line-height: 1.5;">
                    {rag_response}
                </div>
                """
                st.markdown(highlighted_response, unsafe_allow_html=True)
