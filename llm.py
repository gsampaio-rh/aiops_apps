import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import requests

# Configure the page with a minimal, clean layout
st.set_page_config(
    page_title="LLM Demo with Ollama", layout="wide", initial_sidebar_state="collapsed"
)

# Minimal custom CSS for subtle styling (Apple-inspired)
st.markdown(
    """
    <style>
        /* Use system fonts for a clean, native look */
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #ffffff;
            color: #333;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .subheader {
            font-size: 1.25rem;
            margin-bottom: 20px;
        }
        /* Minimal info boxes */
        .stAlert {
            border: none;
            background-color: #f8f8f8;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page header
st.markdown(
    "<div class='header-title'>How Does the LLM Work? (Powered by Ollama)</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subheader'>Experience the inner workings of an LLM in four elegant steps.</div>",
    unsafe_allow_html=True,
)


# Helper function for section headers (Apple-style simplicity)
def section_header(title):
    st.markdown(f"### {title}")


# Function to get embeddings from Ollama
def get_ollama_embeddings(tokens):
    url = "http://localhost:11434/api/embed"
    headers = {"Content-Type": "application/json"}
    data = {"model": "nomic-embed-text", "input": tokens}
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get("embeddings", [])
        else:
            return []
    except Exception as e:
        return []


# Function to query Ollama's API for a real response
def ollama_query(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": "llama3.1", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response found.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"


# Main simulation function with incremental, intuitive steps
def simulate_llm_process(query):
    # Step 1: Tokenization
    section_header("Step 1: Tokenization")
    st.info("Splitting your input into tokens (words) for processing.")
    with st.spinner("Tokenizing..."):
        time.sleep(1)  # Simulate processing delay
        tokens = query.split()
    st.write("**Tokens:**", tokens)

    # Step 2: Embedding Generation
    section_header("Step 2: Embedding Generation")
    st.info("Converting tokens into numerical embeddings for analysis.")
    with st.spinner("Generating embeddings..."):
        time.sleep(1)  # Simulate processing delay
        raw_embeddings = get_ollama_embeddings(tokens)
        # Convert to numpy array; if empty or shape issues, create dummy data
        if raw_embeddings and isinstance(raw_embeddings, list):
            embeddings = np.array(raw_embeddings)
        else:
            embeddings = np.random.rand(len(tokens), 128)

        # Dimensionality reduction for visualization (using TSNE)
        if len(tokens) > 1:
            tsne = TSNE(
                n_components=2, perplexity=min(30, len(tokens) - 1), random_state=42
            )
            reduced_embeddings = tsne.fit_transform(embeddings)
        else:
            reduced_embeddings = np.zeros((len(tokens), 2))

    with st.expander("Token Embeddings"):
        # Create a clean scatter plot for embeddings visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = sns.color_palette("husl", len(tokens))
        for i, token in enumerate(tokens):
            ax.scatter(
                reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=colors[i], s=100
            )
            ax.text(
                reduced_embeddings[i, 0] + 0.01,
                reduced_embeddings[i, 1],
                token,
                fontsize=12,
                color=colors[i],
            )
        ax.set_title("Token Embeddings", fontsize=14, pad=15)
        ax.axis("off")
        st.pyplot(fig)
    
    # Print the embeddings matrix as a dataframe
    st.write("**Embeddings Matrix:**")
    st.dataframe(embeddings)


    # Step 3: (Optional) Show additional internal process (skipped for minimalism)

    # Step 4: Output Generation via Ollama
    section_header("Step 3: Output Generation (via Ollama)")
    st.info("Generating a response using the LLM powered by Ollama.")
    with st.spinner("Generating response..."):
        real_response = ollama_query(query)
        time.sleep(1)  # Simulate processing delay

    # Display the response with a typewriter effect
    response_placeholder = st.empty()
    displayed_text = ""
    for char in real_response:
        displayed_text += char
        response_placeholder.markdown(f"**LLM Response:** {displayed_text}")
        time.sleep(0.03)  # Faster animation for smoothness

    st.success("LLM processing complete!")
    return real_response


# Main input area with clear, uncluttered design
st.markdown("## Enter your query:")
user_query = st.text_input(
    "", placeholder="Type your question here...", key="user_query"
)
if user_query:
    simulate_llm_process(user_query)
