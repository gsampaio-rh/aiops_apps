import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import collections
import graphviz as gv
from collections import Counter

# Bibliotecas para Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# Configuração da página
st.set_page_config(page_title="System Event Prediction App", layout="wide")
st.title("🚀 System Event Prediction Using Transformers")

# ================================
# Funções de Pré-processamento
# ================================

def parse_log(log_entry):
    """
    Extrai informações de um log do sistema Linux.
    Exemplo de log esperado:
    'Jun 14 15:16:01 combo sshd[19939]: authentication failure; ...'
    """
    pattern = (
        r"(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<hostname>\S+)\s+"
        r"(?P<process>\S+)\[\d+\]:\s+"
        r"(?P<event>.+)"
    )
    match = re.match(pattern, log_entry)
    if match:
        return match.groupdict()
    else:
        return {
            "timestamp": None,
            "hostname": None,
            "process": None,
            "event": log_entry.strip(),
        }


def convert_timestamp(ts_str):
    """Converte string de timestamp para objeto datetime (usando o ano corrente)."""
    try:
        current_year = datetime.now().year
        return datetime.strptime(f"{ts_str} {current_year}", "%b %d %H:%M:%S %Y")
    except Exception:
        return None


def clean_log_entry(log_entry):
    """
    Limpa o texto do log removendo IDs de processo, endereços IP e normalizando números.
    """
    log_entry = re.sub(r"\[\d+\]", "", log_entry)  # Remove IDs de processo
    log_entry = re.sub(r"\d+\.\d+\.\d+\.\d+", "IP_ADDRESS", log_entry)  # Substitui IPs
    log_entry = re.sub(
        r"uid=\d+", "uid=USER", log_entry
    )  # Normaliza identificadores de usuário
    return log_entry.strip()


# ================================
# 1. Carregar e Analisar Logs
# ================================
st.header("Carregar e Analisar Logs")
uploaded_file = st.file_uploader(
    "📁 Faça upload do arquivo de logs (.log ou .csv)", type=["log", "csv"]
)

if uploaded_file:
    try:
        # Leitura do arquivo
        log_lines = uploaded_file.getvalue().decode("utf-8").splitlines()
        st.success("✅ Arquivo carregado com sucesso!")

        # Criação do DataFrame inicial
        df_logs = pd.DataFrame({"log_entry": log_lines})
        with st.expander("Visualização dos Logs Carregados"):
            st.dataframe(df_logs.head())

        # Parse dos logs para extrair informações estruturadas
        parsed_logs = df_logs["log_entry"].apply(parse_log).apply(pd.Series)
        # Converter timestamp
        parsed_logs["timestamp"] = parsed_logs["timestamp"].apply(convert_timestamp)

        # Exibição de estatísticas básicas
        st.subheader("Estatísticas Básicas dos Logs")
        total_logs = len(parsed_logs)
        unique_processes = parsed_logs["process"].nunique()
        unique_events = parsed_logs["event"].nunique()
        unique_hosts = parsed_logs["hostname"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Logs", total_logs)
        col2.metric("Processos Únicos", unique_processes)
        col3.metric("Eventos Únicos", unique_events)
        col4.metric("Hosts Únicos", unique_hosts)

        st.subheader("Logs Estruturados")
        st.dataframe(parsed_logs.head())

        # Armazena os logs processados na sessão para uso posterior
        st.session_state["parsed_logs"] = parsed_logs

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Aguardando upload do arquivo de logs...")

# ================================
# 2. Visualizar Tendências e Anomalias
# ================================
if "parsed_logs" in st.session_state:
    st.header("Visualizar Tendências e Anomalias")
    parsed_logs = st.session_state["parsed_logs"].copy()

    # Filtrar registros com timestamp válido
    valid_logs = parsed_logs.dropna(subset=["timestamp"]).copy()
    valid_logs["hora"] = valid_logs["timestamp"].dt.hour

    # Filtros Interativos na Sidebar
    st.sidebar.subheader("Filtros")
    processos_disponiveis = valid_logs["process"].unique().tolist()
    eventos_disponiveis = valid_logs["event"].unique().tolist()

    filtro_processo = st.sidebar.multiselect(
        "Selecione Processo(s)", processos_disponiveis, default=processos_disponiveis
    )
    filtro_evento = st.sidebar.multiselect(
        "Selecione Evento(s)", eventos_disponiveis, default=eventos_disponiveis
    )

    # Aplicar filtros
    df_filtrado = valid_logs[
        (valid_logs["process"].isin(filtro_processo))
        & (valid_logs["event"].isin(filtro_evento))
    ]

    with st.expander("📈 Gráfico: Frequência de Logs ao Longo do Tempo"):
        freq_por_hora = df_filtrado["hora"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=freq_por_hora.index, y=freq_por_hora.values, marker="o", ax=ax)
        ax.set_xlabel("Hora do Dia")
        ax.set_ylabel("Número de Logs")
        ax.set_title("Distribuição de Logs por Hora")
        ax.grid(True)
        st.pyplot(fig)

    with st.expander("📈 Top 10 Eventos Mais Recorrentes"):
        top_eventos = df_filtrado["event"].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_eventos.values, y=top_eventos.index, palette="viridis", ax=ax2)
        ax2.set_xlabel("Frequência")
        ax2.set_ylabel("Evento")
        ax2.set_title("Top 10 Eventos")
        ax2.grid(True)
        st.pyplot(fig2)

else:
    st.info("Por favor, carregue os logs na etapa 1 para visualizar tendências.")

st.header("Tokenizar Logs")
if "parsed_logs" in st.session_state:
    # Create a DataFrame with cleaned log events
    logs_df = st.session_state["parsed_logs"].copy()
    logs_df["event_clean"] = logs_df["event"].apply(clean_log_entry)

    cleaned_logs = logs_df["event_clean"].tolist()

    # Allow the user to select a log event from the list
    selected_index = st.selectbox(
        "Selecione um log para visualizar a tokenização:",
        list(range(len(cleaned_logs))),
        format_func=lambda i: f"{i+1}. {cleaned_logs[i][:80]}{'...' if len(cleaned_logs[i])>80 else ''}",
    )
    selected_log = cleaned_logs[selected_index]

    st.markdown("**Log Selecionado:**")
    st.code(selected_log, language="bash")

    # Tokenize using TF-IDF to show the numerical vector representation
    tfidf_vectorizer_token = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    # Fit the vectorizer on all cleaned logs for consistency
    tfidf_vectorizer_token.fit(logs_df["event_clean"])
    vector = tfidf_vectorizer_token.transform([selected_log])

    # Extract nonzero entries (values and corresponding feature indexes)
    nonzero_indices = vector.nonzero()[1]  # column indices of nonzero elements
    nonzero_values = vector.data

    # Format the results as strings
    values_str = " ".join([f"{val:.8f}" for val in nonzero_values])
    indexes_str = " ".join(map(str, nonzero_indices))

    # -------------------------------------------
    # Show the log broken into tokens with animation
    # -------------------------------------------
    st.markdown("**Log Broken Into Tokens:**")

    # Tokenize the selected log into individual tokens using a simple regex
    tokens = re.findall(r"\S+", selected_log)

    # Join tokens into a single string separated by a vertical bar for clarity
    token_string = " | ".join(tokens)

    # Display the tokens
    st.code(token_string)

    st.markdown("**Tokenized Log Sequences (Nonzero Values Only)**")
    st.code(f"X: [{values_str}]\n(Indexes: [{indexes_str}])", language="python")

else:
    st.info("Carregue os logs na etapa 1 para visualizar os eventos tokenizados.")

with st.expander("Log Transitions Network Graph"):
    if "parsed_logs" in st.session_state:

        label_encoder = LabelEncoder()

        # Ensure y_labels and label_encoder are available
        if "y_labels" not in st.session_state or "label_encoder" not in st.session_state:
            logs_pred = st.session_state["parsed_logs"].dropna(subset=["event"]).copy()
            logs_pred["event_clean"] = logs_pred["event"].apply(clean_log_entry)
            tfidf_vectorizer_temp = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
            tfidf_vectorizer_temp.fit(logs_pred["event_clean"])
            label_encoder_temp = LabelEncoder()
            y_labels = label_encoder_temp.fit_transform(logs_pred["event_clean"])
            st.session_state["y_labels"] = y_labels
            st.session_state["label_encoder"] = label_encoder_temp
        else:
            y_labels = st.session_state["y_labels"]
            label_encoder = st.session_state["label_encoder"]

        # 🎯 Limit to top N most common transitions for clarity
        TOP_N_EVENTS = 100

        # 🔄 Count transitions between consecutive log events
        transition_counts = Counter(zip(y_labels[:-1], y_labels[1:]))
        top_transitions = dict(transition_counts.most_common(TOP_N_EVENTS))

        # 🎨 Create a directed graph focusing on top transitions
        G = nx.DiGraph()
        for (event1, event2), count in top_transitions.items():
            G.add_edge(event1, event2, weight=count)

        # 📌 Define node labels as the original log event texts
        node_labels = {i: label_encoder.inverse_transform([i])[0] for i in G.nodes()}

        # 🎨 Define event categories for color mapping
        event_categories = ["auth", "system", "network", "error", "unknown"]
        event_colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9E9E9E"]

        # 🔄 Assign categories based on event type keywords
        def categorize_event(event_text):
            if "authentication" in event_text or "login" in event_text:
                return "auth"
            elif "systemd" in event_text or "service" in event_text:
                return "system"
            elif "network" in event_text or "IP" in event_text:
                return "network"
            elif "error" in event_text or "failure" in event_text:
                return "error"
            return "unknown"

        category_map = {node: categorize_event(node_labels[node]) for node in G.nodes()}
        node_colors = [
            event_colors[event_categories.index(category_map[node])] for node in G.nodes()
        ]

        # 📌 Position nodes using a force-directed layout
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=0.9)

        # 🟢 Draw nodes with category-based colors
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=100, alpha=0.95, edgecolors="white"
        )

        # 🔗 Draw edges with thickness based on transition strength
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.7,
            edge_color="gray",
            width=[
                (top_transitions[(u, v)] / max(top_transitions.values())) * 6
                for u, v in G.edges()
            ],
            arrows=True,
            connectionstyle="arc3,rad=0.2",
        )

        # 🏷️ Draw labels with adaptive font scaling
        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_labels,
            font_size=5,
            font_weight="bold",
            verticalalignment="center",
        )

        # 🎨 Add title & remove gridlines
        plt.title(
            "Log Event Transitions",
            fontsize=16,
            fontweight="bold",
            color="#333333",
        )
        plt.grid(False)

        # ✅ Display the refined visualization in Streamlit
        st.pyplot(plt)
    else:
        st.info(
            "Carregue os logs na etapa 1 para visualizar o grafo de transições entre eventos."
        )

st.header("Training Loss & Accuracy Monitoring")
if "parsed_logs" in st.session_state:
    # Prepare the data from parsed logs
    logs_pred = st.session_state["parsed_logs"].dropna(subset=["event"]).copy()
    logs_pred["event_clean"] = logs_pred["event"].apply(clean_log_entry)

    # Vectorize the cleaned log events
    tfidf_vectorizer_train = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    X_tfidf_all = tfidf_vectorizer_train.fit_transform(
        logs_pred["event_clean"].tolist()
    )

    # Encode labels using the cleaned events
    label_encoder_train = LabelEncoder()
    y_all = label_encoder_train.fit_transform(logs_pred["event_clean"])

    # Create sequences: use each event to predict the next one
    X = X_tfidf_all[:-1]
    y = y_all[1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize lists to track training metrics
    losses = []
    accuracies = []
    num_iterations = 50  # Adjust as needed

    # Initialize the model with warm_start=True to retain progress
    log_model = LogisticRegression(
        max_iter=1, multi_class="multinomial", solver="lbfgs", warm_start=True
    )

    st.markdown("### Training Progress")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Train in multiple steps and record loss & accuracy
    for i in range(num_iterations):
        log_model.fit(X_train, y_train)  # Train for one iteration
        y_pred_proba = log_model.predict_proba(X_train)
        y_train_pred = log_model.predict(X_train)

        loss = log_loss(y_train, y_pred_proba)
        accuracy = accuracy_score(y_train, y_train_pred)

        losses.append(loss)
        accuracies.append(accuracy)

        progress_text.text(
            f"Iteration {i+1}/{num_iterations}: Log Loss = {loss:.4f}, Accuracy = {accuracy:.2%}"
        )
        progress_bar.progress((i + 1) / num_iterations)

    # Evaluate the final model on test data
    y_test_pred = log_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_test_pred)
    st.success(f"Final Model Accuracy on Test Data: {final_accuracy:.2%}")

    st.session_state["tfidf_vectorizer_train"] = tfidf_vectorizer_train
    st.session_state["log_model"] = log_model
    st.session_state["label_encoder_train"] = label_encoder_train

    with st.expander("📈 Training Loss & Accuracy Over Iterations"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            range(1, num_iterations + 1),
            losses,
            marker="o",
            linestyle="--",
            color="b",
            label="Training Loss",
        )
        ax.plot(
            range(1, num_iterations + 1),
            accuracies,
            marker="s",
            linestyle="-",
            color="g",
            label="Training Accuracy",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_title("📉 Training Loss & Accuracy Over Iterations")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

else:
    st.info("Carregue os logs na etapa 1 para treinar o modelo.")

# ================================
# 3. Prever Próximo Evento (Machine Learning)
# ================================
if "parsed_logs" in st.session_state:
    st.header("Prever Próximo Evento (Machine Learning)")
    logs_pred = st.session_state["parsed_logs"].dropna(subset=["event"]).copy()
    logs_pred["event_clean"] = logs_pred["event"].apply(clean_log_entry)

    if logs_pred.shape[0] < 2:
        st.warning("Número insuficiente de logs para predição.")
    else:
        # Make sure the trained model and vectorizer are saved in session_state
        if (
            "tfidf_vectorizer_train" not in st.session_state
            or "log_model" not in st.session_state
            or "label_encoder_train" not in st.session_state
        ):
            st.error(
                "O modelo não foi treinado. Execute a etapa de treinamento primeiro."
            )
        else:
            # Interface para seleção de um log para predição
            # Use the stored vectorizer to determine the total number of vectorized logs
            tfidf_vectorizer = st.session_state["tfidf_vectorizer_train"]
            total_logs = (
                tfidf_vectorizer.transform(logs_pred["event_clean"]).shape[0] - 1
            )
            sample_index = st.slider(
                "Selecione o índice do log para predição",
                min_value=0,
                max_value=int(total_logs),
                value=0,
                step=1,
            )

            # Exibir o log selecionado
            sample_log = logs_pred["event_clean"].iloc[sample_index]
            st.markdown("**Log Selecionado:**")
            st.code(sample_log, language="bash")

            # Compute the vector for the selected log using the stored vectorizer
            with st.spinner("Realizando previsão..."):
                time.sleep(1)  # Simulate processing delay
                sample_vector = tfidf_vectorizer.transform([sample_log])
                log_model = st.session_state["log_model"]
                predicted_label = log_model.predict(sample_vector)
                predicted_event = st.session_state[
                    "label_encoder_train"
                ].inverse_transform(predicted_label)[0]
                pred_prob = log_model.predict_proba(sample_vector)[0]
                confidence = pred_prob.max()

            st.markdown("**🔮 Próximo Evento Previsto:**")
            st.code(predicted_event, language="bash")
            st.markdown(f"**Nível de Confiança:** {confidence*100:.2f}%")

            with st.expander("Ver Probabilidades de Previsão"):
                prob_df = pd.DataFrame(
                    {
                        "Evento": st.session_state[
                            "label_encoder_train"
                        ].inverse_transform(np.arange(len(pred_prob))),
                        "Probabilidade": pred_prob,
                    }
                ).sort_values("Probabilidade", ascending=False)
                st.dataframe(prob_df)
else:
    st.info("Carregue os logs na etapa 1 para usar o módulo de predição.")


# # ================================
# # 4. Grafo Interativo: Relações entre Eventos (RF-05)
# # ================================
# if "parsed_logs" in st.session_state:
#     st.header("4. Grafo Interativo: Relações entre Eventos")

#     # Para construir o grafo, vamos usar os logs limpos
#     logs_for_graph = st.session_state["parsed_logs"].dropna(subset=["event"]).copy()
#     logs_for_graph["event_clean"] = logs_for_graph["event"].apply(clean_log_entry)

#     # Gerar transições (par ordenado de eventos consecutivos)
#     eventos = logs_for_graph["event_clean"].tolist()
#     transicoes = list(zip(eventos[:-1], eventos[1:]))
#     transition_counts = collections.Counter(transicoes)

#     # Criar grafo direcionado com base nas transições
#     G = nx.DiGraph()
#     for (ev_from, ev_to), count in transition_counts.items():
#         # Adiciona aresta com atributo 'weight'
#         G.add_edge(ev_from, ev_to, weight=count)

#     # Gerar layout com spring layout
#     pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

#     # Criar traços para as arestas
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])

#     edge_trace = go.Scatter(
#         x=edge_x,
#         y=edge_y,
#         line=dict(width=0.5, color="#888"),
#         hoverinfo="none",
#         mode="lines",
#     )

#     # Criar traços para os nós
#     node_x = []
#     node_y = []
#     node_text = []
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         # Exibir o nome do evento e, opcionalmente, o grau do nó
#         node_text.append(f"{node} (Grau: {G.degree(node)})")

#     node_trace = go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode="markers+text",
#         text=node_text,
#         textposition="bottom center",
#         hoverinfo="text",
#         marker=dict(
#             showscale=True,
#             colorscale="YlGnBu",
#             reversescale=True,
#             color=[G.degree(node) for node in G.nodes()],
#             size=10,
#             line_width=2,
#         ),
#     )

#     fig_graph = go.Figure(
#         data=[edge_trace, node_trace],
#         layout=go.Layout(
#             title="<b>Grafo de Transições entre Eventos</b>",
#             # titlefont_size=16,
#             showlegend=False,
#             hovermode="closest",
#             margin=dict(b=20, l=5, r=5, t=40),
#             annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
#             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         ),
#     )
#     st.plotly_chart(fig_graph, use_container_width=True)
# else:
#     st.info(
#         "Carregue os logs na etapa 1 para visualizar o grafo de relações entre eventos."
#     )

# # ================================
# # 5. Explicação Interativa do Pipeline e dos Modelos de IA (RF-06)
# # ================================
# st.header("5. Explicação do Pipeline e dos Modelos de IA")

# with st.expander("Visualizar Fluxo do Pipeline"):
#     pipeline_diagram = """
#     digraph {
#         rankdir=LR;
#         node [shape=box, style=filled, color="#EFEFEF", fontname="Helvetica"];
#         A [label="Upload de Logs"];
#         B [label="Parsing e Pré-processamento"];
#         C [label="Visualização de Tendências\n& Anomalias"];
#         D [label="Modelagem ML\n(Logistic Regression)"];
#         E [label="Predição do Próximo Evento"];
#         F [label="Comparação e Upgrade\npara Transformers"];

#         A -> B -> C;
#         B -> D -> E;
#         D -> F;
#     }
#     """
#     st.graphviz_chart(pipeline_diagram)

# with st.expander("Detalhamento do Pipeline e Modelos"):
#     st.markdown(
#         """
#     **Pipeline de Processamento e Predição:**

#     1. **Upload de Logs:**
#        O usuário faz o upload de arquivos `.log` ou `.csv` contendo os registros do sistema.

#     2. **Parsing e Pré-processamento:**
#        - Extração de informações: timestamp, hostname, processo e descrição do evento.
#        - Conversão de timestamps e limpeza dos textos dos eventos.

#     3. **Visualização de Tendências e Anomalias:**
#        - Geração de gráficos interativos que mostram a distribuição dos logs ao longo do tempo.
#        - Detecção de picos suspeitos e análise dos eventos mais recorrentes.

#     4. **Modelagem ML (Baseline):**
#        - Utilização de **Logistic Regression** para prever o próximo evento com base na sequência de logs.
#        - Vetorização dos eventos com **TF-IDF** e codificação dos rótulos.

#     5. **Predição do Próximo Evento:**
#        - Seleção interativa de um log para que o modelo preveja o próximo evento, exibindo o nível de confiança.
#     """
#     )
