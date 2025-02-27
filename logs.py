import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="System Event Prediction App", layout="wide")
st.title("🚀 System Event Prediction Using Transformers")

# Função para parse dos logs
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
            "event": log_entry.strip()
        }

# Função para converter string de timestamp em datetime (assumindo ano corrente)
def convert_timestamp(ts_str):
    try:
        # Usamos o ano atual para compor o datetime
        current_year = datetime.now().year
        return datetime.strptime(f"{ts_str} {current_year}", "%b %d %H:%M:%S %Y")
    except Exception:
        return None

# Seção 1: Upload de Arquivo
st.header("1. Carregar e Analisar Logs")
uploaded_file = st.file_uploader("📁 Faça upload do arquivo de logs (.log ou .csv)", type=["log", "csv"])

if uploaded_file:
    try:
        # Leitura do arquivo
        log_lines = uploaded_file.getvalue().decode("utf-8").splitlines()
        st.success("✅ Arquivo carregado com sucesso!")

        # Criação do DataFrame inicial
        df_logs = pd.DataFrame({"log_entry": log_lines})
        st.subheader("Visualização dos Logs Carregados")
        st.dataframe(df_logs.head())

        # Parse dos logs para extrair informações estruturadas
        parsed_logs = df_logs["log_entry"].apply(parse_log).apply(pd.Series)
        # Converter timestamp
        parsed_logs["timestamp"] = parsed_logs["timestamp"].apply(convert_timestamp)

        st.subheader("Logs Estruturados")
        st.dataframe(parsed_logs.head())

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

        # Exemplo de gráfico: frequência de eventos ao longo do tempo
        st.subheader("Frequência de Logs ao Longo do Tempo")
        # Remover registros sem timestamp válido
        valid_logs = parsed_logs.dropna(subset=["timestamp"])
        valid_logs["hora"] = valid_logs["timestamp"].dt.hour
        freq_por_hora = valid_logs["hora"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=freq_por_hora.index, y=freq_por_hora.values, marker="o", ax=ax)
        ax.set_xlabel("Hora do Dia")
        ax.set_ylabel("Número de Logs")
        ax.set_title("Logs por Hora")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Aguardando upload do arquivo de logs...")
