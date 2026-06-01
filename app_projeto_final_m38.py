import io
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_scoring_pipeline import TARGET, calculate_metrics, score_dataframe

st.set_page_config(page_title="Credit Scoring M38", page_icon="💳", layout="wide")

MODEL_PATH = Path("model_final.pkl")


@st.cache_resource
def load_model(path: str = "model_final.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".ftr", ".feather")):
        return pd.read_feather(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Formato inválido. Envie CSV, FTR/Feather ou Parquet.")


st.title("💳 Projeto Final M38 — Credit Scoring")
st.caption("Aplicação Streamlit para escorar uma base de clientes usando o modelo treinado `model_final.pkl`.")

with st.sidebar:
    st.header("Configurações")
    cutoff = st.slider("Cutoff para classificar mau pagador", 0.01, 0.99, 0.50, 0.01)
    st.markdown("---")
    st.write("Arquivos necessários no mesmo diretório:")
    st.code("model_final.pkl\ncredit_scoring_pipeline.py")

if not MODEL_PATH.exists():
    st.error("Não encontrei `model_final.pkl`. Treine o modelo antes ou coloque o arquivo na pasta do app.")
    st.stop()

model = load_model(str(MODEL_PATH))

uploaded = st.file_uploader(
    "Suba a base para escoragem",
    type=["csv", "ftr", "feather", "parquet"],
    help="A base deve conter as mesmas colunas usadas no treino. A coluna `mau` é opcional.",
)

if uploaded is None:
    st.info("Envie um arquivo para começar. Para teste, use uma amostra da base `credit_scoring.ftr` exportada como CSV.")
    st.stop()

try:
    df = read_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"Erro ao carregar arquivo: {exc}")
    st.stop()

st.subheader("Prévia da base enviada")
st.write(f"Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]:,}")
st.dataframe(df.head(20), use_container_width=True)

try:
    scored = score_dataframe(model, df, cutoff=cutoff)
except Exception as exc:
    st.error(f"Erro na escoragem. Verifique se as colunas do arquivo batem com o treino. Detalhe: {exc}")
    st.stop()

st.subheader("Resultado da escoragem")
col1, col2, col3 = st.columns(3)
col1.metric("Score médio", f"{scored['score_mau'].mean():.4f}")
col2.metric("% classificados como mau", f"{(scored['classe_prevista'].eq('mau').mean() * 100):.2f}%")
col3.metric("Cutoff usado", f"{cutoff:.2f}")

st.dataframe(scored.head(50), use_container_width=True)

st.subheader("Distribuição do score")
st.bar_chart(scored["score_mau"].value_counts(bins=10).sort_index())

if TARGET in scored.columns:
    st.subheader("Métricas da base enviada")
    metrics = calculate_metrics(scored[TARGET].astype(int), scored["score_mau"], cutoff=cutoff)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia", f"{metrics['Acurácia']:.4f}")
    c2.metric("AUC", f"{metrics['AUC']:.4f}")
    c3.metric("Gini", f"{metrics['Gini']:.4f}")
    c4.metric("KS", f"{metrics['KS']:.4f}")

csv = scored.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Baixar base escorada em CSV",
    data=csv,
    file_name="base_escorada_m38.csv",
    mime="text/csv",
)
