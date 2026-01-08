import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.obesity_tc.make_dataset import preprocessar_base

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models/modelo_obesidade.joblib"
DATA_PATH = BASE_DIR / "data/raw/Obesity.csv"
METRICS_PATH = BASE_DIR / "reports/metrics.json"
REPORT_PATH = BASE_DIR / "reports/classification_report.txt"

MAPA_NIVEL_OBESIDADE = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}


@st.cache_resource
def ler_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Treine com: "
            "python -m src.obesity_tc.train --data data/raw/Obesity.csv --target Obesity"
        )
    return load(MODEL_PATH)


@st.cache_data
def ler_base() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Base de dados não encontrada em data/raw/Obesity.csv.")
    df_raw = pd.read_csv(DATA_PATH)
    return preprocessar_base(df_raw, coluna_alvo="Obesity")


def ler_metricas() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def ler_relatorio() -> str:
    if not REPORT_PATH.exists():
        return ""
    return REPORT_PATH.read_text(encoding="utf-8")


st.title("Métricas e documentação do modelo")

st.markdown(
    """
### 🧭 Predição de nível de obesidade
Bem-vindo à documentação do projeto Tech Challenge - Fase 4. Este trabalho foi
desenvolvido para apoiar a avaliação do nível de obesidade a partir de dados de perfil
e hábitos de vida.

### 🎯 Objetivo do projeto
Construir um modelo de Machine Learning capaz de **classificar o nível de obesidade** de um
indivíduo com base em dados demográficos, hábitos alimentares e estilo de vida. O resultado
serve como apoio à triagem e não substitui a avaliação clínica.

### 🧩 A solução
- **Análise exploratória:** entendimento da distribuição das classes e do perfil dos dados.
- **Engenharia de atributos:** preparo e padronização do dataset, incluindo cálculo de IMC.
- **Modelagem:** Random Forest com balanceamento das classes para reduzir vieses.
- **Aplicação web:** interface em Streamlit para predições individuais.

### 📊 Métricas usadas
- ✅ **Acurácia:** proporção total de acertos no conjunto avaliado.
- 🎯 **Precisão:** entre as previsões de cada classe, quantas estavam corretas.
- 🔎 **Recall (sensibilidade):** entre os casos reais de uma classe, quantos o modelo encontrou.
- 🧪 **F1-score:** equilíbrio entre precisão e recall, útil quando há desbalanceamento.
- 📦 **Support:** número de amostras por classe no conjunto de teste.
- 🧭 **Matriz de confusão:** visão detalhada de acertos e erros entre classes.
- ⚖️ **Médias macro e ponderada:** macro dá o mesmo peso para cada classe, a ponderada
  considera o volume de amostras (support).

### 🏁 Resultados-chave
As métricas abaixo são geradas a partir do **modelo treinado** e dos **dados atuais**.
Use o relatório por classe para comparar precisão, recall e f1-score.
"""
)

metricas = ler_metricas()
relatorio = ler_relatorio()

if metricas:
    st.subheader("Resumo da última execução")
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", f"{float(metricas.get('acuracia', 0)):.4f}")
    col2.metric("Treino", int(metricas.get("n_treino", 0)))
    col3.metric("Teste", int(metricas.get("n_teste", 0)))

    matriz = metricas.get("matriz_confusao")
    if matriz:
        st.subheader("Matriz de confusão (treino/teste)")
        classes = metricas.get("classes") or metricas.get("classes_original")
        if classes:
            st.dataframe(pd.DataFrame(matriz, index=classes, columns=classes))
        else:
            st.dataframe(pd.DataFrame(matriz))
else:
    st.info(
        "Relatórios não encontrados. Execute o notebook "
        "`notebooks/modelo_obesidade_tc.ipynb` para gerar as métricas."
    )

if relatorio:
    st.subheader("Relatório de classificação (treino/teste)")
    st.code(relatorio)

st.divider()
st.subheader("Métricas rápidas com a base atual")

if st.button("Calcular métricas agora"):
    try:
        bundle = ler_modelo()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    try:
        df = ler_base()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if "Obesity_level" not in df.columns:
        st.error("A base não possui a coluna Obesity_level para avaliação.")
        st.stop()

    y_true = df["Obesity_level"]
    X = df.drop(columns=["Obesity_level"])

    pred = bundle["pipeline"].predict(X)
    acc = accuracy_score(y_true, pred)
    st.metric("Acurácia (base atual)", f"{acc:.4f}")

    classes_ordenadas = sorted(y_true.unique().tolist())
    classes_pt = [MAPA_NIVEL_OBESIDADE.get(c, c) for c in classes_ordenadas]
    st.subheader("Relatório de classificação (base atual)")
    st.code(
        classification_report(
            y_true,
            pred,
            labels=classes_ordenadas,
            target_names=classes_pt,
            digits=4,
        )
    )

    st.subheader("Matriz de confusão (base atual)")
    st.dataframe(confusion_matrix(y_true, pred, labels=classes_ordenadas))
