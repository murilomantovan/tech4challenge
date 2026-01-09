import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Caminhos base para relatórios gerados no treino.
BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE_DIR / "reports/metrics.json"
REPORT_PATH = BASE_DIR / "reports/classification_report.txt"


def ler_metricas() -> dict:
    # Lê as métricas salvas no último treinamento.
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def ler_relatorio() -> str:
    # Lê o relatório detalhado de classificação, se existir.
    if not REPORT_PATH.exists():
        return ""
    return REPORT_PATH.read_text(encoding="utf-8")


st.title("Métricas e documentação do modelo")

st.markdown(
    """
### Predição de nível de obesidade
Bem-vindo à documentação do projeto Tech Challenge - Fase 4. Este trabalho foi
desenvolvido para apoiar a avaliação do nível de obesidade a partir de dados de perfil
e hábitos de vida.

### Objetivo do projeto
Construir um modelo de Machine Learning capaz de **classificar o nível de obesidade**
de um indivíduo com base em dados demográficos, hábitos alimentares e estilo de vida.
O resultado serve como apoio à triagem e não substitui a avaliação clínica.

### Materiais e fórmulas usadas na previsão
- **Base de dados:** `data/raw/Obesity.csv`, com informações demográficas e de hábitos.
- **Variáveis de entrada:** gênero, idade, altura, peso, histórico familiar, consumo de
  vegetais (FCVC), refeições principais (NCP), água (CH2O), atividade física (FAF),
  tempo de tecnologia (TUE), consumo de álcool (CALC), meio de transporte (MTRANS) e
  demais campos do questionário.
- **Variável derivada:** IMC (BMI) calculado por `IMC = peso (kg) / altura (m)^2`.
- **Tratamentos:** remoção de espaços em strings e arredondamento de variáveis discretas
  (FCVC, NCP, CH2O, FAF, TUE) para reduzir ruído.

### Como a previsão é gerada
- **Pré-processamento:** MinMaxScaler para variáveis numéricas e OneHotEncoder para
  variáveis categóricas.
- **Balanceamento:** SMOTE aplicado somente no conjunto de treino.
- **Modelo:** Random Forest multiclasse.

### Métricas usadas
- **Acurácia:** proporção total de acertos no conjunto avaliado.
- **Precisão:** entre as previsões de cada classe, quantas estavam corretas.
- **Recall (sensibilidade):** entre os casos reais de uma classe, quantos o modelo encontrou.
- **F1-score:** equilíbrio entre precisão e recall, útil quando há desbalanceamento.
- **Support:** número de amostras por classe no conjunto de teste.
- **Matriz de confusão:** visão detalhada de acertos e erros entre classes.
- **Médias macro e ponderada:** macro dá o mesmo peso para cada classe, a ponderada
  considera o volume de amostras (support).

### Resultados-chave
As métricas abaixo são geradas durante o treinamento (treino/teste) e ficam salvas em
`reports/metrics.json` e `reports/classification_report.txt`.
"""
)

metricas = ler_metricas()
relatorio = ler_relatorio()

if metricas:
    # Exibe resumo e matriz de confusão baseada no treino.
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
    # Orienta sobre como gerar as métricas caso não existam.
    st.info(
        "Relatórios não encontrados. Execute `python -m src.obesity_tc.train --data "
        "data/raw/Obesity.csv --target Obesity` ou o notebook "
        "`notebooks/modelo_obesidade_tc.ipynb` para gerar as métricas."
    )

if relatorio:
    # Mostra o relatório completo por classe.
    st.subheader("Relatório de classificação (treino/teste)")
    st.code(relatorio)
