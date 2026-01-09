from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from src.obesity_tc.make_dataset import atualizar_base_ptbr

st.set_page_config(page_title="Sistema de Predição de Obesidade", layout="wide")

# Caminhos base do projeto para localizar dados e modelo.
BASE_DIR = Path(__file__).resolve().parent
CAMINHO_MODELO = BASE_DIR / "models/modelo_obesidade.joblib"
CAMINHO_BASE = BASE_DIR / "data/raw/Obesity.csv"
CAMINHO_BASE_TRADUZIDA = BASE_DIR / "data/processed/base_traduzida_ptbr.csv"

# Mapas de tradução para exibição amigável no app.
MAPA_GENERO = {"Female": "Feminino", "Male": "Masculino"}
MAPA_SIM_NAO = {"yes": "Sim", "no": "Não"}
MAPA_FREQUENCIA = {
    "no": "não",
    "Sometimes": "às vezes",
    "Frequently": "frequentemente",
    "Always": "sempre",
}
MAPA_TRANSPORTE = {
    "Public_Transportation": "Transporte público",
    "Automobile": "Automóvel",
    "Walking": "Caminhada",
    "Motorbike": "Motocicleta",
    "Bike": "Bicicleta",
}
MAPA_NIVEL_OBESIDADE = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}

COLUNAS_PT = {
    "Gender": "Gênero",
    "Age": "Idade",
    "Height": "Altura (m)",
    "Weight": "Peso (kg)",
    "family_history": "Histórico familiar de sobrepeso",
    "FAVC": "Alimentos hipercalóricos frequentes",
    "FCVC": "Consumo de vegetais",
    "NCP": "Refeições principais",
    "CAEC": "Beliscar entre refeições",
    "SMOKE": "Fuma",
    "CH2O": "Consumo de água",
    "SCC": "Monitoramento de calorias",
    "FAF": "Atividade física",
    "TUE": "Tempo usando tecnologia",
    "CALC": "Consumo de álcool",
    "MTRANS": "Meio de transporte",
    "BMI": "IMC",
}


@st.cache_resource
def ler_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Treine primeiro com: "
            "python -m src.obesity_tc.train --data data/raw/Obesity.csv --target Obesity"
        )
    return load(CAMINHO_MODELO)


# Garante a base traduzida atualizada para o dashboard.
atualizar_base_ptbr(
    data_path=CAMINHO_BASE,
    output_path=CAMINHO_BASE_TRADUZIDA,
    coluna_alvo="Obesity",
)

st.title("Sistema de Predição de Obesidade")
st.caption(
    "Modelo preditivo multiclasse para apoiar a avaliação do nível de obesidade."
)

# Carrega o modelo treinado ou interrompe com mensagem clara.
try:
    pacote_modelo = ler_modelo()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Treine o modelo para habilitar as previsões.")
    st.stop()

pipeline_modelo = pacote_modelo["pipeline"]

# Coleta das entradas do usuário no sidebar.
with st.sidebar:
    st.header("Entradas do paciente")

    st.subheader("Perfil")
    genero = st.selectbox(
        "Gênero",
        ["Female", "Male"],
        format_func=lambda valor: MAPA_GENERO.get(valor, valor),
    )
    idade = st.number_input("Idade", min_value=1, max_value=120, value=30, step=1)
    altura = st.number_input(
        "Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01
    )
    peso = st.number_input(
        "Peso (kg)", min_value=20.0, max_value=300.0, value=80.0, step=0.1
    )

    st.subheader("Hábitos alimentares")
    historico_familiar = st.selectbox(
        "Histórico familiar de sobrepeso",
        ["yes", "no"],
        format_func=lambda valor: MAPA_SIM_NAO.get(valor, valor),
    )
    favc = st.selectbox(
        "Consome alimentos hipercalóricos frequentemente (FAVC)",
        ["yes", "no"],
        format_func=lambda valor: MAPA_SIM_NAO.get(valor, valor),
    )

    fcvc = st.slider("Consumo de vegetais (FCVC)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais (NCP)", 1, 4, 3)

    caec = st.selectbox(
        "Comer entre refeições (CAEC)",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda valor: MAPA_FREQUENCIA.get(valor, valor),
    )

    st.subheader("Estilo de vida")
    fumo = st.selectbox(
        "Fuma (SMOKE)",
        ["yes", "no"],
        format_func=lambda valor: MAPA_SIM_NAO.get(valor, valor),
    )

    ch2o = st.slider("Consumo de água (CH2O)", 1, 3, 2)
    scc = st.selectbox(
        "Monitoramento de calorias (SCC)",
        ["yes", "no"],
        format_func=lambda valor: MAPA_SIM_NAO.get(valor, valor),
    )

    faf = st.slider("Atividade física (FAF)", 0, 3, 1)
    tue = st.slider("Tempo usando tecnologia (TUE)", 0, 2, 1)

    calc = st.selectbox(
        "Consumo de álcool (CALC)",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda valor: MAPA_FREQUENCIA.get(valor, valor),
    )
    mtrans = st.selectbox(
        "Meio de transporte (MTRANS)",
        ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
        format_func=lambda valor: MAPA_TRANSPORTE.get(valor, valor),
    )

    botao_prever = st.button("Prever")

# Monta um registro com as mesmas colunas usadas no treino.
linha = {
    "Gender": genero,
    "Age": float(idade),
    "Height": float(altura),
    "Weight": float(peso),
    "family_history": historico_familiar,
    "FAVC": favc,
    "FCVC": int(fcvc),
    "NCP": int(ncp),
    "CAEC": caec,
    "SMOKE": fumo,
    "CH2O": int(ch2o),
    "SCC": scc,
    "FAF": int(faf),
    "TUE": int(tue),
    "CALC": calc,
    "MTRANS": mtrans,
}

dados_entrada = pd.DataFrame([linha])
# Calcula IMC para uso no modelo e no resumo do paciente.
dados_entrada["BMI"] = dados_entrada["Weight"] / (dados_entrada["Height"] ** 2)
imc = float(dados_entrada["BMI"].iloc[0])

# Converte valores para rótulos legíveis na tabela de exibição.
dados_exibicao = dados_entrada.copy()
for coluna, mapa in {
    "Gender": MAPA_GENERO,
    "family_history": MAPA_SIM_NAO,
    "FAVC": MAPA_SIM_NAO,
    "CAEC": MAPA_FREQUENCIA,
    "SMOKE": MAPA_SIM_NAO,
    "SCC": MAPA_SIM_NAO,
    "CALC": MAPA_FREQUENCIA,
    "MTRANS": MAPA_TRANSPORTE,
}.items():
    if coluna in dados_exibicao.columns:
        dados_exibicao[coluna] = dados_exibicao[coluna].map(mapa).fillna(
            dados_exibicao[coluna]
        )

dados_exibicao = dados_exibicao.rename(columns=COLUNAS_PT)

st.markdown("## Resumo do paciente")
metric_cols = st.columns(4)
metric_cols[0].metric("IMC", f"{imc:.1f}")
metric_cols[1].metric("Idade", int(idade))
metric_cols[2].metric("Peso (kg)", f"{peso:.1f}")
metric_cols[3].metric("Altura (m)", f"{altura:.2f}")

col_dados, col_resultado = st.columns([1.15, 1])

with col_dados:
    st.subheader("Dados informados")
    colunas_resumo = [
        "Gênero",
        "Idade",
        "Altura (m)",
        "Peso (kg)",
        "IMC",
        "Histórico familiar de sobrepeso",
    ]
    colunas_resumo = [c for c in colunas_resumo if c in dados_exibicao.columns]
    st.dataframe(
        dados_exibicao[colunas_resumo],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Ver todos os campos"):
        st.dataframe(dados_exibicao, hide_index=True, use_container_width=True)

with col_resultado:
    st.subheader("Resultado da predição")
    if botao_prever:
        # Executa a predição apenas quando solicitado.
        predicao = pipeline_modelo.predict(dados_entrada)[0]
        predicao_pt = MAPA_NIVEL_OBESIDADE.get(predicao, predicao)
        st.success(f"Nível previsto: **{predicao_pt}**")

        st.markdown("#### Orientação")
        st.write(
            "Este resultado é uma **estimativa estatística** e deve apoiar "
            "(não substituir) a avaliação clínica. Considere histórico "
            "médico, exames e acompanhamento profissional."
        )
        st.caption(
            "As recomendações clínicas devem ser feitas por profissionais de saúde."
        )
    else:
        st.info("Preencha as entradas e clique em **Prever**.")

st.divider()
st.caption("Pipeline: RandomForest + OneHotEncoder + MinMaxScaler.")
