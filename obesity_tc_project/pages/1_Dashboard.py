import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from src.obesity_tc.make_dataset import preprocessar_base, salvar_base_ptbr

MAPA_NIVEL_OBESIDADE = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}

MAPA_GENERO = {"Female": "Feminino", "Male": "Masculino"}

MAPA_TRANSPORTE = {
    "Public_Transportation": "Transporte público",
    "Automobile": "Automóvel",
    "Walking": "Caminhada",
    "Motorbike": "Motocicleta",
    "Bike": "Bicicleta",
}

MAPA_SIM_NAO = {"yes": "Sim", "no": "Não"}
MAPA_FREQUENCIA = {
    "no": "não",
    "Sometimes": "às vezes",
    "Frequently": "frequentemente",
    "Always": "sempre",
}

DATA_PATH = Path("data/raw/Obesity.csv")
CAMINHO_BASE_TRADUZIDA = Path("data/processed/base_traduzida_ptbr.csv")
ORDEM_NIVEIS = list(MAPA_NIVEL_OBESIDADE.values())
ROTULOS_NUMERICOS = {
    "Age": "Idade",
    "Height": "Altura (m)",
    "Weight": "Peso (kg)",
    "FCVC": "Consumo de vegetais",
    "NCP": "Refeições principais",
    "CH2O": "Consumo de água",
    "FAF": "Atividade física",
    "TUE": "Tempo usando tecnologia",
    "BMI": "IMC",
}
ROTULOS_CATEGORICOS = {
    "Nivel_Obesidade_PT": "Nível de obesidade",
    "Genero_PT": "Gênero",
    "FAVC_PT": "Alimentos hipercalóricos",
    "CAEC_PT": "Beliscar entre refeições",
    "CALC_PT": "Consumo de álcool",
    "SCC_PT": "Monitoramento de calorias",
    "Historico_PT": "Histórico familiar",
    "Transporte_PT": "Meio de transporte",
}
ROTULOS_GRAFICOS = {**ROTULOS_NUMERICOS, **ROTULOS_CATEGORICOS}


@st.cache_data
def ler_base() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Base de dados não encontrada em data/raw/Obesity.csv.")
    df_raw = pd.read_csv(DATA_PATH)
    return preprocessar_base(df_raw, coluna_alvo="Obesity")


st.title("Dashboard Analítico")
st.caption("Análises exploratórias da base de pacientes.")

try:
    df = ler_base()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
else:
    if (
        not CAMINHO_BASE_TRADUZIDA.exists()
        or CAMINHO_BASE_TRADUZIDA.stat().st_mtime < DATA_PATH.stat().st_mtime
    ):
        salvar_base_ptbr(df, CAMINHO_BASE_TRADUZIDA)

if "BMI" not in df.columns and "Height" in df.columns and "Weight" in df.columns:
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

df["Nivel_Obesidade_PT"] = df["Obesity_level"].map(MAPA_NIVEL_OBESIDADE).fillna(
    df["Obesity_level"]
)
df["Genero_PT"] = df["Gender"].map(MAPA_GENERO).fillna(df["Gender"])

if "MTRANS" in df.columns:
    df["Transporte_PT"] = df["MTRANS"].map(MAPA_TRANSPORTE).fillna(df["MTRANS"])

if "FAVC" in df.columns:
    df["FAVC_PT"] = df["FAVC"].map(MAPA_SIM_NAO).fillna(df["FAVC"])

if "CAEC" in df.columns:
    df["CAEC_PT"] = df["CAEC"].map(MAPA_FREQUENCIA).fillna(df["CAEC"])

if "CALC" in df.columns:
    df["CALC_PT"] = df["CALC"].map(MAPA_FREQUENCIA).fillna(df["CALC"])

if "SCC" in df.columns:
    df["SCC_PT"] = df["SCC"].map(MAPA_SIM_NAO).fillna(df["SCC"])

if "SMOKE" in df.columns:
    df["SMOKE_PT"] = df["SMOKE"].map(MAPA_SIM_NAO).fillna(df["SMOKE"])

if "family_history" in df.columns:
    df["Historico_PT"] = df["family_history"].map(MAPA_SIM_NAO).fillna(
        df["family_history"]
    )

metric_cols = st.columns(4)
metric_cols[0].metric("Registros", len(df))
metric_cols[1].metric(
    "Idade média",
    f"{df['Age'].mean():.1f}" if "Age" in df.columns else "-",
)
metric_cols[2].metric(
    "IMC médio",
    f"{df['BMI'].mean():.1f}" if "BMI" in df.columns else "-",
)
metric_cols[3].metric(
    "Peso médio (kg)",
    f"{df['Weight'].mean():.1f}" if "Weight" in df.columns else "-",
)

st.subheader("Distribuições principais")
col1, col2 = st.columns(2)

with col1:
    fig_niveis = px.pie(
        df,
        names="Nivel_Obesidade_PT",
        title="Distribuição dos níveis de obesidade",
        category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
        labels=ROTULOS_GRAFICOS,
    )
    st.plotly_chart(fig_niveis, use_container_width=True)

with col2:
    fig_genero = px.bar(
        df,
        x="Genero_PT",
        title="Distribuição por gênero",
        labels=ROTULOS_GRAFICOS,
    )
    st.plotly_chart(fig_genero, use_container_width=True)

st.subheader("Idade e IMC")
col3, col4 = st.columns(2)

with col3:
    if "Age" in df.columns:
        fig_idade = px.histogram(
            df,
            x="Age",
            color="Nivel_Obesidade_PT",
            nbins=20,
            barmode="overlay",
            title="Distribuição de idade por nível de obesidade",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_idade, use_container_width=True)
    else:
        st.info("Coluna Age não disponível para análise de idade.")

with col4:
    if "BMI" in df.columns:
        fig_imc = px.histogram(
            df,
            x="BMI",
            color="Nivel_Obesidade_PT",
            nbins=20,
            barmode="overlay",
            title="Distribuição de IMC por nível de obesidade",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_imc, use_container_width=True)
    else:
        st.info("Coluna BMI não disponível para análise de IMC.")

st.subheader("Hábitos alimentares")
col5, col6 = st.columns(2)

with col5:
    if "FAVC_PT" in df.columns:
        fig_favc = px.histogram(
            df,
            x="FAVC_PT",
            color="Nivel_Obesidade_PT",
            barmode="group",
            title="Consumo de alimentos hipercalóricos",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_favc, use_container_width=True)
    else:
        st.info("Coluna FAVC não disponível para análise de hábitos.")

with col6:
    if "CAEC_PT" in df.columns:
        fig_caec = px.histogram(
            df,
            x="CAEC_PT",
            color="Nivel_Obesidade_PT",
            barmode="group",
            title="Beliscar entre refeições",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_caec, use_container_width=True)
    else:
        st.info("Coluna CAEC não disponível para análise de hábitos.")

st.subheader("Consumo e refeições")
col7, col8 = st.columns(2)

with col7:
    if "NCP" in df.columns:
        fig_ncp = px.histogram(
            df,
            x="NCP",
            color="Nivel_Obesidade_PT",
            nbins=4,
            barmode="group",
            title="Número de refeições principais (NCP)",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_ncp, use_container_width=True)
    else:
        st.info("Coluna NCP não disponível para análise de refeições.")

with col8:
    if "CALC_PT" in df.columns:
        fig_calc = px.histogram(
            df,
            x="CALC_PT",
            color="Nivel_Obesidade_PT",
            barmode="group",
            title="Consumo de álcool",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_calc, use_container_width=True)
    else:
        st.info("Coluna CALC não disponível para análise de consumo.")

st.subheader("Atividade e comportamento")
col9, col10 = st.columns(2)

with col9:
    if "FAF" in df.columns:
        fig_faf = px.histogram(
            df,
            x="FAF",
            color="Nivel_Obesidade_PT",
            nbins=4,
            barmode="group",
            title="Atividade física (FAF) por nível de obesidade",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_faf, use_container_width=True)
    else:
        st.info("Coluna FAF não disponível para análise de atividade física.")

with col10:
    if "TUE" in df.columns:
        fig_tue = px.histogram(
            df,
            x="TUE",
            color="Nivel_Obesidade_PT",
            nbins=3,
            barmode="group",
            title="Tempo usando tecnologia (TUE) por nível de obesidade",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_tue, use_container_width=True)
    else:
        st.info("Coluna TUE não disponível para análise de tempo de tela.")

st.subheader("Hidratação e monitoramento")
col11, col12 = st.columns(2)

with col11:
    if "CH2O" in df.columns:
        fig_ch2o = px.histogram(
            df,
            x="CH2O",
            color="Nivel_Obesidade_PT",
            nbins=3,
            barmode="group",
            title="Consumo de água (CH2O)",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_ch2o, use_container_width=True)
    else:
        st.info("Coluna CH2O não disponível para análise de água.")

with col12:
    if "SCC_PT" in df.columns:
        fig_scc = px.histogram(
            df,
            x="SCC_PT",
            color="Nivel_Obesidade_PT",
            barmode="group",
            title="Monitoramento de calorias (SCC)",
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
            labels=ROTULOS_GRAFICOS,
        )
        st.plotly_chart(fig_scc, use_container_width=True)
    else:
        st.info("Coluna SCC não disponível para análise de monitoramento.")

if "Historico_PT" in df.columns:
    st.subheader("Histórico familiar de sobrepeso")
    fig_hist = px.histogram(
        df,
        x="Historico_PT",
        color="Nivel_Obesidade_PT",
        barmode="group",
        title="Histórico familiar por nível de obesidade",
        category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
        labels=ROTULOS_GRAFICOS,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Relações entre variáveis")
col13, col14 = st.columns(2)

with col13:
    if "Height" in df.columns and "Weight" in df.columns:
        fig_dispersao = px.scatter(
            df,
            x="Height",
            y="Weight",
            color="Nivel_Obesidade_PT",
            title="Relação entre altura e peso",
            labels={
                "Height": "Altura (m)",
                "Weight": "Peso (kg)",
                "Nivel_Obesidade_PT": "Nível de obesidade",
            },
            category_orders={"Nivel_Obesidade_PT": ORDEM_NIVEIS},
        )
        st.plotly_chart(fig_dispersao, use_container_width=True)
    else:
        st.info("Colunas Height/Weight não disponíveis para correlação.")

with col14:
    num_cols = [
        c
        for c in ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
        if c in df.columns
    ]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        corr = corr.rename(columns=ROTULOS_NUMERICOS, index=ROTULOS_NUMERICOS)
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlação entre variáveis numéricas",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Sem variáveis numéricas suficientes para correlação.")

if "Transporte_PT" in df.columns:
    st.subheader("Modos de transporte")
    df_transporte = df["Transporte_PT"].value_counts().reset_index()
    df_transporte.columns = ["Transporte", "Quantidade"]
    fig_transporte = px.bar(
        df_transporte,
        x="Transporte",
        y="Quantidade",
        title="Uso de modos de transporte",
        labels={"Transporte": "Meio de transporte", "Quantidade": "Quantidade"},
    )
    st.plotly_chart(fig_transporte, use_container_width=True)
