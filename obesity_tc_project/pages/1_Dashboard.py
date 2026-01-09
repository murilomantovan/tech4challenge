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
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre",
}

# Caminhos e rótulos usados no dashboard.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/Obesity.csv"
CAMINHO_BASE_TRADUZIDA = BASE_DIR / "data/processed/base_traduzida_ptbr.csv"
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
    "SMOKE_PT": "Fuma",
}
ROTULOS_GRAFICOS = {**ROTULOS_NUMERICOS, **ROTULOS_CATEGORICOS}
ROTULOS_EIXOS = {**ROTULOS_GRAFICOS, "count": "Quantidade"}


@st.cache_data
def ler_base() -> pd.DataFrame:
    # Carrega a base bruta e aplica o pré-processamento padrão.
    if not DATA_PATH.exists():
        raise FileNotFoundError("Base de dados não encontrada em data/raw/Obesity.csv.")
    df_raw = pd.read_csv(DATA_PATH)
    return preprocessar_base(df_raw, coluna_alvo="Obesity")


st.title("Dashboard Analítico")
st.caption("Análises exploratórias da base de pacientes.")

# Carrega dados e mantém a versão traduzida sincronizada.
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

# Garante o IMC disponível para análises numéricas.
if "BMI" not in df.columns and "Height" in df.columns and "Weight" in df.columns:
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Cria colunas traduzidas para filtros e visualizações.
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

st.subheader("Filtros")
filtro_col1, filtro_col2 = st.columns(2)

with filtro_col1:
    niveis_disponiveis = ORDEM_NIVEIS if "Nivel_Obesidade_PT" in df.columns else []
    niveis_selecionados = st.multiselect(
        "Nível de obesidade",
        options=niveis_disponiveis,
        default=niveis_disponiveis,
    )

with filtro_col2:
    generos_disponiveis = (
        sorted(df["Genero_PT"].dropna().unique().tolist())
        if "Genero_PT" in df.columns
        else []
    )
    generos_selecionados = st.multiselect(
        "Gênero",
        options=generos_disponiveis,
        default=generos_disponiveis,
    )

# Aplica filtros para reduzir o conjunto exibido.
df_vis = df.copy()
if niveis_selecionados:
    df_vis = df_vis[df_vis["Nivel_Obesidade_PT"].isin(niveis_selecionados)]
if generos_selecionados:
    df_vis = df_vis[df_vis["Genero_PT"].isin(generos_selecionados)]
if df_vis.empty:
    st.warning("Nenhum registro com os filtros selecionados. Exibindo base completa.")
    df_vis = df

st.caption(f"Exibindo {len(df_vis)} de {len(df)} registros.")

metric_cols = st.columns(4)
metric_cols[0].metric("Registros", len(df_vis))
metric_cols[1].metric(
    "Idade média",
    f"{df_vis['Age'].mean():.1f}" if "Age" in df_vis.columns else "-",
)
metric_cols[2].metric(
    "IMC médio",
    f"{df_vis['BMI'].mean():.1f}" if "BMI" in df_vis.columns else "-",
)
metric_cols[3].metric(
    "Peso médio (kg)",
    f"{df_vis['Weight'].mean():.1f}" if "Weight" in df_vis.columns else "-",
)

# Tabs para organizar o excesso de variáveis na tela.
tab_resumo, tab_numericas, tab_categoricas, tab_relacoes = st.tabs(
    ["Resumo", "Variáveis numéricas", "Variáveis categóricas", "Relações"]
)

with tab_resumo:
    col1, col2 = st.columns(2)

    with col1:
        if "Nivel_Obesidade_PT" in df_vis.columns:
            distribuicao = (
                df_vis["Nivel_Obesidade_PT"]
                .value_counts()
                .reindex(ORDEM_NIVEIS)
                .fillna(0)
                .reset_index()
            )
            distribuicao.columns = ["Nível", "Quantidade"]
            distribuicao = distribuicao[distribuicao["Quantidade"] > 0]
            fig_niveis = px.pie(
                distribuicao,
                names="Nível",
                values="Quantidade",
                title="Distribuição dos níveis de obesidade",
            )
            fig_niveis.update_traces(textposition="inside", textinfo="percent+label")
            fig_niveis.update_layout(legend_title_text="Nível de obesidade")
            st.plotly_chart(fig_niveis, use_container_width=True)
        else:
            st.info("Coluna de nível de obesidade não disponível.")

    with col2:
        if "Genero_PT" in df_vis.columns:
            distribuicao_genero = df_vis["Genero_PT"].value_counts().reset_index()
            distribuicao_genero.columns = ["Gênero", "Quantidade"]
            fig_genero = px.bar(
                distribuicao_genero,
                x="Gênero",
                y="Quantidade",
                text="Quantidade",
                title="Distribuição por gênero",
                labels={"Gênero": "Gênero", "Quantidade": "Quantidade"},
            )
            fig_genero.update_traces(textposition="outside")
            fig_genero.update_layout(yaxis_title="Quantidade", xaxis_title="Gênero")
            st.plotly_chart(fig_genero, use_container_width=True)
        else:
            st.info("Coluna de gênero não disponível.")

with tab_numericas:
    colunas_numericas = [c for c in ROTULOS_NUMERICOS if c in df_vis.columns]
    if not colunas_numericas:
        st.info("Sem variáveis numéricas disponíveis para análise.")
    else:
        coluna_escolhida = st.selectbox(
            "Variável numérica",
            colunas_numericas,
            format_func=lambda col: ROTULOS_NUMERICOS.get(col, col),
        )
        comparar_niveis = st.checkbox(
            "Comparar por nível de obesidade",
            value=True,
            key="comparar_numericas",
        )

        n_bins = 20
        valores_unicos = df_vis[coluna_escolhida].dropna().unique()
        if 0 < len(valores_unicos) <= 10:
            n_bins = len(valores_unicos)

        fig_numerica = px.histogram(
            df_vis,
            x=coluna_escolhida,
            color="Nivel_Obesidade_PT" if comparar_niveis else None,
            nbins=n_bins,
            barmode="overlay" if comparar_niveis else "group",
            opacity=0.7 if comparar_niveis else 1.0,
            title=f"Distribuição de {ROTULOS_NUMERICOS.get(coluna_escolhida, coluna_escolhida)}",
            labels=ROTULOS_EIXOS,
        )
        if comparar_niveis:
            fig_numerica.update_layout(legend_title_text="Nível de obesidade")
        fig_numerica.update_layout(yaxis_title="Quantidade")
        st.plotly_chart(fig_numerica, use_container_width=True)

with tab_categoricas:
    colunas_categoricas = [
        c
        for c in ROTULOS_CATEGORICOS
        if c in df_vis.columns and c != "Nivel_Obesidade_PT"
    ]
    if not colunas_categoricas:
        st.info("Sem variáveis categóricas disponíveis para análise.")
    else:
        coluna_escolhida = st.selectbox(
            "Variável categórica",
            colunas_categoricas,
            format_func=lambda col: ROTULOS_CATEGORICOS.get(col, col),
        )
        comparar_niveis = st.checkbox(
            "Detalhar por nível de obesidade",
            value=True,
            key="comparar_categoricas",
        )

        if comparar_niveis and "Nivel_Obesidade_PT" in df_vis.columns:
            agrupado = (
                df_vis.groupby([coluna_escolhida, "Nivel_Obesidade_PT"])
                .size()
                .reset_index(name="Quantidade")
            )
            fig_cat = px.bar(
                agrupado,
                x=coluna_escolhida,
                y="Quantidade",
                color="Nivel_Obesidade_PT",
                barmode="stack",
                title=(
                    f"{ROTULOS_CATEGORICOS.get(coluna_escolhida, coluna_escolhida)}"
                    " por nível de obesidade"
                ),
                labels=ROTULOS_EIXOS,
            )
            fig_cat.update_layout(legend_title_text="Nível de obesidade")
        else:
            distribuicao = df_vis[coluna_escolhida].value_counts().reset_index()
            distribuicao.columns = ["Categoria", "Quantidade"]
            fig_cat = px.bar(
                distribuicao,
                x="Categoria",
                y="Quantidade",
                text="Quantidade",
                title=(
                    f"Distribuição de {ROTULOS_CATEGORICOS.get(coluna_escolhida, coluna_escolhida)}"
                ),
                labels={
                    "Categoria": ROTULOS_CATEGORICOS.get(coluna_escolhida, coluna_escolhida),
                    "Quantidade": "Quantidade",
                },
            )
            fig_cat.update_traces(textposition="outside")

        fig_cat.update_layout(yaxis_title="Quantidade")
        st.plotly_chart(fig_cat, use_container_width=True)

with tab_relacoes:
    col1, col2 = st.columns(2)

    with col1:
        if "Height" in df_vis.columns and "Weight" in df_vis.columns:
            fig_dispersao = px.scatter(
                df_vis,
                x="Height",
                y="Weight",
                color="Nivel_Obesidade_PT" if "Nivel_Obesidade_PT" in df_vis.columns else None,
                title="Relação entre altura e peso",
                labels={
                    "Height": "Altura (m)",
                    "Weight": "Peso (kg)",
                    "Nivel_Obesidade_PT": "Nível de obesidade",
                },
                opacity=0.7,
            )
            fig_dispersao.update_layout(legend_title_text="Nível de obesidade")
            st.plotly_chart(fig_dispersao, use_container_width=True)
        else:
            st.info("Colunas de altura/peso não disponíveis para correlação.")

    with col2:
        num_cols = [
            c
            for c in ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
            if c in df_vis.columns
        ]
        if len(num_cols) >= 2:
            corr = df_vis[num_cols].corr(numeric_only=True)
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
