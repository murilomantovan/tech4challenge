import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Colunas discretas que chegam com ruído decimal e precisam de arredondamento.
COLUNAS_DISCRETAS_ARREDONDAR = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]

COLUNAS_PT_BR = {
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
    "Obesity": "Nível de obesidade",
    "Obesity_level": "Nível de obesidade",
    "BMI": "IMC",
}

MAPA_GENERO_PT = {"Female": "Feminino", "Male": "Masculino"}
MAPA_SIM_NAO_PT = {"yes": "Sim", "no": "Não"}
MAPA_FREQUENCIA_PT = {
    "no": "não",
    "Sometimes": "às vezes",
    "Frequently": "frequentemente",
    "Always": "sempre",
}
MAPA_TRANSPORTE_PT = {
    "Public_Transportation": "Transporte público",
    "Automobile": "Automóvel",
    "Walking": "Caminhada",
    "Motorbike": "Motocicleta",
    "Bike": "Bicicleta",
}
MAPA_NIVEL_OBESIDADE_PT = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}


def calcular_imc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Evita divisão por zero.
    altura = df["Height"].replace(0, np.nan)
    df["BMI"] = df["Weight"] / (altura**2)
    return df


def preprocessar_base(df: pd.DataFrame, coluna_alvo: str = "Obesity") -> pd.DataFrame:
    df = df.copy()

    # Remove espaços extras em strings.
    for coluna in df.columns:
        if df[coluna].dtype == "object":
            df[coluna] = df[coluna].astype(str).str.strip()

    # Arredondamento de variáveis discretas com ruído decimal.
    for coluna in COLUNAS_DISCRETAS_ARREDONDAR:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").round(0).astype(
                "Int64"
            )

    # Calcula o IMC para uso no treino e análises.
    df = calcular_imc(df)

    # Normaliza nome do alvo (interno).
    if coluna_alvo in df.columns:
        df = df.rename(columns={coluna_alvo: "Obesity_level"})
    return df


def traduzir_ptbr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Traduz valores categóricos para PT-BR.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(MAPA_GENERO_PT).fillna(df["Gender"])

    for coluna in ["family_history", "FAVC", "SMOKE", "SCC"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].map(MAPA_SIM_NAO_PT).fillna(df[coluna])

    for coluna in ["CAEC", "CALC"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].map(MAPA_FREQUENCIA_PT).fillna(df[coluna])

    if "MTRANS" in df.columns:
        df["MTRANS"] = df["MTRANS"].map(MAPA_TRANSPORTE_PT).fillna(df["MTRANS"])

    for coluna in ["Obesity", "Obesity_level"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].map(MAPA_NIVEL_OBESIDADE_PT).fillna(df[coluna])

    # Traduz nomes das colunas para facilitar visualização.
    df = df.rename(columns={k: v for k, v in COLUNAS_PT_BR.items() if k in df.columns})
    return df


def salvar_base_ptbr(
    df: pd.DataFrame,
    output_path: Path = Path("data/processed/base_traduzida_ptbr.csv"),
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Salva base traduzida em UTF-8 para uso no dashboard.
    df_traduzido = traduzir_ptbr(df)
    df_traduzido.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def atualizar_base_ptbr(
    data_path: Path = Path("data/raw/Obesity.csv"),
    output_path: Path = Path("data/processed/base_traduzida_ptbr.csv"),
    coluna_alvo: str = "Obesity",
):
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not data_path.exists():
        return None
    # Evita retrabalho se a base traduzida já estiver atualizada.
    if output_path.exists() and output_path.stat().st_mtime >= data_path.stat().st_mtime:
        return output_path
    df_raw = pd.read_csv(data_path)
    df_processado = preprocessar_base(df_raw, coluna_alvo=coluna_alvo)
    return salvar_base_ptbr(df_processado, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="Obesity")
    parser.add_argument(
        "--output_ptbr",
        default="data/processed/base_traduzida_ptbr.csv",
        help="Caminho para salvar a base traduzida (UTF-8).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_processado = preprocessar_base(df, coluna_alvo=args.target)
    df_processado.to_csv(args.output, index=False, encoding="utf-8")
    if args.output_ptbr:
        salvar_base_ptbr(df_processado, args.output_ptbr)
    print(
        f"OK: salvou {args.output} com {df_processado.shape[0]} linhas e "
        f"{df_processado.shape[1]} colunas."
    )


if __name__ == "__main__":
    main()
