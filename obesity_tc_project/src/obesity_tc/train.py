import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.obesity_tc.make_dataset import preprocessar_base

MAPA_NIVEL_OBESIDADE = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}


def build_pipeline(colunas_numericas, colunas_categoricas, random_state=42):
    pre = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), colunas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None,
    )

    pipe = ImbPipeline(
        steps=[
            ("preprocess", pre),
            ("smote", SMOTE(random_state=random_state)),
            ("model", clf),
        ]
    )
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV bruto (Obesity.csv)")
    parser.add_argument(
        "--target", default="Obesity", help="Nome da coluna alvo no CSV bruto"
    )
    parser.add_argument("--model_out", default="models/modelo_obesidade.joblib")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--min_accuracy",
        type=float,
        default=0.75,
        help="Critério mínimo de acurácia",
    )
    args = parser.parse_args()

    df_bruto = pd.read_csv(args.data)
    df_limpo = preprocessar_base(df_bruto, coluna_alvo=args.target)

    if "Obesity_level" not in df_limpo.columns:
        raise ValueError("Não encontrei a coluna alvo. Verifique --target.")

    alvo = df_limpo["Obesity_level"]
    entradas = df_limpo.drop(columns=["Obesity_level"])

    colunas_numericas = [
        c for c in entradas.columns if pd.api.types.is_numeric_dtype(entradas[c])
    ]
    colunas_categoricas = [c for c in entradas.columns if c not in colunas_numericas]

    entradas_treino, entradas_teste, alvo_treino, alvo_teste = train_test_split(
        entradas,
        alvo,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=alvo,
    )

    pipe = build_pipeline(
        colunas_numericas, colunas_categoricas, random_state=args.random_state
    )
    pipe.fit(entradas_treino, alvo_treino)

    predicoes = pipe.predict(entradas_teste)
    acuracia = float(accuracy_score(alvo_teste, predicoes))

    dir_relatorios = Path("reports")
    dir_relatorios.mkdir(parents=True, exist_ok=True)

    classes_ordenadas = sorted(alvo.unique().tolist())
    classes_pt = [MAPA_NIVEL_OBESIDADE.get(c, c) for c in classes_ordenadas]

    metricas = {
        "acuracia": acuracia,
        "n_treino": int(len(entradas_treino)),
        "n_teste": int(len(entradas_teste)),
        "classes": classes_pt,
        "classes_original": classes_ordenadas,
        "matriz_confusao": confusion_matrix(
            alvo_teste, predicoes, labels=classes_ordenadas
        ).tolist(),
    }

    (dir_relatorios / "metrics.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dir_relatorios / "classification_report.txt").write_text(
        classification_report(
            alvo_teste,
            predicoes,
            labels=classes_ordenadas,
            target_names=classes_pt,
            digits=4,
        ),
        encoding="utf-8",
    )

    # Salva modelo
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "pipeline": pipe,
            "num_cols": colunas_numericas,
            "cat_cols": colunas_categoricas,
            "target": "Obesity_level",
        },
        model_path,
    )

    print(f"OK: acurácia={acuracia:.4f} | modelo salvo em {model_path}")
    print(
        f"Relatórios: {dir_relatorios / 'metrics.json'} e "
        f"{dir_relatorios / 'classification_report.txt'}"
    )

    # Criterio minimo
    if acuracia < args.min_accuracy:
        raise SystemExit(
            f"FALHA: acurácia {acuracia:.4f} < {args.min_accuracy:.2f} "
            "(critério mínimo)"
        )


if __name__ == "__main__":
    main()
