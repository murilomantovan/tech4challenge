# Modelo

## Estratégia
- Problema: **classificação multiclasse** (nível de obesidade)
- Métrica principal: **Acurácia** (requisito do desafio: > 75%)
- Métricas de apoio: precision/recall/F1 por classe

## Pipeline (scikit-learn + imblearn)
- Separação numéricas vs categóricas
- Numéricas: `MinMaxScaler`
- Categóricas: `OneHotEncoder(handle_unknown="ignore")`
- Balanceamento: `SMOTE`
- Modelo final: `RandomForestClassifier`

## Saídas
- `models/modelo_obesidade.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`

Treino:
```bash
python -m src.obesity_tc.train --data data/raw/Obesity.csv --target Obesity --model_out models/modelo_obesidade.joblib
```
