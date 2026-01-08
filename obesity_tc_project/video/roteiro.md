# Roteiro de vídeo (4–10 minutos)

## 1) Contexto (30–45s)
- Problema: apoiar hospital com um sistema que antecipa nível de obesidade.
- Impacto: triagem e ações preventivas baseadas em hábitos e medidas.

## 2) Dados (45–60s)
- Base Obesity.csv: variáveis de hábitos + demografia.
- Alvo multiclasse: nível de obesidade.

## 3) Pipeline e feature engineering (1–2min)
- Criação de BMI (IMC).
- Arredondamentos de variáveis discretas com ruído decimal.
- OneHot em categóricas, scaling em numéricas.
- SMOTE para balanceamento.

## 4) Modelo e métricas (1–2min)
- Treino com Random Forest.
- Mostrar métricas do `reports/metrics.json` e `classification_report.txt`.
- Confirmar critério do desafio: acurácia > 75%.

## 5) Deploy Streamlit (1–2min)
- Mostrar o app, inputs, previsão e mensagem orientativa.
- Explicar como seria usado pelo time médico/gestores.

## 6) Dashboard Streamlit (1–2min)
- Páginas: KPIs, distribuições, segmentos, correlações.

## 7) Encerramento (15–30s)
- Reforçar entregáveis e próximos passos.
