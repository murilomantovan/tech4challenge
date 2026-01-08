# Power BI (guia rápido)

## Dataset sugerido
- `data/processed/base_processada.csv` (gerada pelo pipeline)

## KPIs sugeridos
- Total de registros (pacientes)
- IMC (médio, p25/p50/p75)
- % por nível de obesidade
- Segmentação por sexo / histórico familiar / hábitos

## Medidas (DAX) — exemplos
- IMC Médio = AVERAGE(Base[BMI])
- Pacientes = COUNTROWS(Base)
- % Classe = DIVIDE([Pacientes], CALCULATE([Pacientes], ALL(Base[Obesity_level])))

## Publicação
- Publique no Power BI Service e gere um link compartilhável conforme o desafio.
