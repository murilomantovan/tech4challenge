# Dados

## Fonte
- Arquivo: `data/raw/Obesity.csv`

## Colunas
O dataset contém variáveis demográficas, hábitos e um alvo multiclasse (`Obesity`).

## Tratamentos aplicados no pipeline
- Limpeza básica (tipos, nulos, trim)
- Criação de **BMI (IMC)**: `Weight / Height²`
- Arredondamento de variáveis discretas com ruído decimal:
  `FCVC, NCP, CH2O, FAF, TUE`

O processamento pode ser executado via:
```bash
python -m src.obesity_tc.make_dataset --input data/raw/Obesity.csv --output data/processed/base_processada.csv
```
