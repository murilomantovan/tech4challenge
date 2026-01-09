# Tech Challenge (Fase 4) — Data Analytics: Obesidade (End-to-End)

Este repositório entrega uma **estrutura completa** para atender aos critérios do desafio:

- Pipeline com **feature engineering + treinamento + avaliação**
- Modelo com **acurácia mínima de 75%** validada no treino
- **Deploy Streamlit** (app preditivo + páginas de dashboard e métricas)
- **Documentação** com MkDocs
- Roteiro de **vídeo (4–10 min)** em `roteiro_video.md`

> Base utilizada: `obesity_tc_project/data/raw/Obesity.csv` (fornecida no enunciado).

---

## 1) Como rodar localmente

> Todos os comandos abaixo devem ser executados dentro da pasta `obesity_tc_project/`.

### 1.1 Criar ambiente e instalar dependências
```bash
cd obesity_tc_project
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Gerar base tratada (opcional)
```bash
python -m src.obesity_tc.make_dataset --input data/raw/Obesity.csv --output data/processed/base_processada.csv
```

### 1.3 Treinar e salvar o modelo
```bash
python -m src.obesity_tc.train --data data/raw/Obesity.csv --target Obesity --model_out models/modelo_obesidade.joblib
```

Saídas geradas:
- `models/modelo_obesidade.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `data/processed/base_traduzida_ptbr.csv` (caso não exista)

### 1.4 Rodar o app (Streamlit)
```bash
streamlit run Predicao.py
```

Páginas disponíveis no app:
- **Predição** (`Predicao.py`)
- **Dashboard** (`pages/1_Dashboard.py`)
- **Métricas** (`pages/3_Metricas.py`)

---

## 2) Estrutura do projeto

```
tech4challenge/
├─ obesity_tc_project/
│  ├─ Predicao.py
│  ├─ requirements.txt
│  ├─ mkdocs.yml
│  ├─ TechChallenge_Fase4_Entendendo_o_Codigo.ipynb
│  ├─ data/
│  │  ├─ raw/Obesity.csv
│  │  └─ processed/
│  ├─ docs/
│  ├─ models/
│  ├─ notebooks/
│  │  └─ modelo_obesidade_tc.ipynb
│  ├─ pages/
│  ├─ reports/
│  └─ src/
│     └─ obesity_tc/
├─ README.md
├─ roteiro_video.md
└─ runtime.txt
```

---

## 3) Notebooks disponíveis

- `obesity_tc_project/TechChallenge_Fase4_Entendendo_o_Codigo.ipynb`: guia passo a passo do código e das ideias por trás do projeto.
- `obesity_tc_project/notebooks/modelo_obesidade_tc.ipynb`: notebook de treinamento e análise do modelo.

---

## 4) Observações importantes

- O alvo do dataset original está na coluna `Obesity`; o treino normaliza esse alvo para `Obesity_level`.
- A engenharia de atributos inclui **IMC (BMI)** e arredondamento das variáveis discretas.
- O script de treino encerra com erro se a acurácia no teste ficar abaixo de **0.75**.
