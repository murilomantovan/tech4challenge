# Tech Challenge (Fase 4) — Data Analytics: Obesity (End-to-End)

Este repositório entrega uma **estrutura completa** para atender aos critérios do desafio:

- Pipeline com **feature engineering + treinamento + avaliação**
- Modelo com **acurácia > 75%** (checada automaticamente no treino)
- **Deploy Streamlit** (app preditivo)
- Estrutura para **dashboard Power BI** (insights + KPIs)
- **Documentação** com MkDocs
- Roteiro de **vídeo (4–10 min)** em `/video/roteiro.md`

> Base utilizada: `data/raw/Obesity.csv` (fornecida no enunciado).

---

## 1) Como rodar localmente

### 1.1 Criar ambiente e instalar dependências
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Gerar base tratada (opcional)
```bash
python -m src.obesity_tc.make_dataset --input data/raw/Obesity.csv --output data/processed/base_processada.csv
```

### 1.3 Treinar e salvar modelo
```bash
python -m src.obesity_tc.train --data data/raw/Obesity.csv --target Obesity --model_out models/modelo_obesidade.joblib
```

Saídas geradas:
- `models/modelo_obesidade.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`

### 1.4 Rodar o app (Streamlit)
```bash
streamlit run Predicao.py
```

---

## 2) Estrutura do projeto (estilo Cookiecutter Data Science)

```
obesity_tc_project/
├─ Predicao.py
├─ requirements.txt
├─ mkdocs.yml
├─ data/
│  ├─ raw/Obesity.csv
│  └─ processed/
├─ models/
├─ reports/
├─ dashboards/
│  └─ powerbi/README.md
├─ docs/
├─ video/
└─ src/
   └─ obesity_tc/
```

---

## 3) Observações importantes

- O alvo do dataset original está na coluna `Obesity` (multiclasse). O treino usa esse alvo como **Obesity_level** (nome interno).
- Feature engineering inclui **IMC (BMI)** e **normalização de variáveis discretas** (arredondamentos).
- O script de treino falha (exit code 1) se a acurácia no conjunto de teste ficar abaixo de **0.75**.

---

## 4) Próximos passos para “entrega final”

- Publicar o app no Streamlit Community Cloud (ou outro host) e colar o link no README.
- Criar o PBIX do Power BI e publicar/compartilhar o link conforme o desafio.
- Gravar o vídeo seguindo `/video/roteiro.md`.
