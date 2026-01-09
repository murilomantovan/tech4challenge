📊 Sistema de Predição de Obesidade

Este projeto entrega uma solução completa de Machine Learning aplicada à predição do nível de obesidade, incluindo:

✔️ Pipeline de dados com pré-processamento e engenharia de features
✔️ Modelo preditivo com acurácia superior a 75%
✔️ Deploy em Streamlit para predição interativa
✔️ Dashboard analítico com insights de negócio
✔️ Todas as etapas orientadas para negócio e interpretação médica

🧠 Objetivo

Criar um sistema preditivo que auxilie profissionais de saúde a estimar o nível de obesidade de um paciente com base em variáveis de perfil, hábitos e comportamentos.
O projeto também entrega um painel de análise visual que ajuda a identificar padrões e relações entre variáveis.

🗂 Estrutura do Projeto
obesity_tc_project/
├── data/
│   ├── raw/Obesity.csv
│   └── processed/base_traduzida_ptbr.csv
├── models/
│   └── modelo_obesidade.joblib
├── reports/
│   ├── metrics.json
│   └── classification_report.txt
├── src/obesity_tc/
│   ├── make_dataset.py        # pré-processamento
│   └── train.py               # treinamento do modelo
├── Predicao.py                # app principal Streamlit
├── pages/
│   ├── 1_Dashboard.py         # visualizações e insights
│   └── 3_Metricas.py          # métricas e avaliação
├── requirements.txt
└── README.md

🧪 Dados

A base utilizada é a Obesity.csv, com 2.111 registros e 16 variáveis sobre perfil e hábitos.
A variável alvo é Obesity, com 7 classes ordinais (de muito abaixo do peso até obesidade tipo III).

Principais tratamentos aplicados:

✔ Arredondamento de variáveis discretas que apresentavam ruído decimal
✔ Criação da feature IMC (Índice de Massa Corporal)
✔ Tradução para PT-BR para melhorar a leitura no dashboard

Exemplo de tratamento no código (make_dataset.py):

# arredondamento das variáveis discretas
df["FCVC"] = df["FCVC"].round()
df["NCP"] = df["NCP"].round()
df["CH2O"] = df["CH2O"].round()
df["FAF"] = df["FAF"].round()
df["TUE"] = df["TUE"].round()

# cálculo do IMC
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

🤖 Machine Learning
Pipeline

O modelo é treinado com um pipeline que garante consistência entre treino e produção:

✔ OneHotEncoder para variáveis categóricas
✔ MinMaxScaler para normalização numérica
✔ SMOTE para balanceamento de classes minoritárias
✔ RandomForestClassifier como algoritmo preditivo

Treinamento do modelo (train.py):

pipeline = Pipeline([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])


✔ Acurácia alcançada: ~97,4% > objetivo mínimo de 75%

📊 Métricas

A avaliação do modelo utiliza:

🔹 Matriz de Confusão
🔹 Classification Report (precisão, recall, f1-score)
🔹 Acurácia geral

Exemplo de carregamento dos resultados:

from joblib import load
import json

model = load("models/modelo_obesidade.joblib")
metrics = json.loads(Path("reports/metrics.json").read_text())
print("Acurácia:", metrics["acuracia"])

📈 Dashboard Analítico

O Dashboard (Streamlit) foi pensado para gerar insights visuais úteis para equipes médicas.

Principais seções:

Gráfico	O que mostra
Distribuição de classes	Mix de níveis de obesidade
Gênero	Equilíbrio entre masculino e feminino
Modos de transporte	Padrões de mobilidade
Altura × Peso	Separação visual por classe
Correlação	Relações entre variáveis numéricas
Hidratação, Atividade Física	Comportamentos associados
Alimentação e Hábitos	Insights de rotinas e hábitos
📍 Streamlit — Predição Interativa

Abra o app, preencha os dados do paciente e obtenha uma previsão imediata do nível de obesidade:

🔹 Idade
🔹 Altura / Peso
🔹 Hábitos alimentares e atividade física

O formulário chama internamente:

modelo = joblib.load("models/modelo_obesidade.joblib")
predicao = modelo.predict(dados_do_usuario)

🚀 Deploy

O app está configurado para rodar no Streamlit Cloud com paths relativos, garantindo que:

✔ data/, models/ e reports/ sejam carregados corretamente
✔ o mesmo pré-processamento seja aplicado em produção

🧩 Conclusão

Este projeto entrega pipeline completa + modelo robusto + deploy + dashboard com insights, tudo orientado para tomada de decisão clínica.
Ele evidencia uma abordagem end-to-end que combina ciência de dados com impacto de negócio.
