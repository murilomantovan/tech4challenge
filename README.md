# Sistema de Predição de Obesidade — Tech Challenge Fase 4

Este projeto apresenta uma solução completa de Machine Learning aplicada à predição do nível de obesidade. A proposta une tratamento de dados, modelagem e entrega em uma aplicação web, com foco em apoiar a interpretação clínica e a tomada de decisão.

## Objetivo

O objetivo é construir um sistema preditivo que auxilie profissionais de saúde a estimar o nível de obesidade de um paciente a partir de variáveis de perfil, hábitos alimentares e comportamento. Além da predição individual, o projeto oferece um painel de análise visual para explorar padrões e relações entre variáveis.

## Dados e tratamento

A base utilizada reúne informações demográficas e de estilo de vida. O processamento inclui limpeza de texto, arredondamento de variáveis discretas que podem vir com ruído, criação do índice de massa corporal e tradução para português para facilitar a leitura no dashboard. Essas etapas garantem consistência entre o que é treinado e o que é exibido na aplicação.

## Modelo e pipeline

A modelagem é feita por meio de um pipeline que organiza o pré-processamento e o treinamento em uma sequência única. O fluxo contempla a preparação de variáveis numéricas e categóricas, normalização, balanceamento de classes e um modelo de classificação robusto para lidar com múltiplos níveis de obesidade. Esse formato torna o processo reproduzível e reduz divergências entre treino e produção.

## Aplicação e dashboard

A aplicação em Streamlit permite a predição interativa a partir do preenchimento de um formulário simples. O dashboard analítico complementa a solução ao reunir gráficos que destacam distribuição das classes, relações entre medidas corporais e hábitos associados ao risco.

## Deploy

O projeto está preparado para execução em ambiente de nuvem com caminhos relativos, o que garante que dados, modelos e relatórios sejam encontrados de forma consistente, independentemente de onde o app é iniciado.

## Conclusão

A entrega consolida um fluxo end-to-end que transforma dados em informação acionável, combinando ciência de dados, visão de negócio e suporte à área da saúde.
