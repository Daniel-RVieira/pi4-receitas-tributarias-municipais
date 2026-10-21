# Plataforma de Análise e Projeção de Receitas Tributárias Municipais com Aprendizado de Máquina

Projeto Integrador IV — UNIVESP — Grupo 6 (Ciência de Dados e Engenharia da Computação).

Sistema web que classifica e projeta a arrecadação de 644 municípios do Estado de São Paulo
(todos, exceto a capital) em três grupos — receita tributária própria, transferências
constitucionais e demais receitas não tributárias —, usando modelos de projeção da
Secretaria do Tesouro Nacional (Modelo Sazonal a(t-12) e Modelo Média (t-1)) combinados a
uma regressão linear (scikit-learn) para estimar o fator de crescimento anual.

Fonte dos dados brutos: [Portal de Dados Abertos do TCE-SP](https://transparencia.tce.sp.gov.br/conjunto-de-dados), conjunto "Receitas" (2021-2025).

## Rodando localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Os dados já processados ficam em `data/processed/` (incluídos neste repositório). Para
reprocessar a partir dos dados brutos do TCE-SP:

1. Baixe `receitas-2021.zip` a `receitas-2025.zip` no Portal de Dados Abertos do TCE-SP e
   coloque-os na raiz do projeto.
2. Rode `python src/etl.py`.

## Deploy gratuito (Streamlit Community Cloud)

1. Faça login em [share.streamlit.io](https://share.streamlit.io) com sua conta GitHub.
2. Clique em "New app", selecione este repositório, branch `main` e arquivo `app.py`.
3. O deploy usa automaticamente o `requirements.txt` do repositório.

## Estrutura

- `app.py` — interface Streamlit
- `src/etl.py` — classificação dos lançamentos de receita (Natureza de Receita, MCASP/STN)
- `src/modelo.py` — modelos de projeção (Sazonal a(t-12))
- `src/inflacao.py` — projeção das demais receitas (Média (t-1) + IPCA)
- `data/processed/` — dados agregados já classificados
