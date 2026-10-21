"""
Plataforma de Analise e Projecao de Receitas Tributarias Municipais com
Aprendizado de Maquina

Projeto Integrador IV - UNIVESP - Grupo 6
Dados brutos: TCE-SP, Portal de Dados Abertos, conjunto "Receitas" (2021-2025).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from modelo import (  # noqa: E402
    carregar_dados, carregar_dados_por_tributo, listar_municipios,
    treinar_e_projetar_grupo, projetar_tributo, ORDEM_TRIBUTOS,
)
from inflacao import projetar_outras_receitas  # noqa: E402

st.set_page_config(page_title="Receitas Tributárias Municipais - SP", layout="wide")

st.title("Plataforma de Análise e Projeção de Receitas Tributárias Municipais com Aprendizado de Máquina")
st.caption(
    "Projeto Integrador IV · UNIVESP · Grupo 6 — dados do Portal de Dados Abertos do TCE-SP (2021-2025)"
)

with st.sidebar:
    st.header("Filtros")
    df = carregar_dados()
    municipios = listar_municipios(df)
    municipio = st.selectbox("Município", municipios, index=municipios.index("Tatuí") if "Tatuí" in municipios else 0)
    meses_futuros = st.slider("Meses a projetar", min_value=6, max_value=36, value=24, step=6)
    st.markdown("---")
    st.markdown(
        "**Modelo Sazonal a(t-12) (STN, 2023) + regressão (scikit-learn)\n"
        "para o fator de crescimento anual:**\n"
        "- Receita tributária própria (IPTU, ISS, ITBI, IRRF na fonte, taxas, "
        "contribuição de melhoria e COSIP)\n"
        "- Transferências constitucionais (FPM, ITR, IOF-Ouro, ICMS, IPVA, "
        "IPI-Exportação, CIDE-Combustíveis)\n\n"
        "**Modelo Média (t-1) + IPCA (STN, 2023), sem aprendizado de máquina:**\n"
        "- Demais receitas correntes/capital não tributárias\n\n"
        "**Desconsiderado:**\n"
        "- Emendas parlamentares, convênios e receitas de capital não recorrentes "
        "(alienação de bens, operações de crédito, amortização de empréstimos)"
    )

df_mun = df[df["municipio"] == municipio].copy()

hist_trib, proj_trib = treinar_e_projetar_grupo(df_mun, "tributaria", meses_futuros=meses_futuros)
hist_transf, proj_transf = treinar_e_projetar_grupo(df_mun, "transferencia_constitucional", meses_futuros=meses_futuros)

outras_hist = df_mun.set_index("data")["outras"].asfreq("MS").interpolate(limit_direction="both")
outras_proj_valores = projetar_outras_receitas(outras_hist, meses_futuros)
outras_proj = pd.DataFrame({
    "data": pd.date_range(outras_hist.index[-1] + pd.offsets.MonthBegin(1), periods=meses_futuros, freq="MS"),
    "valor": outras_proj_valores.values,
})

col1, col2, col3, col4 = st.columns(4)
col1.metric("Receita tributária própria (últimos 12 meses)",
            f"R$ {hist_trib['valor'].tail(12).sum():,.0f}".replace(",", "."))
col2.metric("Transferências constitucionais (últimos 12 meses)",
            f"R$ {hist_transf['valor'].tail(12).sum():,.0f}".replace(",", "."))
col3.metric("Outras receitas não tributárias (últimos 12 meses)",
            f"R$ {outras_hist.tail(12).sum():,.0f}".replace(",", "."))
col4.metric(f"Projeção acumulada — próximos {meses_futuros} meses (tributária + transferências)",
            f"R$ {(proj_trib['valor'].sum() + proj_transf['valor'].sum()):,.0f}".replace(",", "."))


def grafico_serie(historico, projecao, titulo, nome_projecao):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historico["data"], y=historico["valor"], mode="lines", name="Histórico (2021-2025)"))
    fig.add_trace(go.Scatter(x=projecao["data"], y=projecao["valor"], mode="lines", name=nome_projecao,
                              line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Mês", yaxis_title="R$ arrecadados", legend_title="Série", height=380,
                       title=titulo)
    return fig


st.subheader("Receita tributária própria: histórico e projeção")
st.caption("Modelo Sazonal a(t-12): projeta cada mês a partir do mesmo mês do ano anterior, corrigido pelo "
           "fator de crescimento anual estimado por regressão linear (scikit-learn).")
st.plotly_chart(grafico_serie(hist_trib, proj_trib, "Receita tributária própria", "Projeção (Sazonal a(t-12))"),
                use_container_width=True)

st.subheader("Transferências constitucionais: histórico e projeção")
st.caption("Modelo Sazonal a(t-12), estimado de forma independente da receita tributária própria, pois cada "
           "grupo tem padrão sazonal e taxa de crescimento próprios (ex.: cotas adicionais do FPM em julho e "
           "dezembro).")
st.plotly_chart(grafico_serie(hist_transf, proj_transf, "Transferências constitucionais", "Projeção (Sazonal a(t-12))"),
                use_container_width=True)

st.subheader("Outras receitas não tributárias: histórico e projeção apenas pela inflação (IPCA)")
st.caption("Modelo Média (t-1): projeta pela média dos últimos 12 meses, atualizada apenas pela variação do "
           "IPCA, sem aprendizado de máquina — por não serem receitas diretamente afetadas pela Reforma "
           "Tributária do Consumo.")
st.plotly_chart(grafico_serie(
    pd.DataFrame({"data": outras_hist.index, "valor": outras_hist.values}),
    outras_proj, "Outras receitas não tributárias", "Projeção (Média (t-1) + IPCA)"
), use_container_width=True)

def formatar_moeda(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


st.subheader("Tabela consolidada")
tabela = df_mun[["ano", "mes", "tributaria", "transferencia_constitucional", "outras"]].copy()
tabela = tabela.rename(columns={
    "ano": "Ano",
    "mes": "Mês",
    "tributaria": "Receita tributária própria",
    "transferencia_constitucional": "Transferências constitucionais",
    "outras": "Outras receitas não tributárias",
})
for coluna in ["Receita tributária própria", "Transferências constitucionais", "Outras receitas não tributárias"]:
    tabela[coluna] = tabela[coluna].apply(formatar_moeda)
st.dataframe(tabela, use_container_width=True, hide_index=True)

st.subheader("Tabela comparativa por tributo: últimos 2 anos (histórico) x anos projetados")
st.caption(
    "Cada linha é um tributo ou transferência constitucional específico (classificado pela Natureza de "
    "Receita do MCASP/STN). Receita tributária própria e transferências constitucionais são projetadas "
    "individualmente pelo Modelo Sazonal a(t-12); o bloco agregado \"Outras receitas não tributárias\" é "
    "projetado pelo Modelo Média (t-1) + IPCA, sem aprendizado de máquina."
)

df_tributo = carregar_dados_por_tributo()
df_tributo_mun = df_tributo[df_tributo["municipio"] == municipio]
datas_completas = pd.date_range(df_tributo_mun["data"].min(), df_tributo_mun["data"].max(), freq="MS")

linhas_tributo = []
for tributo in ORDEM_TRIBUTOS:
    sub = df_tributo_mun[df_tributo_mun["tributo"] == tributo]
    if sub.empty:
        continue
    categoria = sub["categoria"].iloc[0]
    serie = sub.set_index("data")["valor"].reindex(datas_completas, fill_value=0.0)

    hist = pd.DataFrame({"data": serie.index, "valor": serie.values})
    hist["ano"] = hist["data"].dt.year
    anual_hist = hist.groupby("ano")["valor"].sum().tail(2)

    projecao = projetar_tributo(serie, categoria, meses_futuros)
    projecao["ano"] = projecao["data"].dt.year
    anual_proj = projecao.groupby("ano")["valor"].sum()

    linha = {"Tributo": tributo}
    for ano, valor in anual_hist.items():
        linha[f"{ano} (hist.)"] = valor
    for ano, valor in anual_proj.items():
        linha[f"{ano} (proj.)"] = valor
    linhas_tributo.append(linha)

tabela_tributos = pd.DataFrame(linhas_tributo).set_index("Tributo")
colunas_ano = sorted(tabela_tributos.columns, key=lambda c: (int(c.split()[0]), "proj." in c))
tabela_tributos = tabela_tributos[colunas_ano].fillna(0.0)
linha_total = tabela_tributos.sum().to_frame().T
linha_total.index = ["Total"]
tabela_tributos = pd.concat([tabela_tributos, linha_total])
for coluna in colunas_ano:
    tabela_tributos[coluna] = tabela_tributos[coluna].apply(formatar_moeda)
st.dataframe(tabela_tributos, use_container_width=True)
st.caption(
    "Anos incompletos (ano corrente, no histórico, ou último ano do horizonte projetado, quando não "
    "múltiplo de 12 meses) somam apenas os meses efetivamente disponíveis na base ou no horizonte "
    "selecionado no filtro \"Meses a projetar\"."
)

st.caption(
    "Metodologia (STN, 2023 - Modelos de Projeção de Receitas): receita tributária própria e "
    "transferências constitucionais são projetadas separadamente pelo Modelo Sazonal a(t-12), com o "
    "fator de crescimento anual estimado por regressão linear (scikit-learn). Demais receitas não "
    "tributárias são projetadas pelo Modelo Média (t-1), atualizado apenas pela variação do IPCA "
    "(Banco Central/IBGE), sem aprendizado de máquina. Emendas parlamentares, convênios e receitas de "
    "capital não recorrentes foram excluídos da base."
)
