"""
Modelos de projecao de receitas, aplicados separadamente a cada um dos tres
grupos de receita (tributaria, transferencia_constitucional, outras),
seguindo a tipologia de modelos de projecao de receitas orcamentarias da
Secretaria do Tesouro Nacional (BRASIL, 2023):

  - Receita tributaria propria       -> Modelo Sazonal a(t-12)
  - Transferencias constitucionais   -> Modelo Sazonal a(t-12)
  - Outras receitas nao tributarias  -> Modelo Media (t-1), em inflacao.py

O Modelo Sazonal a(t-12) projeta cada mes multiplicando a arrecadacao do
mesmo mes do ano anterior por um fator de crescimento anual, estimado por
regressao linear (scikit-learn) sobre os totais anuais historicos de cada
grupo de receita e de cada municipio. Esse fator incorpora, de forma
agregada, os efeitos de preco (inflacao) e de quantidade (crescimento real)
observados no periodo historico (2021-2025), tal como descrito no modelo
original da STN, que recomenda a correcao da arrecadacao do ano-base por
indices de preco, quantidade e legislacao acumulados ate o mes projetado.

A escolha do Modelo Sazonal a(t-12) para a receita tributaria e para as
transferencias constitucionais justifica-se pela forte concentracao sazonal
dessas receitas em meses especificos (ex.: cotas do IPTU no inicio do ano,
cotas adicionais do FPM em julho e dezembro, concentracao do ITR), tal como
ilustrado pelos proprios exemplos do documento da STN.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_PATH = BASE_DIR / "data" / "processed" / "receitas_agregadas.csv"
DADOS_TRIBUTO_PATH = BASE_DIR / "data" / "processed" / "receitas_por_tributo.csv"

GRUPOS_SAZONAL = ["tributaria", "transferencia_constitucional"]

# Ordem de exibicao dos tributos na tabela comparativa (STN, 2023): primeiro a
# receita tributaria propria, depois as transferencias constitucionais e, por
# fim, o bloco agregado de outras receitas nao tributarias.
ORDEM_TRIBUTOS = [
    "IPTU", "ISS", "ITBI", "IRRF", "Taxas", "Contribuição de Melhoria", "COSIP",
    "ITR (arrecadação direta)", "Outros impostos",
    "FPM", "ICMS", "IPVA", "ITR (cota-parte)", "IOF-Ouro", "IPI-Exportação",
    "CIDE-Combustíveis", "Outras participações",
    "Outras receitas não tributárias",
]


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DADOS_PATH)
    df["receita_modelada"] = df["tributaria"] + df["transferencia_constitucional"]
    df["data"] = pd.to_datetime(dict(year=df["ano"], month=df["mes"], day=1))
    return df.sort_values(["municipio", "data"])


def listar_municipios(df: pd.DataFrame) -> list:
    return sorted(df["municipio"].unique().tolist())


def _serie_mensal(df_municipio: pd.DataFrame, coluna: str) -> pd.Series:
    serie = df_municipio.set_index("data")[coluna].asfreq("MS")
    return serie.interpolate(limit_direction="both")


def _fatores_crescimento_anual(serie_mensal: pd.Series, anos_futuros: list) -> dict:
    """Estima, por regressao linear sobre os totais anuais historicos
    (scikit-learn), o fator de crescimento de cada ano projetado em relacao
    ao ano imediatamente anterior (efeito conjunto de preco e quantidade)."""
    totais = serie_mensal.groupby(serie_mensal.index.year).sum()
    anos_hist = totais.index.values.reshape(-1, 1)
    modelo = LinearRegression().fit(anos_hist, totais.values)

    fatores = {}
    for ano in sorted(set(anos_futuros)):
        previsto_atual = max(modelo.predict([[ano]])[0], 1e-6)
        previsto_ano_anterior = max(modelo.predict([[ano - 1]])[0], 1e-6)
        fatores[ano] = previsto_atual / previsto_ano_anterior
    return fatores


def projetar_sazonal_at12(serie_mensal: pd.Series, meses_futuros: int) -> pd.DataFrame:
    """Modelo Sazonal a(t-12) (STN, 2023): projeta cada mes a partir do valor
    do mesmo mes do ano anterior, corrigido pelo fator de crescimento anual
    estimado por regressao linear. Aplica-se recursivamente, usando os
    proprios valores projetados como base a(t-12) para o segundo ano em
    diante do horizonte de projecao."""
    ultima_data = serie_mensal.index[-1]
    datas_futuras = pd.date_range(ultima_data + pd.offsets.MonthBegin(1), periods=meses_futuros, freq="MS")
    anos_futuros = sorted(set(datas_futuras.year))
    fatores = _fatores_crescimento_anual(serie_mensal, anos_futuros)

    serie_estendida = serie_mensal.copy()
    valores_projetados = []
    for data in datas_futuras:
        data_base = data - pd.DateOffset(years=1)
        valor_base = serie_estendida.get(data_base)
        if valor_base is None or pd.isna(valor_base):
            valor_base = serie_estendida.tail(12).mean()
        valor_projetado = max(valor_base * fatores[data.year], 0.0)
        serie_estendida.loc[data] = valor_projetado
        valores_projetados.append(valor_projetado)

    return pd.DataFrame({"data": datas_futuras, "valor": valores_projetados})


def treinar_e_projetar_grupo(df_municipio: pd.DataFrame, coluna: str, meses_futuros: int = 24):
    """Aplica o Modelo Sazonal a(t-12) a um grupo de receita (tributaria ou
    transferencia_constitucional) de um municipio. Retorna (historico, projecao)."""
    serie = _serie_mensal(df_municipio, coluna)
    historico = pd.DataFrame({"data": serie.index, "valor": serie.values})
    projecao = projetar_sazonal_at12(serie, meses_futuros)
    return historico, projecao


def carregar_dados_por_tributo() -> pd.DataFrame:
    df = pd.read_csv(DADOS_TRIBUTO_PATH)
    df["data"] = pd.to_datetime(dict(year=df["ano"], month=df["mes"], day=1))
    return df.sort_values(["municipio", "categoria", "tributo", "data"])


def projetar_tributo(serie_mensal: pd.Series, categoria: str, meses_futuros: int):
    """Projeta a serie mensal de um tributo especifico pelo modelo adequado a
    sua categoria: Modelo Sazonal a(t-12) para receita tributaria propria e
    transferencias constitucionais; Modelo Media (t-1) + IPCA, sem aprendizado
    de maquina, para o bloco agregado de outras receitas nao tributarias."""
    if categoria == "outras":
        from inflacao import projetar_outras_receitas
        datas_futuras = pd.date_range(serie_mensal.index[-1] + pd.offsets.MonthBegin(1),
                                       periods=meses_futuros, freq="MS")
        valores = projetar_outras_receitas(serie_mensal, meses_futuros)
        return pd.DataFrame({"data": datas_futuras, "valor": valores.values})
    return projetar_sazonal_at12(serie_mensal, meses_futuros)


def treinar_e_projetar(df_municipio: pd.DataFrame, meses_futuros: int = 24):
    """Mantido por compatibilidade: soma as projecoes de tributaria e de
    transferencia_constitucional, cada uma projetada pelo seu proprio Modelo
    Sazonal a(t-12), em vez de um unico modelo sobre a soma das duas series."""
    hist_trib, proj_trib = treinar_e_projetar_grupo(df_municipio, "tributaria", meses_futuros)
    hist_transf, proj_transf = treinar_e_projetar_grupo(df_municipio, "transferencia_constitucional", meses_futuros)

    historico = pd.DataFrame({
        "data": hist_trib["data"],
        "valor": hist_trib["valor"].values + hist_transf["valor"].values,
    })
    projecao = pd.DataFrame({
        "data": proj_trib["data"],
        "valor": proj_trib["valor"].values + proj_transf["valor"].values,
    })
    return historico, projecao
