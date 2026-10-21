"""
Modelos de projecao de receita, aplicados separadamente a cada grupo
(tributaria, transferencia_constitucional, outras), seguindo a tipologia de
modelos de projecao de receitas orcamentarias da STN (BRASIL, 2023):

  - Receita tributaria propria       -> Modelo Sazonal a(t-12)
  - Transferencias constitucionais   -> Modelo Sazonal a(t-12)
  - Outras receitas nao tributarias  -> Modelo Media (t-1), em inflacao.py

No Modelo Sazonal a(t-12), cada mes e projetado multiplicando a arrecadacao
do mesmo mes do ano anterior por um fator de crescimento anual. A gente
estima esse fator com uma regressao linear (scikit-learn) sobre os totais
anuais historicos de cada grupo/municipio - ele acaba juntando, de forma
agregada, o efeito da inflacao e o crescimento real observados entre
2021-2025, que e basicamente o que a STN recomenda fazer (corrigir a
arrecadacao do ano-base pelos indices de preco, quantidade e legislacao
acumulados).

Optamos pelo Sazonal a(t-12) pra receita tributaria e transferencias porque
essas receitas tem bastante concentracao sazonal em meses especificos (cotas
do IPTU no inicio do ano, FPM com cotas extras em julho/dezembro, ITR
concentrado - os mesmos exemplos que aparecem no documento da STN).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_PATH = BASE_DIR / "data" / "processed" / "receitas_agregadas.csv"
DADOS_TRIBUTO_PATH = BASE_DIR / "data" / "processed" / "receitas_por_tributo.csv"

GRUPOS_SAZONAL = ["tributaria", "transferencia_constitucional"]

# Ordem em que os tributos aparecem na tabela comparativa: tributaria propria
# primeiro, depois transferencias constitucionais, e "outras" por ultimo.
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
    """Roda uma regressao linear sobre os totais anuais historicos pra
    estimar quanto cada ano projetado deve crescer em relacao ao anterior
    (preco + quantidade juntos, sem separar os dois efeitos)."""
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
    """Modelo Sazonal a(t-12): projeta cada mes a partir do mesmo mes do ano
    anterior, multiplicado pelo fator de crescimento anual. E recursivo - a
    partir do segundo ano do horizonte, usa os proprios valores ja
    projetados como base a(t-12)."""
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
    """Roda o Modelo Sazonal a(t-12) num grupo de receita (tributaria ou
    transferencia_constitucional) de um municipio. Devolve (historico, projecao)."""
    serie = _serie_mensal(df_municipio, coluna)
    historico = pd.DataFrame({"data": serie.index, "valor": serie.values})
    projecao = projetar_sazonal_at12(serie, meses_futuros)
    return historico, projecao


def carregar_dados_por_tributo() -> pd.DataFrame:
    df = pd.read_csv(DADOS_TRIBUTO_PATH)
    df["data"] = pd.to_datetime(dict(year=df["ano"], month=df["mes"], day=1))
    return df.sort_values(["municipio", "categoria", "tributo", "data"])


def projetar_tributo(serie_mensal: pd.Series, categoria: str, meses_futuros: int):
    """Projeta um tributo especifico com o modelo certo pra categoria dele:
    Sazonal a(t-12) pra tributaria/transferencias, Media (t-1) + IPCA (sem
    ML) pro bloco agregado de outras receitas."""
    if categoria == "outras":
        from inflacao import projetar_outras_receitas
        datas_futuras = pd.date_range(serie_mensal.index[-1] + pd.offsets.MonthBegin(1),
                                       periods=meses_futuros, freq="MS")
        valores = projetar_outras_receitas(serie_mensal, meses_futuros)
        return pd.DataFrame({"data": datas_futuras, "valor": valores.values})
    return projetar_sazonal_at12(serie_mensal, meses_futuros)


def treinar_e_projetar(df_municipio: pd.DataFrame, meses_futuros: int = 24):
    """Mantida por compatibilidade com versoes anteriores do app: soma as
    projecoes de tributaria e transferencia_constitucional, cada uma com seu
    proprio Sazonal a(t-12) (em vez de rodar um unico modelo sobre a soma)."""
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
