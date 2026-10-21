"""
Ajuste por inflacao da serie "outras receitas nao tributarias".

Os dados vem do Banco Central (SGS, serie 433 - IPCA variacao mensal, dados
do IBGE consolidados pelo BCB):
https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json
(acesso em 22 set. 2026)

data/processed/ipca_mensal.csv guarda uma copia local dessa serie. Usamos
ela pra duas coisas: deflacionar "outras receitas" pra preco constante de um
mes de referencia, e projetar essa mesma receita nos meses futuros so
aplicando a variacao de preco mais recente - sem aprendizado de maquina,
que nao faz parte do escopo definido pra essa categoria.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IPCA_PATH = BASE_DIR / "data" / "processed" / "ipca_mensal.csv"


def carregar_ipca() -> pd.DataFrame:
    df = pd.read_csv(IPCA_PATH)
    df["fator_mensal"] = 1 + df["ipca_variacao_pct"] / 100
    df = df.sort_values(["ano", "mes"]).reset_index(drop=True)
    df["indice_acumulado"] = df["fator_mensal"].cumprod()
    return df


def deflacionar_para_referencia(valores: pd.DataFrame, ano_col="ano", mes_col="mes",
                                 valor_col="outras", ano_ref: int = None, mes_ref: int = None) -> pd.Series:
    """Converte 'valor_col' (nominal, por ano/mes) pra reais do mes de
    referencia - por padrao, o mes mais recente que tiver na serie do IPCA."""
    ipca = carregar_ipca()
    if ano_ref is None or mes_ref is None:
        ultima = ipca.iloc[-1]
        ano_ref, mes_ref = int(ultima["ano"]), int(ultima["mes"])

    indice_ref = ipca.loc[(ipca["ano"] == ano_ref) & (ipca["mes"] == mes_ref), "indice_acumulado"].iloc[0]
    ipca_idx = ipca.set_index(["ano", "mes"])["indice_acumulado"]

    merged = valores.merge(
        ipca.rename(columns={"ano": ano_col, "mes": mes_col})[[ano_col, mes_col, "indice_acumulado"]],
        on=[ano_col, mes_col], how="left",
    )
    fator_correcao = indice_ref / merged["indice_acumulado"]
    return merged[valor_col] * fator_correcao


def projetar_outras_receitas(historico_mensal: pd.Series, meses_futuros: int) -> pd.Series:
    """Projeta 'outras receitas nao tributarias' so pela inflacao: pega a
    media dos ultimos 12 meses e aplica a variacao media mensal do IPCA do
    mesmo periodo pra frente."""
    ipca = carregar_ipca()
    variacao_media = ipca["ipca_variacao_pct"].tail(12).mean() / 100
    base = historico_mensal.tail(12).mean()
    projecao = [base * ((1 + variacao_media) ** i) for i in range(1, meses_futuros + 1)]
    return pd.Series(projecao)
