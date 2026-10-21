"""
Ajuste por inflacao da série "outras receitas não tributárias".

Fonte dos dados: Banco Central do Brasil, Sistema Gerenciador de Séries
Temporais (SGS), série 433 - IPCA, variação mensal (%). Dados oficiais do
IBGE consolidados pelo BCB. Disponível em:
https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json
Acesso em: 22 set. 2026.

O arquivo data/processed/ipca_mensal.csv guarda uma copia local dessa serie
(variação percentual mensal do IPCA) usada para converter valores nominais de
"outras receitas não tributárias" em valores a preços constantes de um mês de
referência, e para projetar essas mesmas receitas nos meses futuros aplicando
apenas a variação de preços observada no periodo mais recente (sem uso de
aprendizado de maquina, conforme escopo definido para essa categoria).
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
    """Converte 'valor_col' (nominal, em cada ano/mes) para reais do mes de
    referencia (o mais recente disponivel na serie do IPCA, por padrao)."""
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
    """Projeta a serie 'outras receitas nao tributarias' apenas pela inflacao:
    replica a media dos ultimos 12 meses em termos reais e aplica a variacao
    media mensal do IPCA observada no mesmo periodo para os meses futuros."""
    ipca = carregar_ipca()
    variacao_media = ipca["ipca_variacao_pct"].tail(12).mean() / 100
    base = historico_mensal.tail(12).mean()
    projecao = [base * ((1 + variacao_media) ** i) for i in range(1, meses_futuros + 1)]
    return pd.Series(projecao)
