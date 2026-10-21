"""
ETL das receitas municipais (TCE-SP - Portal de Dados Abertos, 2021-2025).

Classifica cada lancamento de receita orcamentaria, segundo a Natureza de
Receita do MCASP/STN, em tres grupos:

  - tributaria                   : receita tributaria propria do municipio -
                                    Impostos, Taxas e Contribuicoes de Melhoria
                                    (IPTU, ISS, ITBI, IRRF na fonte, taxas,
                                    contribuicao de melhoria), incluindo Divida
                                    Ativa e multas/juros desses tributos, mais a
                                    Contribuicao para o Custeio do Servico de
                                    Iluminacao Publica (COSIP/CIP, art. 149-A CF)
  - transferencia_constitucional : cotas-parte constitucionais devidas aos
                                    municipios (Participacao na Receita da
                                    Uniao/dos Estados: FPM, ITR, IOF-Ouro,
                                    ICMS, IPVA, IPI-Exportacao, CIDE-Combustiveis)
  - outras                       : demais receitas correntes/capital regulares
                                    (patrimonial, servicos, contribuicoes
                                    sociais/economicas, SUS/FNDE/FUNDEB,
                                    agropecuaria, industrial, multas,
                                    indenizacoes etc.)

E descarta (nao entram em nenhum total) as receitas nao recorrentes: emendas
parlamentares individuais, transferencias de convenios, alienacao de bens
(leiloes), operacoes de credito, amortizacao de emprestimos, integralizacao
de capital social e lancamentos Intra-OFSS (movimentacao interna entre
orgaos do mesmo ente, sem ingresso real de recursos).

Os arquivos de 2021 e de 2022-2025 usam layouts de coluna diferentes para o
detalhamento da receita (rubrica/alinea/subalinea x d1/dd2/d3/tipo), mas os
tres primeiros niveis da natureza de receita (categoria/subcategoria/fonte)
sao estaveis nos dois layouts, assim como o texto descritivo dos niveis mais
detalhados (ex.: "Participacao na Receita da Uniao", "Emendas Parlamentares",
"Convenio"). A classificacao usa os codigos estaveis nos tres primeiros
niveis e busca por texto nos niveis de detalhe, o que funciona igualmente
para os dois layouts.

Alem da classificacao em tres grupos, cada lancamento de receita tributaria
propria e de transferencia constitucional e tambem rotulado com o tributo
especifico a que se refere (IPTU, ISS, ITBI, IRRF, Taxas, Contribuicao de
Melhoria, COSIP; FPM, ITR, IOF-Ouro, ICMS, IPVA, IPI-Exportacao,
CIDE-Combustiveis), por busca textual nos niveis de detalhe mais profundos da
Natureza de Receita (rubrica/alinea/subalinea em 2021; d1/dd2/d3 em
2022-2025), ja que o nivel de codificacao desses tributos especificos difere
entre os dois layouts, mas a descricao textual e equivalente nos dois.

Saida:
  data/processed/receitas_agregadas.csv
    Colunas: municipio, ano, mes, tributaria, transferencia_constitucional, outras
  data/processed/receitas_por_tributo.csv
    Colunas: municipio, ano, mes, categoria, tributo, valor
"""
import zipfile
import io
import re
import unicodedata
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "receitas_agregadas.csv"
OUT_TRIBUTO_PATH = BASE_DIR / "data" / "processed" / "receitas_por_tributo.csv"

ANOS = [2021, 2022, 2023, 2024, 2025]

BASE_COLS = ["ano_exercicio", "ds_municipio", "mes_referencia",
             "ds_categoria", "ds_subcategoria", "ds_fonte", "vl_arrecadacao"]

LAYOUT_A = ["ds_rubrica", "ds_alinea", "ds_subalinea"]          # 2021
LAYOUT_B = ["ds_d1", "ds_dd2", "ds_d3", "ds_tipo"]              # 2022-2025

INTRA_OFSS_CATEGORIA = {"70000000", "80000000"}
EXCLUIDA_SUBCATEGORIA = {"21000000", "22000000", "23000000"}
EXCLUIDA_FONTE = {"29100000"}
TRIBUTARIA_SUBCATEGORIA = {"11000000"}
# Contribuicao para o Custeio do Servico de Iluminacao Publica (COSIP/CIP), art.
# 149-A da Constituicao Federal: e uma contribuicao (subcategoria 12000000), nao
# um imposto, mas e receita tributaria propria do municipio, por isso e somada
# ao grupo "tributaria" mesmo estando fora do codigo 11000000.
TRIBUTARIA_FONTE_EXTRA = {"12400000"}


def _cod(serie: pd.Series) -> pd.Series:
    return serie.fillna("").astype(str).str.split(" ", n=1).str[0].str.strip()


def _sem_acento(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def detectar_layout(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fh:
            header = fh.readline().decode("latin1").strip().split(";")
    if "ds_rubrica" in header:
        return LAYOUT_A
    return LAYOUT_B


def classificar_chunk(chunk: pd.DataFrame, detalhe_cols: list) -> pd.Series:
    categoria = _cod(chunk["ds_categoria"])
    subcategoria = _cod(chunk["ds_subcategoria"])
    fonte = _cod(chunk["ds_fonte"])

    texto_detalhe = chunk[detalhe_cols].fillna("").agg(" ".join, axis=1)
    texto_detalhe = texto_detalhe.map(_sem_acento).str.upper()

    classe = pd.Series("outras", index=chunk.index)

    classe[subcategoria.isin(TRIBUTARIA_SUBCATEGORIA) | fonte.isin(TRIBUTARIA_FONTE_EXTRA)] = "tributaria"

    contem_emenda = texto_detalhe.str.contains("EMENDA", regex=False)
    contem_convenio = texto_detalhe.str.contains("CONV", regex=False)
    contem_participacao = texto_detalhe.str.contains("PARTICIPA", regex=False)

    e_transferencia = subcategoria.isin({"17000000", "24000000"})
    classe[e_transferencia & contem_participacao] = "transferencia_constitucional"

    # exclusoes (sobrescrevem qualquer classificacao anterior)
    excluida = (
        categoria.isin(INTRA_OFSS_CATEGORIA)
        | subcategoria.isin(EXCLUIDA_SUBCATEGORIA)
        | fonte.isin(EXCLUIDA_FONTE)
        | (e_transferencia & contem_emenda)
        | (e_transferencia & contem_convenio)
    )
    classe[excluida] = "excluida"
    return classe


def classificar_tributo(chunk: pd.DataFrame, detalhe_cols: list, classe: pd.Series, fonte: pd.Series) -> pd.Series:
    """Rotula cada lancamento de 'tributaria'/'transferencia_constitucional' com
    o tributo especifico, por busca textual (robusta aos dois layouts de
    coluna). Mantem 'outras' como um unico rotulo agregado, pois esse grupo e
    projetado pelo Modelo Media (t-1), sem discriminacao por tributo."""
    texto = chunk[detalhe_cols].fillna("").agg(" ".join, axis=1)
    texto = texto.map(_sem_acento).str.upper()

    tributo = pd.Series("", index=chunk.index)

    m_trib = classe == "tributaria"
    tributo[m_trib & fonte.eq("11200000")] = "Taxas"
    tributo[m_trib & fonte.eq("11300000")] = "Contribuição de Melhoria"
    tributo[m_trib & fonte.eq("12400000")] = "COSIP"
    tributo[m_trib & (tributo == "") & texto.str.contains("TERRITORIAL URBANA")] = "IPTU"
    tributo[m_trib & (tributo == "") & texto.str.contains("TERRITORIAL RURAL")] = "ITR (arrecadação direta)"
    tributo[m_trib & (tributo == "") & texto.str.contains("INTER VIVOS")] = "ITBI"
    tributo[m_trib & (tributo == "") & texto.str.contains("SERVICOS DE QUALQUER NATUREZA")] = "ISS"
    tributo[m_trib & (tributo == "") & texto.str.contains("RETID[AO] NA FONTE", regex=True)] = "IRRF"
    tributo[m_trib & (tributo == "")] = "Outros impostos"

    m_transf = classe == "transferencia_constitucional"
    tributo[m_transf & texto.str.contains("FUNDO DE PARTICIPACAO DOS MUNICIPIOS")] = "FPM"
    tributo[m_transf & (tributo == "") & texto.str.contains("TERRITORIAL RURAL")] = "ITR (cota-parte)"
    tributo[m_transf & (tributo == "") & (
        texto.str.contains("CREDITO, CAMBIO E SEGURO") | texto.str.contains(r"\bOURO\b", regex=True)
    )] = "IOF-Ouro"
    tributo[m_transf & (tributo == "") & texto.str.contains(r"\bICMS\b", regex=True)] = "ICMS"
    tributo[m_transf & (tributo == "") & texto.str.contains(r"\bIPVA\b", regex=True)] = "IPVA"
    tributo[m_transf & (tributo == "") & texto.str.contains(r"\bIPI\b", regex=True)] = "IPI-Exportação"
    tributo[m_transf & (tributo == "") & (
        texto.str.contains("INTERVENCAO NO DOMINIO ECONOMICO") | texto.str.contains(r"\bCIDE\b", regex=True)
    )] = "CIDE-Combustíveis"
    tributo[m_transf & (tributo == "")] = "Outras participações"

    tributo[classe == "outras"] = "Outras receitas não tributárias"
    return tributo


def processar_ano(ano: int):
    zip_path = BASE_DIR / f"receitas-{ano}.zip"
    detalhe_cols = detectar_layout(zip_path)
    print(f"[{ano}] lendo {zip_path.name} (layout: {detalhe_cols}) ...")

    resultados_grupo = []
    resultados_tributo = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fh:
            reader = pd.read_csv(
                io.TextIOWrapper(fh, encoding="latin1"),
                sep=";",
                decimal=",",
                usecols=BASE_COLS + detalhe_cols,
                dtype={
                    "ano_exercicio": "int16",
                    "ds_municipio": "category",
                    "mes_referencia": "int8",
                },
                chunksize=500_000,
            )
            for i, chunk in enumerate(reader):
                classe = classificar_chunk(chunk, detalhe_cols)
                fonte = _cod(chunk["ds_fonte"])
                tributo = classificar_tributo(chunk, detalhe_cols, classe, fonte)
                chunk = chunk.assign(classe=classe, tributo=tributo)
                chunk = chunk[chunk["classe"] != "excluida"]

                agg_grupo = (
                    chunk.groupby(["ds_municipio", "ano_exercicio", "mes_referencia", "classe"], observed=True)
                    ["vl_arrecadacao"].sum().reset_index()
                )
                resultados_grupo.append(agg_grupo)

                agg_tributo = (
                    chunk.groupby(["ds_municipio", "ano_exercicio", "mes_referencia", "classe", "tributo"],
                                  observed=True)["vl_arrecadacao"].sum().reset_index()
                )
                resultados_tributo.append(agg_tributo)
                print(f"  [{ano}] chunk {i + 1} processado ({len(chunk):,} linhas mantidas)")
    return pd.concat(resultados_grupo, ignore_index=True), pd.concat(resultados_tributo, ignore_index=True)


def main():
    partes = [processar_ano(ano) for ano in ANOS]
    dados = pd.concat([p[0] for p in partes], ignore_index=True)
    dados_tributo = pd.concat([p[1] for p in partes], ignore_index=True)

    tabela = dados.pivot_table(
        index=["ds_municipio", "ano_exercicio", "mes_referencia"],
        columns="classe",
        values="vl_arrecadacao",
        aggfunc="sum",
        fill_value=0.0,
        observed=True,
    ).reset_index()

    for col in ["tributaria", "transferencia_constitucional", "outras"]:
        if col not in tabela.columns:
            tabela[col] = 0.0

    tabela = tabela.rename(columns={
        "ds_municipio": "municipio",
        "ano_exercicio": "ano",
        "mes_referencia": "mes",
    })
    tabela = tabela[["municipio", "ano", "mes", "tributaria", "transferencia_constitucional", "outras"]]
    tabela = tabela.sort_values(["municipio", "ano", "mes"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(OUT_PATH, index=False)
    print(f"Arquivo agregado salvo em: {OUT_PATH}  ({len(tabela):,} linhas)")

    tabela_tributo = (
        dados_tributo
        .groupby(["ds_municipio", "ano_exercicio", "mes_referencia", "classe", "tributo"], observed=True)
        ["vl_arrecadacao"].sum().reset_index()
    )
    tabela_tributo = tabela_tributo.rename(columns={
        "ds_municipio": "municipio",
        "ano_exercicio": "ano",
        "mes_referencia": "mes",
        "classe": "categoria",
        "vl_arrecadacao": "valor",
    })
    tabela_tributo = tabela_tributo.sort_values(["municipio", "ano", "mes", "categoria", "tributo"])
    tabela_tributo.to_csv(OUT_TRIBUTO_PATH, index=False)
    print(f"Arquivo por tributo salvo em: {OUT_TRIBUTO_PATH}  ({len(tabela_tributo):,} linhas)")


if __name__ == "__main__":
    main()
