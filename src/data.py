"""
Carregamento e pre-processamento dos dados eleitorais (TSE, Ceara, 2012-2024).

Funcionalidades:
- Leitura e concatenacao dos CSVs por ano
- Normalizacao de tokens historicos do TSE (ex: '#NE#')
- Criacao da variavel-alvo binaria (Eleito / Nao Eleito)
- Codificacao ordinal de features categoricas sem vazamento de dados
"""

from typing import List, Tuple
import unicodedata
import pandas as pd
import numpy as np

# Re-exportamos as chaves categóricas para uso no restante das bibliotecas.
from config import COLUNAS_CATEGORICAS


# AUDITORIA [FASE 2]: Mapeamento canônico do token histórico '#NE#' do TSE (2012).
# Antes de 2016, o TSE usava '#NE#' para indicar raça não especificada.
# Esse token é normalizado para 'Não Informado' para garantir consistência
# semântica entre anos e evitar que 675 candidatos de 2012 sejam tratados
# como uma categoria fantasma na análise de fairness racial.
MAPEAMENTO_RACA_HISTORICO = {"#NE#": "Não Informado"}

# Conjunto canonico de valores esperados para colunas de fairness.
# Qualquer valor fora desse conjunto apos normalizacao aciona um assert.
VALORES_ESPERADOS_FAIRNESS = {
    "Gênero"  : {"Masculino", "Feminino"},
    "Cor/raça": {"Branca", "Parda", "Preta", "Amarela", "Indígena", "Não Informado"},
}


def _normalizar_categoria(valor) -> str:
    """
    Normaliza strings categóricas: aplica NFC unicode, remove espaços marginais
    e aplica o mapeamento histórico de tokens do TSE.
    """
    if pd.isna(valor):
        return valor
    normalizado = unicodedata.normalize("NFC", str(valor)).strip()
    return MAPEAMENTO_RACA_HISTORICO.get(normalizado, normalizado)


# Fontes de dados eleitorais (caminho, encoding, ano).
FONTES_PROCESSAMENTO = [
    ("data/prefeito_ceara2024.csv", "latin1", 2024),
    ("data/prefeito_ceara2020.csv", "utf-8", 2020),
    ("data/prefeito_ceara2016.csv", "utf-8", 2016),
    ("data/prefeito_ceara2012.csv", "utf-8", 2012),
]


def carregar_dados_brutos() -> pd.DataFrame:
    """
    Le os CSVs do TSE para os pleitos de 2012, 2016, 2020 e 2024,
    normaliza tokens historicos e retorna o DataFrame concatenado.
    """
    blocos_anuais = []

    for caminho, padrao_texto, ano in FONTES_PROCESSAMENTO:
        bloco = pd.read_csv(caminho, encoding=padrao_texto, sep=";")
        bloco["eleicao"] = ano

        # Normaliza colunas de fairness para garantir consistencia entre anos.
        for col_fairness in ["Cor/raça", "Gênero"]:
            if col_fairness in bloco.columns:
                bloco[col_fairness] = bloco[col_fairness].apply(_normalizar_categoria)

        # Remove coluna de Regiao (experimento restrito ao Ceara).
        if "Região" in bloco.columns:
            bloco.drop("Região", axis=1, inplace=True)

        blocos_anuais.append(bloco)

    volume_completo = pd.concat(blocos_anuais, ignore_index=True)

    # Assert de integridade: verifica valores inesperados nas colunas de fairness.
    for coluna, valores_ok in VALORES_ESPERADOS_FAIRNESS.items():
        if coluna in volume_completo.columns:
            encontrados = set(volume_completo[coluna].dropna().unique())
            inesperados = encontrados - valores_ok
            assert not inesperados, (
                f"[ASSERT] {coluna} contem valores inesperados: "
                f"{inesperados}: verificar encoding dos CSVs."
            )

    # Politica de missing values: apenas 'Regiao' tem NaN estrutural.
    n_nan = volume_completo[COLUNAS_CATEGORICAS].isnull().sum().sum()
    print(f"[DATA] NaN em colunas categoricas: {n_nan}", flush=True)

    print(f"Candidaturas: {volume_completo.shape[0]} instancias.", flush=True)
    print(f"Por ano: {volume_completo['eleicao'].value_counts().to_dict()}", flush=True)

    return volume_completo


def extrair_alvo_e_processar(volume_completo: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra candidatos do segundo turno, descarta colunas nao preditivas
    e cria a variavel-alvo binaria 'Eleito'.
    """
    print(f"[DATA] Situacao totalizacao (bruto): {volume_completo['Situação totalização'].value_counts().to_dict()}", flush=True)

    # Exclui candidatos do segundo turno.
    volume = volume_completo[volume_completo["Situação totalização"] != "Segundo turno"].copy()
    
    # Remove colunas textuais sem valor preditivo.
    volume = volume.drop(["Nome candidato", "Data de carga", "Município"], axis=1)
    
    # Variavel-alvo: 1 para Eleito, 0 para nao eleito.
    volume["Eleito"] = (volume["Situação totalização"] == "Eleito").astype(int)

    print(f"Amostra pos-filtragem: {volume.shape[0]} instancias.", flush=True)
    print(f"Distribuicao dos rotulos: {volume['Eleito'].value_counts().to_dict()}", flush=True)
    
    return volume


def converter_vetores_categoricos(
    dados_treinamento: pd.DataFrame,
    dados_ineditos_teste: pd.DataFrame,
    vetores_textuais: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Codificacao ordinal de variaveis categoricas, fit no treino, transform no teste.

    Categorias presentes no treino sao mapeadas para inteiros sequenciais.
    Categorias ineditas no conjunto de teste recebem -1 (sem vazamento de dados).
    A codificacao e aplicada independentemente a cada fold da validacao cruzada.
    """
    treino = dados_treinamento.copy()
    teste = dados_ineditos_teste.copy()
    
    for vetor in vetores_textuais:
        categorias = treino[vetor].astype("category").cat.categories
        mapeamento = {classe: idx for idx, classe in enumerate(categorias)}
        
        treino[vetor] = treino[vetor].map(mapeamento).astype(int)
        
        # Categorias ineditas no teste recebem -1.
        teste[vetor] = teste[vetor].map(mapeamento).fillna(-1).astype(int)
        
    return treino, teste
