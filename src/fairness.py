"""
Metricas de fairness e testes estatisticos para comparacao de classificadores.

Implementa:
- DIR (Disparate Impact Ratio)
- EOD (Equal Opportunity Difference)
- FNR_DIFF (diferenca de taxa de falsos negativos)
- Bootstrap CI (1000 iteracoes por padrao)
- Cohen d, McNemar, Wilcoxon (auxiliares)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix
from scipy.stats import wilcoxon, norm
from statsmodels.stats.contingency_tables import mcnemar

# Funcao interna: Disparate Impact Ratio (DIR).
def _calcular_dir(
    alvo_predito: np.ndarray, 
    grupo_demografico: np.ndarray, 
    id_privilegiado: int, 
    id_nao_privilegiado: int
) -> float:
    # Taxa de aprovacao do modelo para cada grupo demografico.
    predicao_nao_priv = alvo_predito[grupo_demografico == id_nao_privilegiado]
    predicao_priv = alvo_predito[grupo_demografico == id_privilegiado]
    
    if len(predicao_nao_priv) == 0 or len(predicao_priv) == 0:
        return np.nan
        
    taxa_minoritaria = np.mean(predicao_nao_priv)
    taxa_hegemonica = np.mean(predicao_priv)
    
    if taxa_hegemonica == 0:
        return np.nan
        
    return taxa_minoritaria / taxa_hegemonica


# Funcao interna: Equal Opportunity Difference (EOD).
def _calcular_eod(
    alvo_original: np.ndarray, 
    alvo_predito: np.ndarray, 
    grupo_demografico: np.ndarray, 
    id_privilegiado: int, 
    id_nao_privilegiado: int
) -> float:
    # TPR por grupo: fracao de verdadeiro-positivos no grupo nao-privilegiado vs privilegiado.
    mascara_vencedor_minoritario = (grupo_demografico == id_nao_privilegiado) & (alvo_original == 1)
    mascara_vencedor_hegemonico = (grupo_demografico == id_privilegiado) & (alvo_original == 1)
    
    predicao_nao_priv = alvo_predito[mascara_vencedor_minoritario]
    predicao_priv = alvo_predito[mascara_vencedor_hegemonico]
    
    if len(predicao_nao_priv) == 0 or len(predicao_priv) == 0:
        return np.nan
        
    taxa_tpr_minoritarios = np.mean(predicao_nao_priv)
    taxa_tpr_hegemonicos = np.mean(predicao_priv)
    
    return taxa_tpr_minoritarios - taxa_tpr_hegemonicos


# Funcao interna: diferenca de taxa de falsos negativos (FNR_DIFF).
def _calcular_fnr_diff(
    alvo_original: np.ndarray, 
    alvo_predito: np.ndarray, 
    grupo_demografico: np.ndarray, 
    id_privilegiado: int, 
    id_nao_privilegiado: int
) -> float:
    # FNR = 1 - TPR. Diferenca positiva indica maior erro no grupo nao-privilegiado.
    mascara_vencedor_minoritario = (grupo_demografico == id_nao_privilegiado) & (alvo_original == 1)
    mascara_vencedor_hegemonico = (grupo_demografico == id_privilegiado) & (alvo_original == 1)
    
    predicao_nao_priv = alvo_predito[mascara_vencedor_minoritario]
    predicao_priv = alvo_predito[mascara_vencedor_hegemonico]
    
    if len(predicao_nao_priv) == 0 or len(predicao_priv) == 0:
        return np.nan
        
    fnr_minoritarios = 1 - np.mean(predicao_nao_priv)
    fnr_hegemonicos  = 1 - np.mean(predicao_priv)
    
    return fnr_minoritarios - fnr_hegemonicos


def avaliar_justica_bootstrap(
    rotulos: np.ndarray,
    predicoes: np.ndarray,
    grupo_demografico: np.ndarray,
    id_privilegiado: int,
    id_nao_privilegiado: int,
    iteracoes_bootstrap: int = 1000
) -> dict:
    """
    Estima DIR, EOD e FNR_DIFF com intervalos de confianca de 95% via bootstrap.

    Parameters
    ----------
    rotulos : np.ndarray
        Rotulos verdadeiros (0/1).
    predicoes : np.ndarray
        Predicoes do classificador (0/1).
    grupo_demografico : np.ndarray
        Vetor booleano ou inteiro indicando o grupo de cada instancia.
    id_privilegiado : int
        Valor que identifica o grupo privilegiado no vetor `grupo_demografico`.
    id_nao_privilegiado : int
        Valor que identifica o grupo nao-privilegiado.
    iteracoes_bootstrap : int
        Numero de reamostras (padrao: 1000).

    Returns
    -------
    dict
        Chaves: DIR, DIR_IC, EOD, EOD_IC, FNR_DIFF, FNR_DIFF_IC.
    """
    dir_orig = _calcular_dir(predicoes, grupo_demografico, id_privilegiado, id_nao_privilegiado)
    eod_orig = _calcular_eod(rotulos, predicoes, grupo_demografico, id_privilegiado, id_nao_privilegiado)
    fnr_orig = _calcular_fnr_diff(rotulos, predicoes, grupo_demografico, id_privilegiado, id_nao_privilegiado)
    
    memoria_dir = []
    memoria_eod = []
    memoria_fnr = []
    
    tamanho_simulacao = len(rotulos)
    np.random.seed(42)
    
    for _ in range(iteracoes_bootstrap):
        # Reamostragem com reposicao (bootstrap).
        indices_pseudo_amostra = np.random.choice(tamanho_simulacao, tamanho_simulacao, replace=True)
        
        rotulos_sim = rotulos[indices_pseudo_amostra]
        predicao_simulada = predicoes[indices_pseudo_amostra]
        grupo_simulado = grupo_demografico[indices_pseudo_amostra]
        
        dir_sim = _calcular_dir(predicao_simulada, grupo_simulado, id_privilegiado, id_nao_privilegiado)
        eod_sim = _calcular_eod(rotulos_sim, predicao_simulada, grupo_simulado, id_privilegiado, id_nao_privilegiado)
        fnr_sim = _calcular_fnr_diff(rotulos_sim, predicao_simulada, grupo_simulado, id_privilegiado, id_nao_privilegiado)
        
        if not np.isnan(dir_sim): memoria_dir.append(dir_sim)
        if not np.isnan(eod_sim): memoria_eod.append(eod_sim)
        if not np.isnan(fnr_sim): memoria_fnr.append(fnr_sim)
        
    # IC 95% por percentil.
    return {
        "DIR": dir_orig,
        "DIR_IC": (np.percentile(memoria_dir, 2.5), np.percentile(memoria_dir, 97.5)) if memoria_dir else (0,0),
        "EOD": eod_orig,
        "EOD_IC": (np.percentile(memoria_eod, 2.5), np.percentile(memoria_eod, 97.5)) if memoria_eod else (0,0),
        "FNR_DIFF": fnr_orig,
        "FNR_DIFF_IC": (np.percentile(memoria_fnr, 2.5), np.percentile(memoria_fnr, 97.5)) if memoria_fnr else (0,0)
    }


def teste_de_efeito_cohens(array_x: np.ndarray, array_y: np.ndarray) -> float:
    """
    Calcula o d de Cohen entre dois arrays (tamanho de efeito padronizado).
    Valores de referencia: 0.2=pequeno, 0.5=medio, 0.8=grande (Cohen 1988).
    """
    media_x, media_y = np.mean(array_x), np.mean(array_y)
    desvio_x, desvio_y = np.var(array_x, ddof=1), np.var(array_y, ddof=1)
    
    tamanho_x, tamanho_y = len(array_x), len(array_y)
    pooled_var = ((tamanho_x - 1) * desvio_x + (tamanho_y - 1) * desvio_y) / (tamanho_x + tamanho_y - 2)
    
    return (media_x - media_y) / np.sqrt(pooled_var)


def teste_discordancia_mcnemar(y_true: np.ndarray, pred_algoritmo_A: np.ndarray, pred_algoritmo_B: np.ndarray) -> tuple:
    """
    Teste de McNemar para comparar dois classificadores em pares.

    Considera apenas os casos de discordancia (b: A certo, B errado;
    c: A errado, B certo). Retorna estatistica, p-valor, b, c.
    """
    # Tabela de contingencia 2x2.
    ambos_erraram = sum((pred_algoritmo_A != y_true) & (pred_algoritmo_B != y_true))
    a_acertou_b_errou = sum((pred_algoritmo_A == y_true) & (pred_algoritmo_B != y_true))  # b
    a_errou_b_acertou = sum((pred_algoritmo_A != y_true) & (pred_algoritmo_B == y_true))  # c
    ambos_acertaram = sum((pred_algoritmo_A == y_true) & (pred_algoritmo_B == y_true))
    
    contingencia = [[ambos_acertaram, a_acertou_b_errou],
                    [a_errou_b_acertou, ambos_erraram]]
                    
    resultado = mcnemar(contingencia, exact=False, correction=True)
    return resultado.statistic, resultado.pvalue, a_acertou_b_errou, a_errou_b_acertou


def redigir_relatorio_fairness(pacote_metricas: dict, modelo_analisado: str, privilegios: int, minorias: int):
    """Imprime as metricas de fairness de um modelo no stdout."""
    print(f"\nModelo: {modelo_analisado}")
    print(f"  DIR      : {pacote_metricas['DIR']:.3f}  IC-95%: [{pacote_metricas['DIR_IC'][0]:.3f}, {pacote_metricas['DIR_IC'][1]:.3f}]")
    print(f"  EOD      : {pacote_metricas['EOD']:.3f}  IC-95%: [{pacote_metricas['EOD_IC'][0]:.3f}, {pacote_metricas['EOD_IC'][1]:.3f}]")
    print(f"  FNR_DIFF : {pacote_metricas['FNR_DIFF']:.3f}  IC-95%: [{pacote_metricas['FNR_DIFF_IC'][0]:.3f}, {pacote_metricas['FNR_DIFF_IC'][1]:.3f}]")
    print(f"  n privilegiado={privilegios}, n nao-privilegiado={minorias}")
