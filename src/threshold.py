"""
Busca em grade do threshold de decisão.

O threshold tau* é otimizado por MCC no conjunto de treinamento.
Referencia: Chicco & Jurman (2020). The advantages of the Matthews correlation
coefficient (MCC). BMC Genomics, 21, 6.
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef

from config import (
    LIMITE_MINIMO_DECISAO,
    LIMITE_MAXIMO_DECISAO,
    INCREMENTO_DE_BUSCA,
)


def buscar_ponto_de_corte_otimo(
    y_train: np.ndarray,
    probs_train: np.ndarray,
) -> float:
    """
    Busca em grade do threshold que maximiza o MCC no conjunto de treinamento.

    O intervalo de busca e o passo sao definidos em config.py. O threshold
    otimizado e aplicado ao conjunto de avaliacao sem reajuste (sem leakage).

    Parameters
    ----------
    y_train : np.ndarray
        Rotulos binarios do conjunto de treinamento (0/1).
    probs_train : np.ndarray
        Probabilidades calibradas P(Y=1|X) do conjunto de treinamento.

    Returns
    -------
    float
        Threshold tau* em [LIMITE_MINIMO_DECISAO, LIMITE_MAXIMO_DECISAO].
    """
    best_mcc = -1.0
    best_threshold = 0.50

    threshold_grid = np.arange(
        LIMITE_MINIMO_DECISAO,
        LIMITE_MAXIMO_DECISAO + INCREMENTO_DE_BUSCA,
        INCREMENTO_DE_BUSCA,
    )

    for tau in threshold_grid:
        y_pred = (probs_train >= tau).astype(int)
        mcc = matthews_corrcoef(y_train, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = tau

    return best_threshold
