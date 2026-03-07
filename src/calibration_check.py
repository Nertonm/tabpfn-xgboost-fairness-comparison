"""
Reliability Diagram : Diagnóstico de Calibração
================================================
Gera reliability diagrams (curvas de calibração empírica) para os modelos
XGBoost e TabPFN, usando as probabilidades out-of-fold coletadas durante
a validação cruzada estratificada.

Uso: python src/calibration_check.py

Saída:
  - output/calibration_check.png  (dpi=200)
  - ECE impresso no terminal para ambos os modelos

ECE (Expected Calibration Error):
  ECE = Σ_m (|B_m| / n) × |acc(B_m) − conf(B_m)|
  onde B_m são bins de 10 intervalos uniformes em [0, 1].
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from config import SEMENTE_ALEATORIA
from data import carregar_dados_brutos, extrair_alvo_e_processar
from training import executar_validacao_cruzada_estratificada


# =============================================================================
# CÁLCULO DO ECE
# =============================================================================

def calcular_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcula o Expected Calibration Error (ECE) com bins uniformes.

    ECE = Σ_m (|B_m| / n) × |acc(B_m) − conf(B_m)|

    Parâmetros
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros (0/1).
    y_prob : np.ndarray
        Probabilidades preditas para a classe positiva.
    n_bins : int
        Número de bins uniformes.

    Retorna
    -------
    float
        Valor do ECE.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        mascara = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mascara.sum() == 0:
            continue
        acc_bin  = y_true[mascara].mean()
        conf_bin = y_prob[mascara].mean()
        peso_bin = mascara.sum() / n
        ece += peso_bin * abs(acc_bin - conf_bin)

    return float(ece)


# =============================================================================
# GERAÇÃO DA FIGURA
# =============================================================================

def _subplot_reliability(
    ax_diag: plt.Axes,
    ax_hist: plt.Axes,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    titulo: str,
    cor: str,
) -> float:
    """
    Plota reliability diagram + histograma de scores num par de eixos.

    Retorna o ECE calculado.
    """
    ece = calcular_ece(y_true, y_prob, n_bins=10)

    # Curva de calibração empírica
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    ax_diag.plot(
        mean_pred, frac_pos,
        marker="o", linewidth=2, color=cor,
        label=f"Calibração empírica",
    )
    ax_diag.plot(
        [0, 1], [0, 1],
        linestyle="--", color="gray", linewidth=1.2,
        label="Calibração perfeita",
    )
    ax_diag.set_xlim(0, 1)
    ax_diag.set_ylim(0, 1)
    ax_diag.set_xlabel("Confiança média por bin")
    ax_diag.set_ylabel("Fração de positivos")
    ax_diag.set_title(f"{titulo}\nECE = {ece:.4f}")
    ax_diag.legend(fontsize=9)

    # Histograma de scores no eixo sobreposto
    ax_hist.hist(y_prob, bins=20, alpha=0.3, color=cor, edgecolor="white")
    ax_hist.set_ylabel("Contagem", color="gray", fontsize=9)
    ax_hist.tick_params(axis="y", labelcolor="gray", labelsize=8)

    return ece


def gerar_reliability_diagram(caderno: dict) -> None:
    """
    Gera e salva o painel de reliability diagrams para XGBoost e TabPFN.
    """
    y_true  = np.array(caderno["xgboost"]["rotulos_reais"])
    probs_xgb = np.array(caderno["xgboost"]["probabilidades_estimadas"])
    probs_pfn = np.array(caderno["tabpfn"]["probabilidades_calibradas"])

    fig, ((ax1, ax2), (ax1h, ax2h)) = plt.subplots(
        2, 2, figsize=(12, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ece_xgb = _subplot_reliability(ax1, ax1h, y_true, probs_xgb, "XGBoost", "#E07B39")
    ece_pfn = _subplot_reliability(ax2, ax2h, y_true, probs_pfn, "TabPFN",  "#3E8BBD")

    fig.suptitle("Reliability Diagrams : Calibração dos Modelos", fontsize=15, y=1.01)
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    caminho = "output/calibration_check.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nCalibração XGBoost : ECE: {ece_xgb:.4f}")
    print(f"Calibração TabPFN  : ECE: {ece_pfn:.4f}")

    if ece_xgb > 0.05:
        print(
            "\n  [AVISO] ECE_XGB > 0.05: calibração insuficiente : "
            "considerar retreinamento com Platt scaling (CalibratedClassifierCV sigmoid)."
        )

    print(f"\n[OK] Reliability diagram salvo em {caminho}")


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

def main() -> None:
    np.random.seed(SEMENTE_ALEATORIA)

    print("Carregando dados e re-executando validação cruzada...")
    dados_brutos = carregar_dados_brutos()
    dados = extrair_alvo_e_processar(dados_brutos)
    caderno = executar_validacao_cruzada_estratificada(dados)

    gerar_reliability_diagram(caderno)


if __name__ == "__main__":
    main()
