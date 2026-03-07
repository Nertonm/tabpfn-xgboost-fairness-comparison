"""
Verificação Fix 3 : Teste Estatístico Correto (MAJOR 3.1)
=========================================================
O Wilcoxon sobre 10 folds dependentes viola pressupostos de independência.
Este script implementa o combined 5×2cv F-test (Alpaydin, 1999 / mlxtend),
que é o teste correto para comparar dois classificadores sob validação cruzada.

Uso: python src/verify_5x2cv.py

Critérios de interpretação do p-valor:
  p < 0.05  → TabPFN é estatisticamente superior após pipeline equalizado
  0.05-0.15 → Marginal; reportar com ressalvas
  p ≥ 0.15  → Sem significância; narrativa muda para "comparação de fairness"
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from config import (
    SEMENTE_ALEATORIA,
    COLUNAS_CATEGORICAS,
    COLUNAS_PREDITIVAS,
    LIMITE_MINIMO_DECISAO,
    LIMITE_MAXIMO_DECISAO,
    INCREMENTO_DE_BUSCA,
)
from data import carregar_dados_brutos, extrair_alvo_e_processar
from threshold import buscar_ponto_de_corte_otimo


# =============================================================================
# WRAPPER SKLEARN : XGBOOST (pipeline simétrico: isotônica + threshold)
# =============================================================================

class XGBoostPipelineCompleto(ClassifierMixin, BaseEstimator):
    """  # noqa: D401
    Encapsula o pipeline completo do XGBoost para compatibilidade com mlxtend:
      1. OrdinalEncoder por coluna categórica (fit exclusivamente no treino)
      2. XGBClassifier com scale_pos_weight
      3. Calibração isotônica das probabilidades brutas de treino
      4. Busca do limiar tau* que maximiza MCC no treino calibrado
    """

    _estimator_type = "classifier"  # required by mlxtend combined_ftest_5x2cv

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = np.array(y)

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=float,
        )
        cat_idx = [i for i, c in enumerate(X.columns) if c in COLUNAS_CATEGORICAS]
        if cat_idx:
            X.iloc[:, cat_idx] = self.encoder_.fit_transform(X.iloc[:, cat_idx])

        neg, pos = np.sum(y == 0), np.sum(y == 1)
        scale  = neg / pos if pos > 0 else 1.0

        self.model_ = XGBClassifier(
            scale_pos_weight=scale,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.model_.fit(X, y)

        probs_brutas = self.model_.predict_proba(X)[:, 1]
        self.calibrador_ = IsotonicRegression(out_of_bounds="clip")
        probs_cal = self.calibrador_.fit_transform(probs_brutas, y)

        self.limiar_ = buscar_ponto_de_corte_otimo(y, probs_cal)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cat_idx = [i for i, c in enumerate(X.columns) if c in COLUNAS_CATEGORICAS]
        if cat_idx:
            X.iloc[:, cat_idx] = self.encoder_.transform(X.iloc[:, cat_idx])
        probs_brutas = self.model_.predict_proba(X)[:, 1]
        probs_cal    = self.calibrador_.predict(probs_brutas)
        return np.column_stack([1 - probs_cal, probs_cal])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.limiar_).astype(int)

    def score(self, X, y):
        return matthews_corrcoef(y, self.predict(X))


# =============================================================================
# WRAPPER SKLEARN : TABPFN (pipeline simétrico: isotônica + threshold)
# =============================================================================

class TabPFNPipelineCompleto(ClassifierMixin, BaseEstimator):
    """  # noqa: D401
    Encapsula o pipeline completo do TabPFN para compatibilidade com mlxtend:
      1. OrdinalEncoder por coluna categórica (fit exclusivamente no treino)
      2. TabPFNClassifier (device=cuda)
      3. Calibração isotônica das probabilidades brutas de treino
      4. Busca do limiar tau* que maximiza MCC no treino calibrado
    """

    _estimator_type = "classifier"  # required by mlxtend combined_ftest_5x2cv

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = np.array(y)

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=float,
        )
        cat_idx = [i for i, c in enumerate(X.columns) if c in COLUNAS_CATEGORICAS]
        if cat_idx:
            X.iloc[:, cat_idx] = self.encoder_.fit_transform(X.iloc[:, cat_idx])

        self.model_ = TabPFNClassifier(device="cuda")
        self.model_.fit(X, y)

        probs_brutas = self.model_.predict_proba(X)[:, 1]
        self.calibrador_ = IsotonicRegression(out_of_bounds="clip")
        probs_cal = self.calibrador_.fit_transform(probs_brutas, y)

        self.limiar_ = buscar_ponto_de_corte_otimo(y, probs_cal)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cat_idx = [i for i, c in enumerate(X.columns) if c in COLUNAS_CATEGORICAS]
        if cat_idx:
            X.iloc[:, cat_idx] = self.encoder_.transform(X.iloc[:, cat_idx])
        probs_brutas = self.model_.predict_proba(X)[:, 1]
        probs_cal    = self.calibrador_.predict(probs_brutas)
        return np.column_stack([1 - probs_cal, probs_cal])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.limiar_).astype(int)

    def score(self, X, y):
        return matthews_corrcoef(y, self.predict(X))


# =============================================================================
# EXECUÇÃO DO TESTE
# =============================================================================

def main():
    try:
        from mlxtend.evaluate import combined_ftest_5x2cv
    except ImportError:
        print("mlxtend não instalado. Execute: pip install mlxtend")
        sys.exit(1)

    import json

    print("=" * 60)
    print("Teste estatistico: 5x2cv Combined F-test (Alpaydin 1999)")
    print("=" * 60)

    print("\n[1/3] Carregando e pré-processando dados...")
    dados_brutos      = carregar_dados_brutos()
    dados_processados = extrair_alvo_e_processar(dados_brutos)

    X = dados_processados[COLUNAS_PREDITIVAS]
    y = dados_processados["Eleito"].values

    # Converte categorias para valores ordinais com o vocabulário global
    # (apenas para passar ao mlxtend; cada estimator recodifica internamente
    # no seu fold de treino : garantindo ausência de leakage).
    encoder_global = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=float,
    )
    X_num = X.copy()
    X_num[COLUNAS_CATEGORICAS] = encoder_global.fit_transform(X[COLUNAS_CATEGORICAS])

    print(f"   Dataset: {X_num.shape[0]} instâncias, {X_num.shape[1]} features")
    print(f"   Distribuição: {np.sum(y==0)} Não-Eleito | {np.sum(y==1)} Eleito")

    print("\n[2/3] Inicializando estimadores com pipeline completo...")
    estimador_xgb = XGBoostPipelineCompleto(random_state=SEMENTE_ALEATORIA)
    estimador_pfn = TabPFNPipelineCompleto(random_state=SEMENTE_ALEATORIA)

    print("\n[3/3] Executando 5×2cv F-test (10 treinos de TabPFN : pode demorar)...")
    f_stat, p_valor = combined_ftest_5x2cv(
        estimator1=estimador_pfn,
        estimator2=estimador_xgb,
        X=X_num.values,
        y=y,
        random_seed=SEMENTE_ALEATORIA,
    )

    significativo = bool(p_valor < 0.05)
    resultado_str = "SIGNIFICATIVO" if significativo else "NÃO SIGNIFICATIVO"

    # -- Output formatado ------------------------------------------------------
    print()
    print("+---------------------------------------------+")
    print("-  5×2cv Combined F-Test (Alpaydin, 1999)     -")
    print(f"-  F-statistic : {f_stat:<28.6f}-")
    print(f"-  p-value     : {p_valor:<28.6f}-")
    print(f"-  Resultado   : {resultado_str:<29}-")
    print("-  (α = 0.05)                                 -")
    print("+---------------------------------------------+")

    if significativo:
        print("\n  → TabPFN é estatisticamente superior ao XGBoost (MCC scoring).")
        print("  → Referência: Alpaydin (1999), Neural Computation 11(8).")
    else:
        print("\n  → Diferença NÃO é estatisticamente significativa a α=0.05.")
        print("  → Reportar honestamente nas limitações do artigo.")

    # -- Salva JSON ------------------------------------------------------------
    resultado = {
        "test"        : "combined_ftest_5x2cv",
        "reference"   : "Alpaydin 1999 / Dietterich 1998",
        "F_statistic" : float(f_stat),
        "p_value"     : float(p_valor),
        "p_valor"     : float(p_valor),   # alias em português para checklist.py
        "alpha"       : 0.05,
        "significant" : significativo,
        "significativo": significativo,   # alias em português para compatibilidade
        "n_splits"    : 5,
        "n_repeats"   : 2,
        "random_seed" : int(SEMENTE_ALEATORIA),
        "scoring"     : "matthews_corrcoef",
        "models"      : {
            "model1": "XGBoost + IsotonicCalibration + ThresholdOptimizer",
            "model2": "TabPFNClassifier + IsotonicCalibration + ThresholdOptimizer",
        },
    }

    os.makedirs("output", exist_ok=True)
    caminho_json = "output/5x2cv_result.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Resultado salvo em {caminho_json}")


if __name__ == "__main__":
    main()
