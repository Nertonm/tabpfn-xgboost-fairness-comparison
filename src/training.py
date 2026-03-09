"""
Validação cruzada estratificada: XGBoost vs TabPFN.

Implementa K-Fold estratificado (K=10) com calibração isotonica e
otimizacao de threshold por MCC, conforme descrito em Materiais e Metodos.
Nenhuma informaçao do conjunto de avaliacao contamina o treinamento.
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import matthews_corrcoef
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from config import (
    SEMENTE_ALEATORIA,
    NUMERO_DE_FOLDS,
    COLUNAS_CATEGORICAS,
    COLUNAS_PREDITIVAS,
)
from data import converter_vetores_categoricos
from threshold import buscar_ponto_de_corte_otimo


# Tipo estrutural do dicionario de resultados acumulados ao longo dos K folds.
CadernoDeResultados = Dict[str, Any]


def _inicializar_caderno_de_resultados() -> CadernoDeResultados:
    """
    Inicializa o dicionario de resultados com listas vazias por modelo.

    Returns
    -------
    CadernoDeResultados
        Dicionario hierarquico com listas vazias para acumular predicoes fold a fold.
    """
    return {
        "xgboost": {
            "rotulos_reais"            : [],
            "predicoes_duras"          : [],
            "probabilidades_estimadas" : [],
            "limiares_otimizados"      : [],
            "mcc_folds"                : [],
        },
        "tabpfn": {
            "rotulos_reais"            : [],
            "predicoes_duras"          : [],
            "probabilidades_calibradas": [],
            "limiares_otimizados"      : [],
            "mcc_folds"                : [],
        },
        # Atributos sociodemograficos do subconjunto de avaliacao preservados
        # para calculo das metricas de fairness (DIR, EOD) apos o loop de folds.
        # 'eleicao' preservado para analise de sensibilidade racial (2012 sem raca).
        "genero_no_conjunto_avaliacao": [],
        "raca_no_conjunto_avaliacao"  : [],
        "eleicao_no_conjunto_avaliacao": [],
    }


def _calcular_peso_da_classe_majoritaria(
    y_train: pd.Series,
) -> float:
    """
    Calcula scale_pos_weight = N_negativos / N_positivos para o fold corrente.

    O calculo e restrito ao subconjunto de treinamento para evitar leakage
    de informacao estatistica do conjunto de avaliacao.

    Parameters
    ----------
    y_train : pd.Series
        Rotulos binarios do subconjunto de treinamento (0 = nao eleito, 1 = eleito).

    Returns
    -------
    float
        Fator de escala para o parametro scale_pos_weight do XGBClassifier.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos


def _treinar_e_avaliar_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Treina XGBoost com calibracao isotonica e otimizacao de threshold no treino.

    Probabilidades OOF (cross_val_predict cv=5) sao usadas para calibracao
    a fim de evitar colapso de scores in-sample gerado pelo scale_pos_weight.
    O threshold tau* e otimizado por MCC exclusivamente no conjunto de treino.
    Pipeline simetrico ao TabPFN (Fix metodologico 2.1/2.2).

    Parameters
    ----------
    X_train : pd.DataFrame
        Matriz de covariaveis do subconjunto de treinamento.
    y_train : pd.Series
        Rotulos do subconjunto de treinamento.
    X_test : pd.DataFrame
        Matriz de covariaveis do subconjunto de avaliacao.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (y_pred, probs_test, best_threshold):
        - y_pred: rotulos binarios com tau* aplicado
        - probs_test: P_calibrado(Y=1|X) para o conjunto de avaliacao
        - best_threshold: tau* encontrado no conjunto de treinamento
    """
    scale_pos_weight = _calcular_peso_da_classe_majoritaria(y_train)

    xgb_params = dict(
        scale_pos_weight=scale_pos_weight,
        random_state=SEMENTE_ALEATORIA,
        eval_metric="logloss",
    )

    # Modelo final treinado em todo o conjunto de treino.
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train)

    # Probabilidades OOF (cv=5) no treino para calibracao isotonica.
    # XGBoost com scale_pos_weight gera scores extremos (aprox. 0/1) in-sample;
    # calibracao in-sample colapsaria tau* para o boundary inferior.
    # Usando OOF evita esse efeito e gera tau* variavel por fold (Fix 1).
    probs_oof = cross_val_predict(
        XGBClassifier(**xgb_params),
        X_train,
        y_train,
        cv=5,
        method="predict_proba",
    )[:, 1]

    # Calibracao isotonica sobre OOF (nao in-sample).
    calibrator = IsotonicRegression(out_of_bounds="clip")
    probs_cal_oof = calibrator.fit_transform(probs_oof, y_train)

    # Threshold otimizado por MCC no treino calibrado (mesmo protocolo do TabPFN).
    best_threshold = buscar_ponto_de_corte_otimo(y_train, probs_cal_oof)

    # Inferencia no conjunto de avaliacao com threshold congelado.
    probs_raw_test = xgb.predict_proba(X_test)[:, 1]
    probs_test = calibrator.predict(probs_raw_test)
    y_pred = (probs_test >= best_threshold).astype(int)

    return y_pred, probs_test, best_threshold


def _treinar_e_avaliar_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Treina TabPFN com calibracao isotonica e otimizacao de threshold no treino.

    A calibracao e feita in-sample (TabPFNClassifier.predict_proba no treino)
    pois o TabPFN ja e conservador e nao sofre o colapso de scores visto no XGBoost.
    O threshold tau* e otimizado por MCC exclusivamente no conjunto de treino.
    Sem acesso ao conjunto de avaliacao em nenhuma etapa.

    Parameters
    ----------
    X_train : pd.DataFrame
        Matriz de covariaveis do subconjunto de treinamento.
    y_train : pd.Series
        Rotulos do subconjunto de treinamento.
    X_test : pd.DataFrame
        Matriz de covariaveis do subconjunto de avaliacao.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (y_pred, probs_test, best_threshold):
        - y_pred: rotulos binarios com tau* aplicado
        - probs_test: P_calibrado(Y=1|X) para o conjunto de avaliacao
        - best_threshold: tau* encontrado no conjunto de treinamento
    """
    # Treinamento do TabPFN (pesos congelados; apenas scores de saida sao calibrados).
    tabpfn = TabPFNClassifier(device="cuda")
    tabpfn.fit(X_train, y_train)

    # Calibracao isotonica sobre scores in-sample do treino.
    # out_of_bounds="clip" projeta scores fora do intervalo de treino para os extremos.
    probs_raw_train = tabpfn.predict_proba(X_train)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    probs_cal_train = calibrator.fit_transform(probs_raw_train, y_train)

    # Threshold otimizado por MCC no treino calibrado. Conjunto de avaliacao vedado.
    best_threshold = buscar_ponto_de_corte_otimo(y_train, probs_cal_train)

    # Inferencia no conjunto de avaliacao com threshold e calibrador congelados.
    probs_raw_test = tabpfn.predict_proba(X_test)[:, 1]
    probs_test = calibrator.predict(probs_raw_test)
    y_pred = (probs_test >= best_threshold).astype(int)

    return y_pred, probs_test, best_threshold


def executar_validacao_cruzada_estratificada(
    df: pd.DataFrame,
) -> CadernoDeResultados:
    """
    K-Fold estratificado (K=10) comparando XGBoost e TabPFN.

    Estratificacao por estrato composto (Genero x Raca_binaria x Eleito) para
    garantir proporcao equilibrada dos atributos protegidos em cada fold.
    Codificacao categorica recalculada por fold (vocabulario restrito ao treino).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame processado com covariaveis e variavel-alvo "Eleito".

    Returns
    -------
    CadernoDeResultados
        Dicionario com predicoes out-of-sample, probabilidades, thresholds e
        atributos sociodemograficos para fairness.
    """
    print(f"\nProtocolo: Validacao Cruzada Estratificada com K={NUMERO_DE_FOLDS} particoes.")
    print("=" * 60)

    y = df["Eleito"]
    X = df[COLUNAS_PREDITIVAS]

    # Estrato composto: Genero x Raca_binaria x Eleito.
    # Raca e o segundo atributo protegido; omiti-la do estrato poderia gerar
    # folds com distribuicao racial desequilibrada (AUDITORIA Fase 3).
    raca_binaria = df["Cor/raça"].apply(
        lambda x: "Branca" if x == "Branca" else "NaoBranca"
    )
    strat_labels = (
        df["Gênero"].astype(str)
        + "_"
        + raca_binaria.astype(str)
        + "_"
        + y.astype(str)
    )

    # Verificar viabilidade do menor estrato.
    strat_counts = strat_labels.value_counts()
    min_strat = strat_counts.min()
    print(f"[STRAT] Estratos compostos (Genero x Raca x Eleito): {len(strat_counts)} grupos")
    print(f"[STRAT] Menor estrato: {strat_counts.idxmin()} = {min_strat} instancias")
    if min_strat < NUMERO_DE_FOLDS:
        print(f"[STRAT] Estrato insuficiente para K={NUMERO_DE_FOLDS}. Revertendo para Genero+Eleito.")
        strat_labels = df["Gênero"].astype(str) + "_" + y.astype(str)

    skf = StratifiedKFold(
        n_splits=NUMERO_DE_FOLDS,
        shuffle=True,
        random_state=SEMENTE_ALEATORIA,
    )

    caderno = _inicializar_caderno_de_resultados()

    for fold, (idx_train, idx_test) in enumerate(skf.split(X, strat_labels), start=1):
        # Particao treino/avaliacao: o conjunto de avaliacao nao e acessado
        # em nenhuma etapa de estimacao (threshold, calibracao, hiperparametros).
        X_train = X.iloc[idx_train]
        y_train = y.iloc[idx_train]
        X_test  = X.iloc[idx_test]
        y_test  = y.iloc[idx_test]

        # Codificacao categorica restrita ao vocabulario do treino (anti-leakage).
        # Categorias ausentes no treino recebem codigo -1 no conjunto de avaliacao.
        X_train, X_test = converter_vetores_categoricos(
            X_train, X_test, COLUNAS_CATEGORICAS,
        )

        # XGBoost: calibracao isotonica OOF + threshold por MCC.
        # Ref: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
        y_pred_xgb, probs_xgb, tau_xgb = _treinar_e_avaliar_xgboost(
            X_train, y_train, X_test,
        )

        caderno["xgboost"]["rotulos_reais"].extend(y_test)
        caderno["xgboost"]["predicoes_duras"].extend(y_pred_xgb)
        caderno["xgboost"]["probabilidades_estimadas"].extend(probs_xgb)
        caderno["xgboost"]["limiares_otimizados"].append(tau_xgb)

        # TabPFN: calibracao isotonica in-sample + threshold por MCC.
        # Ref: Hollmann et al. (2022). TabPFN: A Transformer That Solves
        #      Small Tabular Classification Problems in a Second.
        y_pred_tab, probs_tab, tau_tab = _treinar_e_avaliar_tabpfn(
            X_train, y_train, X_test,
        )

        caderno["tabpfn"]["rotulos_reais"].extend(y_test)
        caderno["tabpfn"]["predicoes_duras"].extend(y_pred_tab)
        caderno["tabpfn"]["probabilidades_calibradas"].extend(probs_tab)
        caderno["tabpfn"]["limiares_otimizados"].append(tau_tab)

        # Atributos sociodemograficos preservados em forma textual original
        # para calculo de fairness apos o loop (DIR, EOD, FNR_DIFF).
        # 'eleicao' preservado para analise de sensibilidade racial (2012 sem raça).
        df_test_orig = df.iloc[idx_test]
        caderno["genero_no_conjunto_avaliacao"].extend(df_test_orig["Gênero"])
        caderno["raca_no_conjunto_avaliacao"].extend(df_test_orig["Cor/raça"])
        caderno["eleicao_no_conjunto_avaliacao"].extend(df_test_orig["eleicao"])

        # MCC por fold armazenado diretamente (evita reconstrucao por fatiamento).
        mcc_xgb = matthews_corrcoef(y_test, y_pred_xgb)
        mcc_tab = matthews_corrcoef(y_test, y_pred_tab)
        caderno["xgboost"]["mcc_folds"].append(mcc_xgb)
        caderno["tabpfn"]["mcc_folds"].append(mcc_tab)

        print(
            f"Particao {fold:>2}/{NUMERO_DE_FOLDS}"
            f" | tau*(XGB)={tau_xgb:.2f}"
            f" | tau*(TabPFN)={tau_tab:.2f}"
            f" | MCC(XGB)={mcc_xgb:.3f}"
            f" | MCC(TabPFN)={mcc_tab:.3f}"
        )

    print("=" * 60)
    print(f"Validacao cruzada concluida. Total de instancias avaliadas: {len(caderno['xgboost']['rotulos_reais'])}.")

    return caderno
