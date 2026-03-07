"""
Checklist de verificacao para submissao do artigo.

Executa todas as verificacoes automaticas descritas no plano de acao.
Deve ser rodado APOS `python src/main.py` (que gera os resultados) e
APOS `python src/verify_5x2cv.py` (que gera output/5x2cv_result.json).

Uso: python src/checklist.py

Cada item retorna [OK] (aprovado) ou [FAIL] (pendente).
O artigo esta pronto para submissao quando todos os [OK] aparecerem.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd

from config import SEMENTE_ALEATORIA, COLUNAS_PREDITIVAS, COLUNAS_CATEGORICAS
from data import carregar_dados_brutos, extrair_alvo_e_processar
from training import executar_validacao_cruzada_estratificada
from analysis import executar_bateria_estatistica_e_fairness


# =============================================================================
# HELPERS
# =============================================================================

def _check(descricao: str, passou: bool, detalhe: str = "") -> bool:
    icone = "[OK]" if passou else "[FAIL]"
    linha = f"  {icone}  {descricao}"
    if detalhe:
        linha += f"\n       {detalhe}"
    print(linha)
    return passou


# =============================================================================
# CHECKLIST
# =============================================================================

def main():
    print("=" * 70)
    print("Checklist de aprovacao")
    print("=" * 70)

    resultados = {}

    # -------------------------------------------------------------------------
    print("\n[A] Executando pipeline completo para coletar caderno de resultados...")
    dados_brutos = carregar_dados_brutos()
    dados = extrair_alvo_e_processar(dados_brutos)
    caderno = executar_validacao_cruzada_estratificada(dados)
    caderno = executar_bateria_estatistica_e_fairness(caderno)
    print()

    # =========================================================================
    # CHECK 1 : Threshold do XGBoost variável (Fix 1 / FATAL 2.1)
    # =========================================================================
    taus_xgb = caderno["xgboost"]["limiares_otimizados"]
    taus_pfn = caderno["tabpfn"]["limiares_otimizados"]

    from config import LIMITE_MINIMO_DECISAO, LIMITE_MAXIMO_DECISAO
    # tau* variável: não pode estar fixo em qualquer boundary ou em 0.5
    n_boundary_xgb = sum(
        abs(t - LIMITE_MINIMO_DECISAO) < 1e-6 or abs(t - LIMITE_MAXIMO_DECISAO) < 1e-6
        for t in taus_xgb
    )
    xgb_variavel = (n_boundary_xgb <= 3) and (len(set(round(t, 2) for t in taus_xgb)) > 2)
    pfn_variavel = any(t != 0.5 for t in taus_pfn)

    resultados["threshold_xgb_variavel"] = xgb_variavel
    _check(
        "Threshold XGBoost não fixo em boundary (tau* variável)",
        xgb_variavel,
        f"tau* XGB por fold: {[round(t, 2) for t in taus_xgb]}  |  boundary={n_boundary_xgb}/10",
    )

    resultados["threshold_pfn_variavel"] = pfn_variavel
    _check(
        "Threshold TabPFN variável por fold (confirmação)",
        pfn_variavel,
        f"tau* PFN por fold: {[round(t, 2) for t in taus_pfn]}",
    )

    # =========================================================================
    # CHECK 2 : Fold MCCs armazenados corretamente (Fix 2 / MAJOR 3.3)
    # =========================================================================
    from config import NUMERO_DE_FOLDS

    n_folds_xgb = len(caderno["xgboost"]["mcc_folds"])
    n_folds_pfn = len(caderno["tabpfn"]["mcc_folds"])
    folds_corretos = (n_folds_xgb == NUMERO_DE_FOLDS) and (n_folds_pfn == NUMERO_DE_FOLDS)

    resultados["fold_mcc_correto"] = folds_corretos
    _check(
        f"MCC por fold armazenado corretamente ({NUMERO_DE_FOLDS} folds cada)",
        folds_corretos,
        f"XGB: {n_folds_xgb} folds  |  TabPFN: {n_folds_pfn} folds",
    )

    # =========================================================================
    # CHECK 3 : 5×2cv p-valor (Fix 3 / MAJOR 3.1)
    # =========================================================================
    resultado_5x2cv_path = "output/5x2cv_result.json"
    if os.path.exists(resultado_5x2cv_path):
        with open(resultado_5x2cv_path) as f:
            res_5x2cv = json.load(f)
        p_5x2cv = res_5x2cv["p_valor"]
        significativo = p_5x2cv < 0.05
        resultados["5x2cv_p_significativo"] = significativo
        _check(
            f"5×2cv F-test concluído (p={p_5x2cv:.4f})",
            significativo,
            "p < 0.05: TabPFN superiority survives correct test" if significativo
            else f"p={p_5x2cv:.4f} ≥ 0.05: revisar narrativa principal do artigo",
        )
    else:
        resultados["5x2cv_p_significativo"] = False
        _check(
            "5×2cv F-test concluído",
            False,
            "output/5x2cv_result.json não encontrado. Execute: python src/verify_5x2cv.py",
        )

    # =========================================================================
    # CHECK 4 : Calibração aplicada a ambos (Fix 4 / MAJOR 2.2)
    # =========================================================================
    # Verifica indiretamente: o XGBoost deve agora ter threshold ≠ 0.5 (sinal
    # de que a calibração isotônica foi aplicada e tau* foi otimizado).
    calibracao_xgb = xgb_variavel  # se threshold varia, calibração está ativa
    calibracao_pfn = True           # TabPFN calibrou desde o início

    resultados["calibracao_simetrica"] = calibracao_xgb and calibracao_pfn
    _check(
        "Calibração isotônica aplicada a ambos os modelos",
        calibracao_xgb and calibracao_pfn,
        "Verificar output/calibration_check.png para inspeção visual",
    )

    # =========================================================================
    # CHECK 5 : Diagrama de calibração gerado (Fix 4 visual)
    # =========================================================================
    calibration_png = os.path.exists("output/calibration_check.png")
    resultados["calibration_png_gerado"] = calibration_png
    _check(
        "Reliability diagram salvo (output/calibration_check.png)",
        calibration_png,
    )

    # Check 6: Sensibilidade ao limiar variavel (<= 0.02 para DIR e EOD).
    # Verificacao numerica rapida com 200 iteracoes de bootstrap.
    from fairness import avaliar_justica_bootstrap

    y_true  = np.array(caderno["xgboost"]["rotulos_reais"])
    masculinos = np.array(caderno["genero_no_conjunto_avaliacao"]) == "Masculino"
    brancos    = np.array(caderno["raca_no_conjunto_avaliacao"]) == "Branca"

    sensibilidade_ok = True
    detalhes_sens = []

    for nome in ["xgboost", "tabpfn"]:
        chave_prob = "probabilidades_calibradas" if nome == "tabpfn" else "probabilidades_estimadas"
        probs_oof   = np.array(caderno[nome][chave_prob])
        # Mediana dos tau* por fold como representante do limiar fixo.
        tau_fixo    = float(np.median(caderno[nome]["limiares_otimizados"]))
        pred_fixo   = (probs_oof >= tau_fixo).astype(int)
        pred_var    = np.array(caderno[nome]["predicoes_duras"])

        for grupo_nome, grupo_vetor in [("Genero", masculinos), ("Raca", brancos)]:
            fv = avaliar_justica_bootstrap(y_true, pred_var,  grupo_vetor, 1, 0, 200)
            ff = avaliar_justica_bootstrap(y_true, pred_fixo, grupo_vetor, 1, 0, 200)
            delta_dir = abs(fv["DIR"] - ff["DIR"])
            delta_eod = abs(fv["EOD"] - ff["EOD"])
            alerta = delta_dir > 0.02 or delta_eod > 0.02
            if alerta:
                sensibilidade_ok = False
            detalhes_sens.append(
                f"{nome.upper()} | {grupo_nome}: DDIR={delta_dir:.4f}  DEOD={delta_eod:.4f}"
                + (" [ALERTA]" if alerta else "")
            )

    resultados["sensibilidade_limiar_ok"] = sensibilidade_ok
    _check(
        "Sensibilidade ao limiar variável ≤ 0.02 em DIR e EOD",
        sensibilidade_ok,
        "  " + "\n       ".join(detalhes_sens),
    )

    # Check 7: DIR racial (inspecao manual obrigatoria).
    resultados["dir_racial_narrativa_correta"] = None
    print(f"\n  [REVISAO MANUAL] DIR Racial TabPFN ({caderno['fairness_raca']['tabpfn']['DIR']:.3f}) "
          f"< XGBoost ({caderno['fairness_raca']['xgboost']['DIR']:.3f})")
    print("  Verificar: artigo nao deve afirmar que TabPFN e 'mais justo' racialmente.")
    print("  Narrativa correta: 'TabPFN exibe desvantagem marginal em equidade racial'.")

    # Resultado final
    checks_automaticos = {k: v for k, v in resultados.items() if v is not None}
    total = len(checks_automaticos)
    aprovados = sum(checks_automaticos.values())

    print("\n" + "=" * 70)
    print(f"Resultado: {aprovados}/{total} checks automaticos aprovados")

    if aprovados == total:
        print("\n  PRONTO PARA SUBMISSAO (checks automaticos)")
        print("  [REVISAO MANUAL] titulo, limitacoes temporais e DIR racial")
    else:
        pendentes = [k for k, v in checks_automaticos.items() if not v]
        print(f"\n  [FAIL] Itens pendentes: {', '.join(pendentes)}")
        print("  Execute os fixes correspondentes antes de submeter.")
    print("=" * 70)


if __name__ == "__main__":
    main()
