"""
Analise estatistica e de fairness dos resultados de validacao cruzada.

Recebe o dicionario de resultados produzido por training.py e calcula:
- MCC medio e AUC-ROC por modelo
- Testes estatisticos: Wilcoxon (suplementar), Cohen d, McNemar
- Fairness por genero e raca: DIR, EOD, FNR_DIFF com bootstrap CI
- Sensibilidade ao threshold variavel
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from scipy.stats import wilcoxon

from fairness import avaliar_justica_bootstrap, teste_de_efeito_cohens, teste_discordancia_mcnemar, redigir_relatorio_fairness
from config import NUMERO_DE_FOLDS

def executar_bateria_estatistica_e_fairness(livro_registros: dict) -> dict:
    """
    Calcula metricas de desempenho e fairness a partir do dicionario de
    resultados produzido por training.py. Atualiza livro_registros com
    chaves 'analises_de_sucesso', 'fairness_genero' e 'fairness_raca'.
    """
    print("\n" + "=" * 60)
    print("MCC por modelo (validacao cruzada K=10)")
    print("=" * 60)

    resultados_analiticos = {}

    for nome_do_algoritmo in ["xgboost", "tabpfn"]:
        y_labels = np.array(livro_registros[nome_do_algoritmo]["rotulos_reais"])
        predicoes_binarias = np.array(livro_registros[nome_do_algoritmo]["predicoes_duras"])
        chave_prob = "probabilidades_calibradas" if nome_do_algoritmo == "tabpfn" else "probabilidades_estimadas"
        probabilidades = np.array(livro_registros[nome_do_algoritmo][chave_prob])

        mcc_folds = livro_registros[nome_do_algoritmo]["mcc_folds"]

        # Verifica que o caderno contem exatamente K entradas de MCC.
        assert len(mcc_folds) == NUMERO_DE_FOLDS, (
            f"[ASSERT] {nome_do_algoritmo.upper()}: esperados {NUMERO_DE_FOLDS} folds de MCC, "
            f"encontrados {len(mcc_folds)}. Caderno de resultados corrompido."
        )

        media_geral_mcc = float(np.mean(mcc_folds))
        desvio_padrao_mcc = float(np.std(mcc_folds))
        area_curva_roc = float(roc_auc_score(y_labels, probabilidades))

        resultados_analiticos[nome_do_algoritmo] = {
            "MCC_Media": media_geral_mcc,
            "MCC_Desvio": desvio_padrao_mcc,
            "AUC_ROC": area_curva_roc,
            "MCC_Blocos": mcc_folds
        }
        
        print(f"\n{nome_do_algoritmo.upper()}")
        print(f"  MCC: {media_geral_mcc:.4f} (+/- {desvio_padrao_mcc:.4f})")

    # Vetores de referencia compartilhados
    y_true = np.array(livro_registros["xgboost"]["rotulos_reais"])
    is_masculino = np.array(livro_registros["genero_no_conjunto_avaliacao"]) == "Masculino"
    is_branca    = np.array(livro_registros["raca_no_conjunto_avaliacao"])   == "Branca"

    livro_registros["analises_de_sucesso"] = resultados_analiticos
    
    print("\n" + "=" * 60)
    print("Testes Estatisticos")
    print("=" * 60)
    
    bloco_mcc_x = np.array(resultados_analiticos["xgboost"]["MCC_Blocos"])
    bloco_mcc_y = np.array(resultados_analiticos["tabpfn"]["MCC_Blocos"])

    estat_wilcox, p_valor_wilcox = wilcoxon(bloco_mcc_x, bloco_mcc_y)
    
    # Cohen d com IC bootstrap (n=10 insuficiente para formula analitica).
    d_cohen = teste_de_efeito_cohens(bloco_mcc_y, bloco_mcc_x)

    # IC 95% do Cohen d via bootstrap.\n    np.random.seed(42)
    n_boot = 5000
    d_boot = []
    n = len(bloco_mcc_x)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        d_boot.append(teste_de_efeito_cohens(bloco_mcc_y[idx], bloco_mcc_x[idx]))
    d_ic_inf = float(np.percentile(d_boot, 2.5))
    d_ic_sup = float(np.percentile(d_boot, 97.5))
    
    # McNemar: quais candidatos um modelo acertou e o outro errou.
    pred_xgb = np.array(livro_registros["xgboost"]["predicoes_duras"])
    pred_tab = np.array(livro_registros["tabpfn"]["predicoes_duras"])
    est_mcn, p_mcn, b_tab, b_xgb = teste_discordancia_mcnemar(y_true, pred_tab, pred_xgb)
    
    print(f"1. Wilcoxon (suplementar): W={estat_wilcox:.3f}, p={p_valor_wilcox:.4f}")
    print(f"2. Cohen d: {d_cohen:.3f}  IC-95% [{d_ic_inf:.3f}, {d_ic_sup:.3f}]")
    print(f"3. McNemar: estatistica={est_mcn:.3f}, p={p_mcn:.4f}")
    print(f"   b (TabPFN correto, XGBoost errado): {b_tab}  |  c (XGBoost correto, TabPFN errado): {b_xgb}")
    print(f"4. AUC-ROC: XGBoost={resultados_analiticos['xgboost']['AUC_ROC']:.4f}, TabPFN={resultados_analiticos['tabpfn']['AUC_ROC']:.4f}")

    print("\n" + "=" * 60)
    print("Fairness por Genero")
    print("=" * 60)

    livro_registros["fairness_genero"] = {}
    for algorit_nome in ["xgboost", "tabpfn"]:
        pred_atual = np.array(livro_registros[algorit_nome]["predicoes_duras"])
        
        pacote_botstrap = avaliar_justica_bootstrap(
            rotulos=y_true,
            predicoes=pred_atual,
            grupo_demografico=is_masculino,
            id_privilegiado=1,
            id_nao_privilegiado=0
        )
        livro_registros["fairness_genero"][algorit_nome] = pacote_botstrap
        redigir_relatorio_fairness(pacote_botstrap, algorit_nome.upper(), np.sum(is_masculino==1), np.sum(is_masculino==0))
        

    print("\n" + "=" * 60)
    print("Fairness por Raca")
    print("=" * 60)

    livro_registros["fairness_raca"] = {}
    
    for algorit_nome in ["xgboost", "tabpfn"]:
        pred_atual = np.array(livro_registros[algorit_nome]["predicoes_duras"])
        
        pacote_botstrap = avaliar_justica_bootstrap(
            rotulos=y_true,
            predicoes=pred_atual,
            grupo_demografico=is_branca,
            id_privilegiado=1,
            id_nao_privilegiado=0
        )
        livro_registros["fairness_raca"][algorit_nome] = pacote_botstrap
        redigir_relatorio_fairness(pacote_botstrap, algorit_nome.upper(), np.sum(is_branca==1), np.sum(is_branca==0))

    # =========================================================================
    # Fix 5: sensibilidade ao limiar variavel
    # =========================================================================
    # As metricas de fairness foram calculadas com predicoes geradas por
    # limiares tau* distintos por fold. Para verificar se essa variabilidade
    # distorce DIR/EOD, recalculamos as metricas usando um limiar fixo unico
    # (mediana dos tau* do treino) e reportamos a diferenca maxima observada.
    print("\n" + "=" * 60)
    print("Sensibilidade ao limiar variavel (Fix 5)")
    print("=" * 60)

    LIMIAR_SENSIBILIDADE = 0.02  # limiar de alerta para variacao de fairness

    for algorit_nome in ["xgboost", "tabpfn"]:
        chave_prob = "probabilidades_calibradas" if algorit_nome == "tabpfn" else "probabilidades_estimadas"
        probs_oof   = np.array(livro_registros[algorit_nome][chave_prob])
        taus_treino = np.array(livro_registros[algorit_nome]["limiares_otimizados"])
        tau_fixo    = float(np.median(taus_treino))

        pred_fixo = (probs_oof >= tau_fixo).astype(int)
        pred_var  = np.array(livro_registros[algorit_nome]["predicoes_duras"])

        alerta_genero = False
        alerta_raca   = False

        for grupo_nome, grupo_vetor, grupo_priv, grupo_nao_priv in [
            ("Genero",  is_masculino, 1, 0),
            ("Raca",    is_branca,    1, 0),
        ]:
            fair_var  = avaliar_justica_bootstrap(
                rotulos=y_true, predicoes=pred_var,
                grupo_demografico=grupo_vetor, id_privilegiado=grupo_priv,
                id_nao_privilegiado=grupo_nao_priv, iteracoes_bootstrap=500,
            )
            fair_fixo = avaliar_justica_bootstrap(
                rotulos=y_true, predicoes=pred_fixo,
                grupo_demografico=grupo_vetor, id_privilegiado=grupo_priv,
                id_nao_privilegiado=grupo_nao_priv, iteracoes_bootstrap=500,
            )

            delta_dir = abs(fair_var["DIR"]     - fair_fixo["DIR"])
            delta_eod = abs(fair_var["EOD"]     - fair_fixo["EOD"])
            delta_fnr = abs(fair_var["FNR_DIFF"] - fair_fixo["FNR_DIFF"])
            alerta    = delta_dir > LIMIAR_SENSIBILIDADE or delta_eod > LIMIAR_SENSIBILIDADE

            print(f"  {algorit_nome.upper()} | {grupo_nome} | tau_var med={tau_fixo:.2f}")
            print(f"    DDIR={delta_dir:.4f}  DEOD={delta_eod:.4f}  DFNR={delta_fnr:.4f}", end="  ")
            print(f"{'[ALERTA: >0.02]' if alerta else '[OK]'}")

        # Salva tau fixo no livro
        livro_registros[algorit_nome]["tau_medio_treino"] = tau_fixo

    print("[Nota] Se DDIR ou DEOD > 0.02, reportar ambos os valores no artigo.")

    return livro_registros
