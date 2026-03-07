"""
Geracao de figuras para o artigo (PNG, 300 dpi, headless).

Produz:
- Curvas ROC por modelo
- Matrizes de confusao
- Graficos de fairness com IC bootstrap
- Diagramas de calibracao (reliability diagram)
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as _mpath
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

# Permite as parametrizações de folha (plt.style, sns.palette) da Configuração Central.
import config

# Patch de compatibilidade: Python 3.14 quebra o __deepcopy__ nativo do matplotlib.
# matplotlib.path.Path.__deepcopy__ chama copy.deepcopy(super(), memo), gerando
# recursao infinita no Python 3.14. Substituido por implementacao direta.
def _path_deepcopy_fix(self, memo):
    return _mpath.Path(
        copy.deepcopy(self.vertices, memo),
        copy.deepcopy(self.codes, memo),
        self._interpolation_steps,
        self.should_simplify,
        self.simplify_threshold,
    )

_mpath.Path.__deepcopy__ = _path_deepcopy_fix


def formatar_titulo_em_texto_visivel(eixo_matplotlib, p_text=""):
    """Aplica titulo em negrito ao eixo."""
    eixo_matplotlib.set_title(p_text, fontdict={'weight': 'bold', 'size': 13})


def desenhar_painel_de_limiares(eixo_ax, algorit_nome, proba, pred, y_true):
    """Plota a curva ROC com AUC para o modelo especificado."""
    taxa_falsos_positivos, taxa_verdadeiros_positivos, _ = roc_curve(y_true, proba)
    from sklearn.metrics import roc_auc_score
    area_real = roc_auc_score(y_true, proba)

    eixo_ax.plot(taxa_falsos_positivos, taxa_verdadeiros_positivos, linewidth=2.5, label=f"AUC = {area_real:.3f}")
    eixo_ax.plot([0, 1], [0, 1], color='red', linestyle='dotted', label='Referencia aleatoria', linewidth=1.5)
    eixo_ax.set_xlabel("Taxa de Falsos Positivos")
    eixo_ax.set_ylabel("Taxa de Verdadeiros Positivos")
    formatar_titulo_em_texto_visivel(eixo_ax, f"Curva ROC - {algorit_nome.upper()}")
    eixo_ax.legend(loc="lower right")


def fabricar_matriz_erros(eixo_ax, algorit_nome, pred_duras, y_true):
    """Plota a matriz de confusao para o modelo especificado."""
    quadrantes = confusion_matrix(y_true, pred_duras)
    
    sns.heatmap(
        quadrantes, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        ax=eixo_ax,
        linewidths=.5,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    eixo_ax.set_ylabel("Real")
    eixo_ax.set_xlabel("Previsto")
    eixo_ax.set_xticklabels(["0", "1"])
    eixo_ax.set_yticklabels(["0", "1"])
    formatar_titulo_em_texto_visivel(eixo_ax, f"Matriz de Confusao - {algorit_nome.upper()}")


def extrair_painel_de_fairness(eixo_ax, registros_analiticos, metrica_alvo, label_eixo, limite_justica_ideal=None):
    """Plota barras com IC bootstrap para uma metrica de fairness (DIR, EOD ou FNR_DIFF)."""
    modelos_aprovados = ["xgboost", "tabpfn"]
    cores = ["#50505A", "#6A4C93"]  # XGBoost=cinza, TabPFN=roxo
    
    # Extrai o ponto central avaliado e o raio inferior e superior do intervalo de confiança de 95%
    medias = [registros_analiticos[m][metrica_alvo] for m in modelos_aprovados]
    limites_inferiores = [registros_analiticos[m][f"{metrica_alvo}_IC"][0] for m in modelos_aprovados]
    limites_superiores = [registros_analiticos[m][f"{metrica_alvo}_IC"][1] for m in modelos_aprovados]
    
    # Distancia relativa para as barras de erro.
    erros = [
        [m - li for m, li in zip(medias, limites_inferiores)],
        [ls - m for m, ls in zip(medias, limites_superiores)]
    ]

    painel_barras = eixo_ax.bar(modelos_aprovados, medias, yerr=erros, color=cores, capsize=6, alpha=0.9, width=0.6)
    eixo_ax.set_ylabel(label_eixo)
    formatar_titulo_em_texto_visivel(eixo_ax, f"Métrica de Equidade: {metrica_alvo.replace('_DIFF', '')}")
    
    if limite_justica_ideal is not None:
        eixo_ax.axhline(y=limite_justica_ideal, color='red', linestyle='dotted', label=f'Referencia ideal ({limite_justica_ideal})', linewidth=2)
        eixo_ax.legend(loc="upper right")
        
    for haste, valor_media in zip(painel_barras, medias):
        ainda_superior = haste.get_height()
        eixo_ax.text(
            haste.get_x() + haste.get_width() / 2,
            valor_media if valor_media > 0 else valor_media - 0.03,
            f"{valor_media:.3f}",
            ha="center",
            va="bottom" if valor_media > 0 else "top",
            fontweight="bold"
        )


def gerar_diagrama_calibracao(eixo_ax, nome_maquina, probs, y_true, n_bins=10):
    """
    Reliability diagram para verificacao visual de calibracao (Fix 4).

    Curva ideal: probabilidade predita = frequencia empirica real.
    Desvios da diagonal indicam over/under-confidence sistematico.
    """
    frac_pos, media_pred = calibration_curve(y_true, probs, n_bins=n_bins)
    eixo_ax.plot(media_pred, frac_pos, marker='o', linewidth=2, label=nome_maquina.upper())
    eixo_ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfeita')
    eixo_ax.set_xlabel("Probabilidade Predita")
    eixo_ax.set_ylabel("Fracao de Positivos Reais")
    formatar_titulo_em_texto_visivel(eixo_ax, f"Calibracao - {nome_maquina.upper()}")
    eixo_ax.legend(loc="upper left")


def gerar_paineis_estaticos(livro_registros):
    """Gera e salva todos os graficos do experimento em output/*.png."""
    diretorio_saida = "output"
    os.makedirs(diretorio_saida, exist_ok=True)
    
    # Grafico 1: Curvas ROC
    fig, quadros = plt.subplots(1, 2, figsize=(14, 6))
    for i, nome_maquina in enumerate(["xgboost", "tabpfn"]):
        chave_prob = "probabilidades_calibradas" if nome_maquina == "tabpfn" else "probabilidades_estimadas"
        desenhar_painel_de_limiares(
            quadros[i], 
            nome_maquina, 
            np.array(livro_registros[nome_maquina][chave_prob]), 
            np.array(livro_registros[nome_maquina]["predicoes_duras"]), 
            np.array(livro_registros[nome_maquina]["rotulos_reais"])
        )
    plt.tight_layout()
    plt.savefig(f"{diretorio_saida}/roc_mcc.png", dpi=300)
    plt.close()
    
    # Grafico 2: Matrizes de confusao
    fig2, quadros_matrizes = plt.subplots(1, 2, figsize=(12, 5))
    for i, nome_maquina in enumerate(["xgboost", "tabpfn"]):
        fabricar_matriz_erros(
            quadros_matrizes[i], 
            nome_maquina, 
            np.array(livro_registros[nome_maquina]["predicoes_duras"]), 
            np.array(livro_registros[nome_maquina]["rotulos_reais"])
        )
    plt.tight_layout()
    plt.savefig(f"{diretorio_saida}/confusion_matrices.png", dpi=300)
    plt.close()
    
    # Grafico 3: Fairness com IC bootstrap
    metodos_auditados = [
        ("DIR", "Disparate Impact Ratio (DIR)", 1.0),
        ("EOD", "Equal Opportunity Difference (EOD)", 0.0),
        ("FNR_DIFF", "FNR Difference", 0.0)
    ]
    
    for nome_social, pacotao_do_livro in [("Genero", "fairness_genero"), ("Raca", "fairness_raca")]:
        fig_fairness, quadros_sociais = plt.subplots(1, 3, figsize=(16, 5))
        fig_fairness.suptitle(f"Fairness por {nome_social}", fontweight="bold", fontsize=16)
        
        for idx_painel, (sigla_metrica, eiqueta_label, linha_ideal) in enumerate(metodos_auditados):
            extrair_painel_de_fairness(
                quadros_sociais[idx_painel],
                livro_registros[pacotao_do_livro],
                sigla_metrica,
                eiqueta_label,
                limite_justica_ideal=linha_ideal
            )
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{diretorio_saida}/fairness_metrics_{nome_social.lower()}.png", dpi=300)
        plt.close()

    # Grafico 4: Reliability diagram (calibracao)
    fig_cal, quadros_cal = plt.subplots(1, 2, figsize=(12, 5))
    fig_cal.suptitle("Calibracao das Probabilidades", fontweight="bold", fontsize=14)
    for i, nome_maquina in enumerate(["xgboost", "tabpfn"]):
        chave_prob = "probabilidades_calibradas" if nome_maquina == "tabpfn" else "probabilidades_estimadas"
        gerar_diagrama_calibracao(
            quadros_cal[i],
            nome_maquina,
            np.array(livro_registros[nome_maquina][chave_prob]),
            np.array(livro_registros[nome_maquina]["rotulos_reais"]),
        )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{diretorio_saida}/calibration_check.png", dpi=300)
    plt.close()

    print("[5/5] Figuras salvas em output/")
