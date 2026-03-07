"""
Parametros centrais do experimento.

Centralizar configuracoes garante reproducibilidade e elimina constantes
espalhadas no codigo. Altere apenas este modulo para modificar
hiperparametros globais.
"""

import pandas as pd
import numpy as np

# Configuracao do backend grafico para execucao sem interface grafica (servidores).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# SEMENTES E ESTRUTURA
SEMENTE_ALEATORIA = 42

# Numero de folds para validacao cruzada estratificada.
NUMERO_DE_FOLDS = 10

# LIMITES DO GRID DE THRESHOLD
# Expandido de [0.20, 0.80] para [0.05, 0.95] para que tau*(XGB)
# possa sair do boundary inferior (Fix 1).
LIMITE_MINIMO_DECISAO = 0.05
LIMITE_MAXIMO_DECISAO = 0.95
INCREMENTO_DE_BUSCA = 0.01


# COLUNAS DE ENTRADA
COLUNAS_CATEGORICAS = [
    "Cor/raça",
    "Estado civil",
    "Faixa etária",
    "Gênero",
    "Grau de instrução",
    "Ocupação",
    "Partido"
]

# Features preditivas: colunas categoricas + ano do pleito.
# 'Votos validos' e 'Votos nominais' removidos (resultado da eleicao,
# correlacao ~1.0 com o alvo : data leakage confirmado em auditoria).
COLUNAS_PREDITIVAS = COLUNAS_CATEGORICAS + [
    "eleicao"
]


# Estetica dos graficos.
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.autolayout": True,
})
