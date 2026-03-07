"""
Pipeline principal do experimento.

Chama, em sequência: carregamento de dados, pré-processamento, validação
cruzada estratificada e análise estatística/fairness.
"""

from data import carregar_dados_brutos, extrair_alvo_e_processar
from training import executar_validacao_cruzada_estratificada
from analysis import executar_bateria_estatistica_e_fairness
from visualization import gerar_paineis_estaticos

def fluxo_principal():
    print("TabPFN vs XGBoost: comparacao de fairness em eleicoes municipais do Ceara")
    print("=" * 70)

    print("\n[1/4] Carregando dados...")
    df_raw = carregar_dados_brutos()

    print("\n[2/4] Pre-processando...")
    df = extrair_alvo_e_processar(df_raw)

    print("\n[3/4] Treinando modelos (validacao cruzada K=10)...")
    results = executar_validacao_cruzada_estratificada(df)

    print("\n[4/4] Analise estatistica e fairness...")
    results = executar_bateria_estatistica_e_fairness(results)

    print("\n[5/5] Gerando figuras...")
    gerar_paineis_estaticos(results)

    print("\nConcluido.")


if __name__ == "__main__":
    fluxo_principal()
