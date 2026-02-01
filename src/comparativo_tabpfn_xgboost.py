import pandas as pd
import numpy as np
import xgboost as xgb
from tabpfn import TabPFNClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, average_precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print(">>> Iniciando Pipeline de Comparação: TabPFN v2.5 vs XGBoost (Absenteísmo de Mesários)")
    
    # ---------------------------------------------------------
    # 1. Carregamento e Limpeza
    # ---------------------------------------------------------
    try:
        # Carregando com encoding latin1 e separador ;
        df = pd.read_csv('../data/convocacao_mesarios_2024_CE.csv', sep=';', encoding='latin1')
        print(f"Dataset bruto carregado. Linhas (perfis): {len(df)}")
    except FileNotFoundError:
        print("ERRO: Arquivo 'convocacao_mesarios_2024_CE.csv' não encontrado.")
        return

    # Tratamento de '#NULO' como categoria
    df = df.fillna('#NULO')
    
    # EXPANSÃO DO DATASET (Desagregando perfis)
    if 'QT_CONVOCADOS_PERFIL' in df.columns:
        print("Expandindo dataset baseado em 'QT_CONVOCADOS_PERFIL'...")
        # Repete as linhas
        df = df.loc[df.index.repeat(df['QT_CONVOCADOS_PERFIL'])].reset_index(drop=True)
        print(f"Dataset expandido. Total de mesários individuais: {len(df)}")
    
    # Limitação para 100k amostras (Requisito TabPFN v2.5 / Instrução User)
    if len(df) > 100000:
        print("Limitando dataset a 100.000 amostras (Sorteio aleatório)...")
        df = df.sample(n=100000, random_state=42).reset_index(drop=True)

    # Definição de Target
    # O user quer prever FALTAR (Ausência).
    # Originalmente: Compareceu=1, Faltou=0.
    # Mas para focar métricas na classe minoritária (Falta), definimos Falta=1.
    
    # Verificando a coluna de target
    if 'ST_COMPARECIMENTO' in df.columns:
        # Mapeamento: 'Ausências' -> 1 (Alvo), 'Comparecimentos' -> 0
        y = df['ST_COMPARECIMENTO'].apply(lambda x: 1 if x.strip() == 'Ausências' else 0)
        # Remove colunas auxiliares
        X = df.drop(columns=['ST_COMPARECIMENTO', 'QT_CONVOCADOS_PERFIL', 'DT_GERACAO', 'HH_GERACAO'], errors='ignore')
    elif 'CD_COMPARECIMENTO' in df.columns:
         y = (df['CD_COMPARECIMENTO'] == 0).astype(int)
         X = df.drop(columns=['CD_COMPARECIMENTO', 'QT_CONVOCADOS_PERFIL'], errors='ignore')
    else:
        raise ValueError("Coluna de target (ST_COMPARECIMENTO ou CD_COMPARECIMENTO) não encontrada.")

    # Pré-processamento: Ordinal Encoding
    print("Codificando variáveis categóricas...")
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])

    # ---------------------------------------------------------
    # 2. Split (80/20) - Sem SMOTE
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"Taxa de Ausência (Classe 1 após inversão) no Treino: {y_train.mean():.2%}")

    # ---------------------------------------------------------
    # 3. XGBoost Baseline (scale_pos_weight)
    # ---------------------------------------------------------
    print("\n--- Treinando Baseline: XGBoost ---")
    # scale_pos_weight = sum(negative) / sum(positive)
    # Como invertemos: Negative = Compareceu (Maioria), Positive = Faltou (Minoria)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Razão de Desbalanceamento calculada: {ratio:.2f}")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio, # Crucial para o desbalanceamento
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_clf.fit(X_train, y_train)
    
    y_pred_xgb = xgb_clf.predict(X_test)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 4. TabPFN (Challenger)
    print("\n--- Treinando Challenger: TabPFN v2.5 ---")
    
    # Flags para controle de sucesso
    tabpfn_success = False
    y_pred_tab = None
    y_prob_tab = None

    # Treino
    # TabPFN tem tempo de inferência O(N^2) ou O(N) dependendo da implementação e overhead na CPU.
    # Com 80k samples, na v2.5, pode levar horas em CPU sem destilação apropriada.
    # Para validar o funcionamento AGORA, vamos usar um subset representativo.
    # Treino
    # Com GPU (CUDA), podemos usar o dataset completo, mas o tempo pode ser longo para 80k.
    # Vamos usar 10.000 amostras para garantir resposta rápida sem sacrificar muita performance (TabPFN satura rápido).
    print("DEMO MODE: Usando subset de 10.000 amostras para o TabPFN (GPU)...")
    subset_idx = np.random.choice(len(X_train), 10000, replace=False)
    X_train_sub = X_train.iloc[subset_idx]
    y_train_sub = y_train.iloc[subset_idx]
    
    try:
        # TabPFN v2.xx usa n_estimators em vez de N_ensemble_configurations
        # device='cuda' para usar a placa NVIDIA
        # balance_probabilities=True: CRÍTICO para dataset desbalanceado (faz o papel do class_weight='balanced')
        tabpfn_clf = TabPFNClassifier(device='cuda', n_estimators=32, ignore_pretraining_limits=True, balance_probabilities=True) 
        
        print("Treinando TabPFN (pode exigir autenticação HuggingFace na 1ª execução)...")
        tabpfn_clf.fit(X_train_sub, y_train_sub)
        
        y_pred_tab = tabpfn_clf.predict(X_test)
        y_prob_tab = tabpfn_clf.predict_proba(X_test)[:, 1]
        tabpfn_success = True

    except Exception as e:
        print("\n" + "!"*60)
        print(f"ERRO TABPFN: {e}")
        print("Para usar o TabPFN v2.5, você precisa aceitar os termos em https://huggingface.co/Prior-Labs/tabpfn_2_5")
        print("e fazer login com 'huggingface-cli login'.")
        print("!"*60 + "\n")
        # Segue sem o TabPFN

    # ---------------------------------------------------------
    # 5. Avaliação e Comparação
    # ---------------------------------------------------------
    
    def get_metrics(y_true, y_pred, y_prob, name):
        if y_pred is None:
            return {'Model': name, 'MCC': -1, 'AUPRC': -1, 'Recall_Faltosos': 0}
            
        mcc = matthews_corrcoef(y_true, y_pred)
        auprc = average_precision_score(y_true, y_prob)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall_minoria = tp / (tp + fn) # Recall da classe positiva (falta)
        precision_minoria = tp / (tp + fp) if (tp+fp) > 0 else 0
        
        print(f"\n>> {name} <<")
        print(f"MCC: {mcc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Recall (Faltosos): {recall_minoria:.2%}")
        print(f"Precision (Faltosos): {precision_minoria:.2%}")
        print("Matriz de Confusão:")
        print(cm)
        
        return {'Model': name, 'MCC': mcc, 'AUPRC': auprc, 'Recall_Faltosos': recall_minoria}

    metrics_xgb = get_metrics(y_test, y_pred_xgb, y_prob_xgb, "XGBoost (Baseline)")
    
    if tabpfn_success:
        metrics_tab = get_metrics(y_test, y_pred_tab, y_prob_tab, "TabPFN v2.5")
    else:
        metrics_tab = {'Model': "TabPFN v2.5 (Falha Auth)", 'MCC': 0.0, 'AUPRC': 0.0, 'Recall_Faltosos': 0.0}

    # ---------------------------------------------------------
    # 6. Tabela Comparativa Final
    # ---------------------------------------------------------
    results = pd.DataFrame([metrics_xgb, metrics_tab])
    
    # Definindo o 'Presumivelmente Melhor' baseado no MCC (mais robusto que F1 para desbalanceados)
    best_mcc = results.loc[results['MCC'].idxmax()]
    
    print("\n" + "="*60)
    print("TABELA COMPARATIVA DE PERFORMANCE (UFCA STYLE)")
    print("="*60)
    print(results.to_markdown(index=False, floatfmt=".4f"))
    print("="*60)
    print(f"CONCLUSÃO: O modelo '{best_mcc['Model']}' é presumivelmente melhor,")
    print(f"apresentando um MCC de {best_mcc['MCC']:.4f} e AUPRC de {best_mcc['AUPRC']:.4f}.")
    print("="*60)

if __name__ == "__main__":
    main()
