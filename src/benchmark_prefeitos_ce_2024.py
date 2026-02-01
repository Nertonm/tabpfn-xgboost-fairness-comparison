import pandas as pd
import numpy as np
import xgboost as xgb
from tabpfn import TabPFNClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wilcoxon
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO
# ------------------------------------------------------------------------------
DATA_PATH = '../data/votacao_candidato.csv'
TARGET_COL = 'ST_ELEITO'
SENSITIVE_GENERO = 'DS_GENERO'
SENSITIVE_RACA = 'DS_COR_RACA'

# ------------------------------------------------------------------------------
# UTILITÁRIOS DE FAIRNESS
# ------------------------------------------------------------------------------
def calculate_dir(y_pred, sensitive_values, privileged_group):
    """
    Disparate Impact Ratio (DIR).
    DIR = P(Y=1 | Unprivileged) / P(Y=1 | Privileged)
    Ideal: 1.0. < 0.8 implies bias against unprivileged.
    """
    y_pred = np.array(y_pred)
    sens = np.array(sensitive_values)
    
    priv_mask = (sens == privileged_group)
    unpriv_mask = ~priv_mask
    
    if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
        return np.nan
        
    prob_priv = y_pred[priv_mask].mean()
    prob_unpriv = y_pred[unpriv_mask].mean()
    
    if prob_priv == 0:
        return 0.0 if prob_unpriv == 0 else np.inf
        
    return prob_unpriv / prob_priv

def calculate_eod(y_true, y_pred, sensitive_values, privileged_group):
    """
    Equal Opportunity Difference (EOD).
    EOD = TPR(Unprivileged) - TPR(Privileged)
    Ideal: 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sens = np.array(sensitive_values)
    
    priv_y1_mask = (sens == privileged_group) & (y_true == 1)
    unpriv_y1_mask = (sens != privileged_group) & (y_true == 1)
    
    if priv_y1_mask.sum() == 0 or unpriv_y1_mask.sum() == 0:
        return np.nan
        
    tpr_priv = y_pred[priv_y1_mask].mean()
    tpr_unpriv = y_pred[unpriv_y1_mask].mean()
    
    return tpr_unpriv - tpr_priv

# ------------------------------------------------------------------------------
# DATA LOADING & GENERATION
# ------------------------------------------------------------------------------
def load_and_prep_data():
    print(">>> 1. Carregando Dados...")
    df = None
    target_mapping = {'Eleito': 1, 'Não Eleito': 0}
    
    try:
        df_raw = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')
        
        rename_map = {
            'Situação totalização': TARGET_COL,
            'Gênero': SENSITIVE_GENERO,
            'Cor/raça': SENSITIVE_RACA,
            'Grau de instrução': 'DS_GRAU_INSTRUCAO',
            'Estado civil': 'DS_ESTADO_CIVIL',
            'Faixa etária': 'DS_FAIXA_ETARIA',
            'Ocupação': 'DS_OCUPACAO',
            'Partido': 'SG_PARTIDO',
            'Votos válidos': 'QT_VOTOS'
        }
        df_raw = df_raw.rename(columns=rename_map)
        
        if 'Cargo' in df_raw.columns:
            df_raw = df_raw[df_raw['Cargo'] == 'Prefeito']
        
        target_valid = False
        if TARGET_COL in df_raw.columns:
            if df_raw[TARGET_COL].dtype == object:
                 unique_vals = df_raw[TARGET_COL].unique()
                 if 'Eleito' in unique_vals:
                     df_raw[TARGET_COL] = df_raw[TARGET_COL].map(target_mapping)
            
            counts = df_raw[TARGET_COL].value_counts()
            if len(counts) >= 2:
                target_valid = True
            else:
                 print(f"ALERTA: Dataset contém apenas uma classe: {counts.to_dict()}.")
                 print("Motivo: O arquivo 'votacao_candidato.csv' parece conter apenas os eleitos.")

        if target_valid:
            df = df_raw
            print("Dataset carregado com sucesso.")
        else:
             raise ValueError("Dataset inviável para classificação binária.")
            
    except Exception as e:
        print(f"Erro ao carregar/processar dataset original: {e}")
        print(">>> ATIVANDO GERAÇÃO DE DADOS SINTÉTICOS (Baseado no Notebook Entrega 1.2)")
        
        np.random.seed(42)
        n_samples = 471
        
        generos = np.random.choice(['Masculino', 'Feminino'], n_samples, p=[0.794, 0.206])
        racas = np.random.choice(['Branca', 'Parda', 'Preta', 'Outros'], n_samples, p=[0.535, 0.426, 0.030, 0.009])
        
        instrucao = np.random.choice(['Superior', 'Medio', 'Fundamental'], n_samples, p=[0.6, 0.3, 0.1])
        partido = np.random.choice(['PT', 'PDT', 'PSD', 'MDB', 'PL', 'PP'], n_samples)
        
        logits = -0.5 
        logits += np.where(generos == 'Masculino', 0.2, -0.1)
        logits += np.where(racas == 'Branca', 0.1, 0.0)
        logits += np.where(instrucao == 'Superior', 0.3, -0.1)
        
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        
        df = pd.DataFrame({
            SENSITIVE_GENERO: generos,
            SENSITIVE_RACA: racas,
            'DS_GRAU_INSTRUCAO': instrucao,
            'SG_PARTIDO': partido,
            TARGET_COL: y
        })
        
        print(f"Dataset sintético gerado. Shape: {df.shape}")
        print(f"Distribuição Target: {df[TARGET_COL].value_counts().to_dict()}")

    le_gen = LabelEncoder()
    df[SENSITIVE_GENERO] = le_gen.fit_transform(df[SENSITIVE_GENERO].astype(str))
    
    le_raca = LabelEncoder()
    df[SENSITIVE_RACA] = le_raca.fit_transform(df[SENSITIVE_RACA].astype(str))
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
    return df

# ------------------------------------------------------------------------------
# MODELOS
# ------------------------------------------------------------------------------

def train_xgboost(X_train, y_train):
    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [sum(y_train==0)/sum(y_train==1)]
    }
    
    xgb_clf = None
    try:
        xgb_clf = xgb.XGBClassifier(
            eval_metric='logloss', 
            device='cuda', 
            tree_method='hist',
            use_label_encoder=False, 
            random_state=42,
            base_score=0.5
        )
        dummy_X = pd.DataFrame(np.random.rand(10, X_train.shape[1]), columns=X_train.columns)
        dummy_y = pd.Series([0, 1] * 5)
        xgb_clf.fit(dummy_X, dummy_y)
    except Exception as e:
        print(f"Aviso: XGBoost falhou com GPU ({e}). Revertendo para CPU.")
        xgb_clf = xgb.XGBClassifier(eval_metric='logloss', device='cpu', use_label_encoder=False, random_state=42)
    
    search = RandomizedSearchCV(
        xgb_clf, params, n_iter=10, scoring='matthews_corrcoef', cv=3, random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def get_tabpfn_model():
    print("Inicializando TabPFN em CUDA (se disponível)...")
    try:
        return TabPFNClassifier(device='cuda', n_estimators=32, ignore_pretraining_limits=True, balance_probabilities=True)
    except Exception as e:
        print(f"Erro ao init TabPFN v2 (cuda): {e}. Tentando CPU...")
        try:
             return TabPFNClassifier(device='cpu', n_estimators=32, ignore_pretraining_limits=True, balance_probabilities=True)
        except:
             return TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

# ------------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ------------------------------------------------------------------------------
def main():
    print("--- INICIANDO BENCHMARK RIGOROSO: XGBOOST vs TABPFN ---")
    
    # 1. Dados e validação de isolamento de fatores
    df = load_and_prep_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Grupos sensíveis
    priv_gender = 1 # Masculino
    priv_race = 0   # Branca
    
    # Validação rigorosa: Stratified 5-Fold com Random Seed fixo
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'XGBoost': {'mcc': [], 'dir_gender': [], 'eod_gender': [], 'dir_race': [], 'eod_race': []},
        'TabPFN': {'mcc': [], 'dir_gender': [], 'eod_gender': [], 'dir_race': [], 'eod_race': []},
        'XGBoost_Blind': {'mcc': []} # Para teste de vazamento
    }
    
    tabpfn = get_tabpfn_model()
    
    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        print(f"\nProcessing Fold {fold}/5...")
        
        # Split com isolamento garantido pelo seed 42
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # --- MODELO A: XGBOOST ---
        print("  Treinando XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        preds_xgb = xgb_model.predict(X_test)
        
        # --- MODELO B: TABPFN ---
        print("  Inferindo TabPFN...")
        tabpfn.fit(X_train, y_train)
        preds_tab = tabpfn.predict(X_test)
        
        # --- LEAKAGE TEST (XGBoost Blind) ---
        print("  Treinando XGBoost Blind (Leakage Test)...")
        # Remover colunas sensíveis
        X_train_blind = X_train.drop(columns=[SENSITIVE_GENERO, SENSITIVE_RACA])
        X_test_blind = X_test.drop(columns=[SENSITIVE_GENERO, SENSITIVE_RACA])
        xgb_blind = train_xgboost(X_train_blind, y_train)
        preds_xgb_blind = xgb_blind.predict(X_test_blind)
        
        # --- MÉTRICAS ---
        sens_gen_test = X_test[SENSITIVE_GENERO]
        sens_race_test = X_test[SENSITIVE_RACA]
        
        # XGBoost Normal
        results['XGBoost']['mcc'].append(matthews_corrcoef(y_test, preds_xgb))
        results['XGBoost']['dir_gender'].append(calculate_dir(preds_xgb, sens_gen_test, priv_gender))
        results['XGBoost']['eod_gender'].append(calculate_eod(y_test, preds_xgb, sens_gen_test, priv_gender))
        results['XGBoost']['dir_race'].append(calculate_dir(preds_xgb, sens_race_test, priv_race))
        results['XGBoost']['eod_race'].append(calculate_eod(y_test, preds_xgb, sens_race_test, priv_race))
        
        # TabPFN Normal
        results['TabPFN']['mcc'].append(matthews_corrcoef(y_test, preds_tab))
        results['TabPFN']['dir_gender'].append(calculate_dir(preds_tab, sens_gen_test, priv_gender))
        results['TabPFN']['eod_gender'].append(calculate_eod(y_test, preds_tab, sens_gen_test, priv_gender))
        results['TabPFN']['dir_race'].append(calculate_dir(preds_tab, sens_race_test, priv_race))
        results['TabPFN']['eod_race'].append(calculate_eod(y_test, preds_tab, sens_race_test, priv_race))
        
        # XGBoost Blind
        results['XGBoost_Blind']['mcc'].append(matthews_corrcoef(y_test, preds_xgb_blind))

    # --- ESTATÍSTICA (Wilcoxon) ---
    print("\n" + "="*80)
    print("ANÁLISE ESTATÍSTICA (WILCOXON SIGNED-RANK TEST)")
    print("="*80)
    
    # MCC Comparison
    try:
        stat, p_value = wilcoxon(results['XGBoost']['mcc'], results['TabPFN']['mcc'])
        print(f"MCC Comparison (XGBoost vs TabPFN): p-value = {p_value:.5f}")
        if p_value < 0.05:
            print(">> Resultado SIGNIFICATIVO: Diferença estatística real entre os modelos.")
        else:
            print(">> Resultado NÃO SIGNIFICATIVO: A diferença pode ser acaso.")
    except Exception as e:
        print(f"Erro no Wilcoxon: {e}")

    # --- LEAKAGE ANALYSIS ---
    print("\n" + "="*80)
    print("ANÁLISE DE DATA LEAKAGE (Dependência de Atributos Sensíveis)")
    print("="*80)
    
    mcc_xgb = np.mean(results['XGBoost']['mcc'])
    mcc_blind = np.mean(results['XGBoost_Blind']['mcc'])
    diff_leakage = mcc_xgb - mcc_blind
    
    print(f"MCC XGBoost (Completo): {mcc_xgb:.4f}")
    print(f"MCC XGBoost (Blind - Sem Gênero/Raça): {mcc_blind:.4f}")
    print(f"Impacto da Remoção: {diff_leakage:.4f}")
    
    if diff_leakage > 0.05:
         print(">> ALERTA: Queda significativa de performance sem atributos protegidos.")
         print(">> Indício de que o modelo usa Gênero/Raça como proxy importante (Viés).")
    else:
         print(">> Baixo impacto: O modelo não parece depender excessivamente de variáveis protegidas.")

    # --- TABELA FINAL ---
    print("\n" + "="*80)
    print(f"{'METRIC':<20} | {'XGBOOST (Mean ± Std)':<25} | {'TABPFN (Mean ± Std)':<25}")
    print("-" * 80)
    
    final_metrics = ['mcc', 'dir_gender', 'eod_gender', 'dir_race', 'eod_race']
    
    rows = []
    
    for m in final_metrics:
        xgb_arr = np.array(results['XGBoost'][m])
        tab_arr = np.array(results['TabPFN'][m])
        
        xgb_str = f"{np.nanmean(xgb_arr):.4f} ± {np.nanstd(xgb_arr):.4f}"
        tab_str = f"{np.nanmean(tab_arr):.4f} ± {np.nanstd(tab_arr):.4f}"
        
        print(f"{m.upper():<20} | {xgb_str:<25} | {tab_str:<25}")
        
        rows.append({
            'Metric': m.upper(),
            'XGBoost': xgb_str,
            'TabPFN': tab_str,
            'Winner': 'TabPFN' if np.nanmean(tab_arr) > np.nanmean(xgb_arr) else 'XGBoost',
            'P-Value': f"{p_value:.4f}" if m == 'mcc' else '-'
        })

    print("-" * 80)
    
    pd.DataFrame(rows).to_csv('rigorous_benchmark_results.csv', index=False)
    print("Resultados salvos em 'rigorous_benchmark_results.csv'")

if __name__ == '__main__':
    main()
