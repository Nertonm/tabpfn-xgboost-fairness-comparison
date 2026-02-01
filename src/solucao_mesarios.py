import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_balanced_model(X, y):
    """
    Treina um modelo RandomForest com class_weight='balanced' para lidar com o desequilíbrio.
    
    Parâmetros:
    X: DataFrame com as features.
    y: Series com o target (0/1).
    """

    # 1. Definição das colunas
    # Variáveis Categóricas Nominais (OneHotEncoder)
    # Nota: NR_ZONA é categórica pois zonas são identificadores, não quantidades.
    cols_nominal = ['NR_ZONA', 'CD_ESTADO_CIVIL', 'CD_GENERO']
    
    # Variáveis Numéricas/Ordinais (StandardScaler)
    cols_numeric = ['CD_FAIXA_ETARIA', 'CD_GRAU_INSTRUCAO']
    
    # Verificação de colunas para evitar erros de chave (caso alguma não esteja no X)
    cols_nominal = [c for c in cols_nominal if c in X.columns]
    cols_numeric = [c for c in cols_numeric if c in X.columns]
    
    print(f"Features Categoricas: {cols_nominal}")
    print(f"Features Numéricas: {cols_numeric}")
    
    # 2. Pipeline de Pré-processamento
    
    # OneHotEncoder: 
    # handle_unknown='ignore' -> Evita erros se aparecer uma categoria nova no teste
    # sparse_output=False -> Retorna array denso (opcional, mas facilita debug e visualização)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # StandardScaler: Padroniza para média 0 e desvio padrão 1 (importante para alguns modelos, 
    # e boa prática geral, embora RF não dependa estritamente disso)
    numeric_transformer = StandardScaler()
    
    # Preprocessor unificado
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cols_nominal),
            ('num', numeric_transformer, cols_numeric)
        ],
        remainder='drop'  # Descarta colunas que não foram especificadas (ou use 'passthrough')
    )
    
    # 3. Modelagem (O CORAÇÃO DA SOLUÇÃO)
    # class_weight='balanced' -> O modelo penaliza mais os erros na classe minoritária (0),
    # forçando-o a aprender padrões da classe "Faltoso" em vez de ignora-la.
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',  # <-- AQUI ESTÁ O SEGREDO PARA RECALL > 0
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # 4. Divisão Treino/Teste Estratificada
    # stratify=y -> Garante que a proporção 94.5% / 5.5% se mantenha idêntica no treino e no teste.
    print("Dividindo dados em Treino e Teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Treinamento
    print("Treinando modelo...")
    model.fit(X_train, y_train)
    
    # 5. Avaliação
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    # Target values: 0 = Ausente, 1 = Compareceu
    print(classification_report(y_test, y_pred, target_names=['Ausente (0)', 'Compareceu (1)']))
    
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualização Gráfica
    plt.figure(figsize=(8, 6))
    group_names = ['Verdadeiro Negativo\n(Acertou Falta)','Falso Positivo\n(Erro: Predisse Comparecimento)','Falso Negativo\n(Erro: Predisse Falta)','Verdadeiro Positivo\n(Acertou Comparecimento)']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    # Labels combinadas não funcionam bem se não tivermos numpy importado aqui dentro para a lista, 
    # então vamos simplificar para garantir robustez
    sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', cbar=False,
                xticklabels=['Pred: Faltou', 'Pred: Compareceu'],
                yticklabels=['Real: Faltou', 'Real: Compareceu'])
    plt.title('Matriz de Confusão - Random Forest Balanceado')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.show()
    
    return model

# Exemplo de chamada (se fosse executar este script diretamente):
if __name__ == "__main__":
    import numpy as np
    # Mock de dados para teste rápido se executado
    df_mock = pd.DataFrame({
        'NR_ZONA': np.random.randint(1, 10, 1000),
        'CD_ESTADO_CIVIL': np.random.randint(1, 5, 1000),
        'CD_GENERO': np.random.randint(2, 4, 1000),
        'CD_FAIXA_ETARIA': np.random.randint(18, 70, 1000),
        'CD_GRAU_INSTRUCAO': np.random.randint(1, 8, 1000),
        'CD_COMPARECIMENTO': np.random.choice([0, 1], 1000, p=[0.055, 0.945])
    })
    
    X = df_mock.drop(columns='CD_COMPARECIMENTO')
    y = df_mock['CD_COMPARECIMENTO']
    
    train_balanced_model(X, y)
