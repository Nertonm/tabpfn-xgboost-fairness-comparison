import xgboost as xgb
import numpy as np
import pandas as pd

print("XGBoost version:", xgb.__version__)

# Create synthetic data with 2 classes
X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
y = pd.Series(np.random.randint(0, 2, 100))

print("Target distribution:", y.value_counts().to_dict())

try:
    print("Attempting GPU training...")
    clf = xgb.XGBClassifier(
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        base_score=0.5
    )
    clf.fit(X, y)
    print("GPU training successful!")
except Exception as e:
    print(f"GPU training failed: {e}")
