import pandas as pd

# Read raw lines to verify separator
with open('/home/nerton/TRABALHO/metodologia_trabalho/data/votacao_candidato.csv', 'r', encoding='latin1') as f:
    head_lines = [f.readline() for _ in range(5)]

print("RAW HEAD LINES:")
for l in head_lines:
    print(repr(l))

# Try reading with comma
print("\nTrying comma separator:")
try:
    df_comma = pd.read_csv('/home/nerton/TRABALHO/metodologia_trabalho/data/votacao_candidato.csv', sep=',', encoding='latin1')
    print(f"Shape: {df_comma.shape}")
    print(f"Columns: {df_comma.columns.tolist()}")
    if 'Situação totalização' in df_comma.columns:
         print(f"Unique Status: {df_comma['Situação totalização'].unique()}")
except Exception as e:
    print(f"Error with comma: {e}")

# Try reading with semicolon
print("\nTrying semicolon separator:")
try:
    df_semi = pd.read_csv('/home/nerton/TRABALHO/metodologia_trabalho/data/votacao_candidato.csv', sep=';', encoding='latin1')
    print(f"Shape: {df_semi.shape}")
    print(f"Columns: {df_semi.columns.tolist()}")
    if 'Situação totalização' in df_semi.columns:
         print(f"Unique Status: {df_semi['Situação totalização'].unique()}")
except Exception as e:
    print(f"Error with semicolon: {e}")
