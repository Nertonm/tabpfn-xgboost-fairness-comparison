import pandas as pd
try:
    df = pd.read_csv('/home/nerton/TRABALHO/metodologia_trabalho/data/votacao_candidato.csv', sep=';', encoding='latin1')
except:
    df = pd.read_csv('/home/nerton/TRABALHO/metodologia_trabalho/data/votacao_candidato.csv', sep=',', encoding='utf-8')

with open('data_vals.txt', 'w') as f:
    f.write(f"Cargo unique values: {df['Cargo'].unique()}\n")
    f.write(f"Situação totalização unique values: {df['Situação totalização'].unique()}\n")
    f.write(f"Gênero unique values: {df['Gênero'].unique()}\n")
    f.write(f"Cor/raça unique values: {df['Cor/raça'].unique()}\n")
