import pandas as pd

# Carrega o CSV que foi baixado
df = pd.read_csv("data/raw/portugal_housing.csv", low_memory=False)

print("🔍 Colunas disponíveis no dataset:")
print(df.columns.tolist())

print("\n📊 Primeiras linhas:")
print(df.head(3))

print("\nℹ️ Informações gerais (tipos e valores nulos):")
print(df.info())

print("\n❌ Valores nulos por coluna:")
print(df.isnull().sum())

print("\n✅ Lista de tipos de dados por coluna:")
for col in df.columns:
    print(f'{col}: {df[col].dtype}')