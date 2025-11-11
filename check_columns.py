import pandas as pd

# Cargar y mostrar columnas
train = pd.read_csv('data/processed/data_train.csv')
print("Columnas disponibles:")
print(train.columns.tolist())
print(f"\nTotal: {len(train.columns)} columnas")
print(f"\nPrimeras filas:")
print(train.head(2))
