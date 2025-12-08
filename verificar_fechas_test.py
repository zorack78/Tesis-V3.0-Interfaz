import pandas as pd

# Cargar y verificar fechas del test set
df_test = pd.read_csv('data/processed/data_test.csv')
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

print(f"📅 Rango de fechas en data_test.csv:")
print(f"   Inicio: {df_test['timestamp'].min()}")
print(f"   Fin: {df_test['timestamp'].max()}")
print(f"   Total registros: {len(df_test):,}")

print(f"\n📊 Primeras 5 fechas:")
print(df_test[['timestamp']].head())

print(f"\n📊 Últimas 5 fechas:")
print(df_test[['timestamp']].tail())

print(f"\n📆 Distribución por mes:")
df_test['mes'] = df_test['timestamp'].dt.to_period('M')
print(df_test['mes'].value_counts().sort_index())
