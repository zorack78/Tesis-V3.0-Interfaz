import pandas as pd

df = pd.read_csv('data/raw/BD_Qin_m3_Local.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Registros totales: {len(df)}")
print(f"Desde: {df['timestamp'].min()}")
print(f"Hasta: {df['timestamp'].max()}")

# Ver si incluye test set (Jun-Sep 2025)
test_data = df[df['timestamp'] >= '2025-06-26']
print(f"\nDatos en periodo test (>= Jun 26, 2025): {len(test_data)}")

if len(test_data) > 0:
    print("❌ PROBLEMA: BD_Qin incluye datos del test set!")
    print(f"  Qin promedio test: {test_data['Qin'].mean():,.0f}")
