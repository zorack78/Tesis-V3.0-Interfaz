import pandas as pd

# Ver estructura del archivo de volumen
vol_df = pd.read_csv('data/raw/BD_VolTotal_X_Hr_m3_UTC.csv')
print("Columnas en archivo de volumen:")
print(vol_df.columns.tolist())
print(f"\nShape: {vol_df.shape}")
print(f"\nPrimeras 10 filas:")
print(vol_df.head(10))
print(f"\nTimestamps únicos: {vol_df['timestamp'].nunique()}")
print(f"\nTotal registros: {len(vol_df)}")
