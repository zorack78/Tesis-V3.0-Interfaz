import pandas as pd

print("=" * 70)
print("REVISIÓN: BD_Clima2024a202509_UTC.csv")
print("=" * 70)

df = pd.read_csv('data/raw/BD_Clima2024a202509_UTC.csv')

print(f"\n📊 Estructura:")
print(f"   Filas: {len(df)}")
print(f"   Columnas: {len(df.columns)}")
print(f"   Columnas: {df.columns.tolist()}")

print(f"\n📋 Primeras 10 filas:")
print(df.head(10))

print(f"\n📋 Últimas 10 filas:")
print(df.tail(10))

# Verificar timestamps
if 'timestamp' in df.columns or 'timestamp_utc' in df.columns:
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_utc'
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    print(f"\n🕐 Análisis de Timestamps:")
    print(f"   Total registros: {len(df)}")
    print(f"   Timestamps únicos: {df[ts_col].nunique()}")
    print(f"   Duplicados: {len(df) - df[ts_col].nunique()}")
    print(f"   Rango: {df[ts_col].min()} a {df[ts_col].max()}")
    
    # Verificar frecuencia
    diff = df[ts_col].sort_values().diff()
    print(f"\n   Frecuencia más común: {diff.mode().values[0] if len(diff.mode()) > 0 else 'N/A'}")

# Verificar valores nulos
print(f"\n❓ Valores nulos por columna:")
nulls = df.isnull().sum()
for col, null_count in nulls.items():
    if null_count > 0:
        print(f"   {col}: {null_count} ({100*null_count/len(df):.1f}%)")
