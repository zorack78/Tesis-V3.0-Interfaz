import pandas as pd

print("=" * 70)
print("REVISIÓN: calendar_social_ES_COMPLETO_20240101_20250930.csv")
print("=" * 70)

df = pd.read_csv('data/raw/calendar_social_ES_COMPLETO_20240101_20250930.csv')

print(f"\n📊 Estructura:")
print(f"   Filas: {len(df)}")
print(f"   Columnas: {len(df.columns)}")
print(f"   Columnas (primeras 10): {df.columns.tolist()[:10]}")

print(f"\n📋 Primeras 5 filas:")
print(df.head(5))

# Verificar timestamps
if 'timestamp_utc' in df.columns:
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    print(f"\n🕐 Análisis de Timestamps:")
    print(f"   Total registros: {len(df)}")
    print(f"   Timestamps únicos: {df['timestamp_utc'].nunique()}")
    print(f"   Duplicados: {len(df) - df['timestamp_utc'].nunique()}")
    print(f"   Rango: {df['timestamp_utc'].min()} a {df['timestamp_utc'].max()}")

# Verificar valores nulos
print(f"\n❓ Columnas con valores nulos:")
nulls = df.isnull().sum()
for col, null_count in nulls.items():
    if null_count > 0:
        pct = 100*null_count/len(df)
        if pct > 1:  # Solo mostrar si > 1%
            print(f"   {col}: {null_count} ({pct:.1f}%)")
