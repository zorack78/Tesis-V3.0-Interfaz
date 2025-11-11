import pandas as pd

# Cargar datos
df = pd.read_csv('data/raw/BD_VolTotal_X_Hr_m3_UTC.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=" * 70)
print("ANÁLISIS DE DUPLICADOS")
print("=" * 70)

print(f"\n📊 Resumen:")
print(f"   Total de filas: {len(df)}")
print(f"   Timestamps únicos: {df['timestamp'].nunique()}")
print(f"   Diferencia (duplicados): {len(df) - df['timestamp'].nunique()}")

# Encontrar timestamps que aparecen más de una vez
duplicated_timestamps = df[df.duplicated(subset=['timestamp'], keep=False)]
print(f"\n   Filas con timestamps duplicados: {len(duplicated_timestamps)}")

if len(duplicated_timestamps) > 0:
    print(f"\n🔍 Ejemplos de timestamps que aparecen múltiples veces:")
    print("=" * 70)
    
    # Agrupar por timestamp y mostrar los primeros 3 casos
    grouped = duplicated_timestamps.groupby('timestamp')
    
    for i, (timestamp, group) in enumerate(grouped):
        if i >= 3:  # Solo mostrar 3 ejemplos
            break
        print(f"\n   Timestamp: {timestamp}")
        print(f"   Aparece {len(group)} veces:")
        for idx, row in group.iterrows():
            print(f"      - Fila {idx}: Volumen = {row[' Volumen_Total_m3']:,.2f} m³")
    
    # Mostrar estadísticas de cuántas veces se repite cada timestamp
    print(f"\n📈 Frecuencia de repeticiones:")
    repeat_counts = duplicated_timestamps.groupby('timestamp').size()
    print(f"   - Promedio de repeticiones: {repeat_counts.mean():.2f}")
    print(f"   - Máximo de repeticiones: {repeat_counts.max()}")
    print(f"   - Mínimo de repeticiones: {repeat_counts.min()}")
    
    print(f"\n   Distribución:")
    value_counts = repeat_counts.value_counts().sort_index()
    for count, freq in value_counts.items():
        print(f"   - {freq} timestamps aparecen {count} veces")

else:
    print("\n✅ No hay timestamps duplicados")

print("\n" + "=" * 70)
