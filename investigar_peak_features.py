"""
Investiga qué features permiten a los modelos predecir el peak anómalo
"""
import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Cargar features usadas
with open('models/forecasting/features.txt') as f:
    features = [l.strip() for l in f.readlines()]

# Peak anómalo
peak_date = '2025-09-25 08:00:00'
peak_row = df[df['timestamp'] == peak_date].iloc[0]

# Semana anterior (normal)
week_before = '2025-09-18 08:00:00'
normal_row = df[df['timestamp'] == week_before].iloc[0]

print("="*80)
print("ANÁLISIS DE FEATURES EN PEAK ANÓMALO vs SEMANA ANTERIOR")
print("="*80)
print(f"\nPeak anómalo: {peak_date}")
print(f"Q_net: {peak_row['Q_net_m3h']:,.0f} m³/hr")
print(f"\nSemana anterior: {week_before}")
print(f"Q_net: {normal_row['Q_net_m3h']:,.0f} m³/hr")

print("\n" + "="*80)
print("FEATURES CON DIFERENCIAS SIGNIFICATIVAS (> 5000)")
print("="*80)

diferencias = []
for feat in features:
    if feat not in df.columns:
        continue
    
    val_peak = peak_row[feat]
    val_normal = normal_row[feat]
    
    # Calcular diferencia
    diff = abs(val_peak - val_normal)
    
    if diff > 5000:
        diferencias.append({
            'feature': feat,
            'val_peak': val_peak,
            'val_normal': val_normal,
            'diferencia': diff
        })

# Ordenar por diferencia
diferencias = sorted(diferencias, key=lambda x: x['diferencia'], reverse=True)

for d in diferencias[:10]:
    print(f"\n{d['feature']}:")
    print(f"  Peak:    {d['val_peak']:>15,.1f}")
    print(f"  Normal:  {d['val_normal']:>15,.1f}")
    print(f"  Δ:       {d['diferencia']:>15,.1f}")

# Buscar features sospechosas: que tengan información del futuro
print("\n" + "="*80)
print("BUSCANDO FEATURES SOSPECHOSAS")
print("="*80)

# Verificar si hay alguna feature que contenga información del volumen actual
volumen_features = [f for f in features if 'vol' in f.lower() or 'vtotal' in f.lower()]
print(f"\nFeatures con 'vol' o 'vtotal': {volumen_features}")

# Verificar emas y medias móviles recientes
ema_features = [f for f in features if 'ema' in f.lower() and ('6h' in f or '12h' in f)]
print(f"\nFeatures EMA recientes (6h, 12h): {ema_features}")

if ema_features:
    print("\nValores en peak:")
    for feat in ema_features:
        print(f"  {feat}: {peak_row[feat]:,.0f}")
