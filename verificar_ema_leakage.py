"""
INVESTIGACIÓN CRÍTICA: ¿Los EMAs se calcularon con TODO el dataset?
Si los EMAs incluyen datos del test set, hay leakage.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

n = len(df)
train_end = int(n * 0.70)
test_start = int(n * 0.85)

print("="*80)
print("VERIFICACIÓN CRÍTICA: CÁLCULO DE EMAs")
print("="*80)

print(f"\nTotal registros: {n}")
print(f"Train: 0 - {train_end} ({train_end} registros)")
print(f"Val: {train_end+1} - {test_start} ({test_start - train_end} registros)")
print(f"Test: {test_start+1} - {n-1} ({n - test_start} registros)")

print(f"\nTrain end date: {df.iloc[train_end]['timestamp']}")
print(f"Test start date: {df.iloc[test_start]['timestamp']}")

# Verificar el peak
peak_idx = df[df['timestamp'] == '2025-09-25 08:00:00'].index[0]
print(f"\nPeak index: {peak_idx}")
print(f"Peak en test set: {peak_idx >= test_start} (debe ser True)")

# VERIFICACIÓN CRÍTICA: ¿Cómo se calculó el EMA?
# Si el EMA en el peak usa datos del mismo periodo de test, hay problema
print("\n" + "="*80)
print("ANÁLISIS DE EMA EN EL PEAK")
print("="*80)

# Obtener valores del peak y previos
peak_row = df.iloc[peak_idx]
print(f"\nPeak (Sep 25 08:00):")
print(f"  Q_net_m3h: {peak_row['Q_net_m3h']:,.0f}")
print(f"  Q_net_m3h__ema_win_6h: {peak_row['Q_net_m3h__ema_win_6h']:,.0f}")

# Ver las 6 horas previas (de donde viene el EMA)
print(f"\n6 horas previas (ventana del EMA 6h):")
for i in range(6, 0, -1):
    prev_row = df.iloc[peak_idx - i]
    in_test = (peak_idx - i) >= test_start
    print(f"  {prev_row['timestamp']}: Q_net = {prev_row['Q_net_m3h']:>8,.0f} m³/hr  [{'TEST' if in_test else 'VAL/TRAIN'}]")

# CONCLUSIÓN
print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)

# Calcular cuántas de las 6 horas previas están en test
horas_en_test = sum(1 for i in range(1, 7) if (peak_idx - i) >= test_start)
print(f"\nDe las 6 horas previas al peak:")
print(f"  - {horas_en_test} están en el TEST SET")
print(f"  - {6 - horas_en_test} están en VAL/TRAIN")

if horas_en_test > 0:
    print("\n⚠️ PROBLEMA: El EMA usa datos del mismo test set!")
    print("   Esto NO es leakage técnico, pero explica la alta precisión.")
    print("   Los modelos 'ven' las horas inmediatamente anteriores del test.")
else:
    print("\n✅ OK: El EMA solo usa datos históricos (train/val)")

# Verificar desde cuándo empiezan los datos anormales
print("\n" + "="*80)
print("¿CUÁNDO COMIENZA LA ANOMALÍA?")
print("="*80)

# Buscar primera hora con Q_net muy negativo (< -5000)
for i in range(peak_idx, test_start, -1):
    row = df.iloc[i]
    if row['Q_net_m3h'] < -5000:
        print(f"\nPrimera anomalía severa (Q_net < -5000):")
        print(f"  Fecha: {row['timestamp']}")
        print(f"  Q_net: {row['Q_net_m3h']:,.0f} m³/hr")
        print(f"  Índice: {i} ({'TEST' if i >= test_start else 'VAL'})")
        break
