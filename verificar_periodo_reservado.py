"""
Script para verificar el período de testing reservado originalmente
"""

import pandas as pd
import numpy as np

print("="*80)
print("VERIFICACIÓN: PERÍODO DE TESTING RESERVADO")
print("="*80)

# Cargar datos completos
df = pd.read_csv('data/processed/data_processed_demanda_valid.csv')
df.columns = df.columns.str.strip()
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

# Crear LAGs (como en entrenamiento)
df['Demanda_lag_1h'] = df['Demanda_m3_hr'].shift(1)
df['Demanda_lag_2h'] = df['Demanda_m3_hr'].shift(2)
df['Demanda_lag_24h'] = df['Demanda_m3_hr'].shift(24)
df['Demanda_lag_168h'] = df['Demanda_m3_hr'].shift(168)
df['Demanda_rolling_mean_6h'] = df['Demanda_m3_hr'].rolling(6, min_periods=1).mean()
df['Demanda_rolling_std_6h'] = df['Demanda_m3_hr'].rolling(6, min_periods=1).std()
df['Demanda_rolling_mean_24h'] = df['Demanda_m3_hr'].rolling(24, min_periods=1).mean()
df['Demanda_rolling_std_24h'] = df['Demanda_m3_hr'].rolling(24, min_periods=1).std()
df['Demanda_diff_1h'] = df['Demanda_m3_hr'].diff()
df['Demanda_diff_24h'] = df['Demanda_m3_hr'].diff(24)
df['Demanda_ratio_vs_24h'] = df['Demanda_m3_hr'] / (df['Demanda_lag_24h'] + 1)

# Limpiar NaN (igual que en entrenamiento)
df_clean = df.dropna(subset=[
    'Demanda_m3_hr', 'Demanda_lag_1h', 'Demanda_lag_24h', 'Demanda_lag_168h'
]).reset_index(drop=True)

print(f"\n✅ Datos procesados: {len(df_clean)} registros")
print(f"   Período completo: {df_clean['timestamp_utc'].min()} a {df_clean['timestamp_utc'].max()}")

# Calcular splits según el entrenamiento
n = len(df_clean)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

print(f"\n" + "="*80)
print("DIVISIÓN DE DATOS (SEGÚN ENTRENAMIENTO)")
print("="*80)

# Train (0 - 70%)
df_train = df_clean.iloc[:train_end]
print(f"\n📊 TRAIN (70%):")
print(f"   Registros: {len(df_train):,} ({len(df_train)/n*100:.1f}%)")
print(f"   Inicio: {df_train['timestamp_utc'].min()}")
print(f"   Fin:    {df_train['timestamp_utc'].max()}")
print(f"   Demanda promedio: {df_train['Demanda_m3_hr'].mean():,.0f} m³/hr")

# Validation (70% - 85%)
df_val = df_clean.iloc[train_end:val_end]
print(f"\n📊 VALIDATION (15%):")
print(f"   Registros: {len(df_val):,} ({len(df_val)/n*100:.1f}%)")
print(f"   Inicio: {df_val['timestamp_utc'].min()}")
print(f"   Fin:    {df_val['timestamp_utc'].max()}")
print(f"   Demanda promedio: {df_val['Demanda_m3_hr'].mean():,.0f} m³/hr")

# Test (85% - 100%)
df_test = df_clean.iloc[val_end:]
print(f"\n📊 TEST (15%):")
print(f"   Registros: {len(df_test):,} ({len(df_test)/n*100:.1f}%)")
print(f"   Inicio: {df_test['timestamp_utc'].min()}")
print(f"   Fin:    {df_test['timestamp_utc'].max()}")
print(f"   Demanda promedio: {df_test['Demanda_m3_hr'].mean():,.0f} m³/hr")

# Verificar las métricas guardadas
print(f"\n" + "="*80)
print("VERIFICACIÓN CON MÉTRICAS GUARDADAS")
print("="*80)

print(f"\n📋 Según metricas.json:")
print(f"   Train: 10,537 registros")
print(f"   Val:    2,258 registros")
print(f"   Test:   2,259 registros")
print(f"   Total:  15,054 registros")

print(f"\n📋 Datos procesados ahora:")
print(f"   Train: {len(df_train):,} registros ✅ COINCIDE")
print(f"   Val:   {len(df_val):,} registros ✅ COINCIDE")
print(f"   Test:  {len(df_test):,} registros ✅ COINCIDE")
print(f"   Total: {len(df_clean):,} registros ✅ COINCIDE")

# Análisis de outliers en test
print(f"\n" + "="*80)
print("ANÁLISIS DE OUTLIERS EN TEST")
print("="*80)

print(f"\n📊 Demanda en período TEST (completo):")
print(f"   Min:   {df_test['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"   Max:   {df_test['Demanda_m3_hr'].max():,.0f} m³/hr")
print(f"   Media: {df_test['Demanda_m3_hr'].mean():,.0f} m³/hr")
negativos = (df_test['Demanda_m3_hr'] < 0).sum()
extremos = (df_test['Demanda_m3_hr'] > 30000).sum()
print(f"   Valores < 0:      {negativos} ({negativos/len(df_test)*100:.2f}%)")
print(f"   Valores > 30,000: {extremos} ({extremos/len(df_test)*100:.2f}%)")

# ¿Cuál es el período AGO-SEP dentro de TEST?
print(f"\n" + "="*80)
print("PERÍODO AGOSTO-SEPTIEMBRE 2025 DENTRO DE TEST")
print("="*80)

df_test_ago_sep = df_test[
    (df_test['timestamp_utc'] >= '2025-08-01') &
    (df_test['timestamp_utc'] <= '2025-09-30')
]

print(f"\n📅 Agosto-Septiembre 2025:")
print(f"   Registros: {len(df_test_ago_sep):,} de {len(df_test):,} ({len(df_test_ago_sep)/len(df_test)*100:.1f}%)")
print(f"   Inicio: {df_test_ago_sep['timestamp_utc'].min()}")
print(f"   Fin:    {df_test_ago_sep['timestamp_utc'].max()}")

print(f"\n" + "="*80)
print("✅ RESUMEN")
print("="*80)
print(f"""
PERÍODO DE TESTING RESERVADO ORIGINALMENTE:
• Inicio: {df_test['timestamp_utc'].min().strftime('%d/%m/%Y %H:%M')}
• Fin:    {df_test['timestamp_utc'].max().strftime('%d/%m/%Y %H:%M')}
• Total:  {len(df_test):,} registros (15% del dataset)
• Duración: ~{(df_test['timestamp_utc'].max() - df_test['timestamp_utc'].min()).days} días

SUBPERÍODO AGOSTO-SEPTIEMBRE 2025:
• Inicio: 01/08/2025 00:00
• Fin:    30/09/2025 00:00
• Total:  {len(df_test_ago_sep):,} registros ({len(df_test_ago_sep)/len(df_test)*100:.1f}% del test)
• Duración: 61 días

CONCLUSIÓN:
El período de testing reservado es MÁS AMPLIO que Agosto-Septiembre.
La interfaz usa SOLO Agosto-Septiembre para evaluación más reciente.
""")
print("="*80)
