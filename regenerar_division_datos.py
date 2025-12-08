# -*- coding: utf-8 -*-
"""
Regenerar Division Train/Validation/Test
Division: 70% train, 15% validation, 15% test
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("REGENERANDO DIVISION TRAIN/VALIDATION/TEST (70-15-15)")
print("=" * 80)

# Cargar dataset completo con features
print("\n[1] Cargando dataset features completo...")
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   OK - {len(df):,} registros cargados")
print(f"   Periodo: {df['timestamp'].min()} -> {df['timestamp'].max()}")

# Dividir datos (70-15-15)
print("\n[2] Dividiendo datos (70% train, 15% val, 15% test)...")
n = len(df)

# Calcular indices
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Dividir
df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f"\n   Train: {len(df_train):,} registros ({len(df_train)/n*100:.1f}%)")
print(f"   Periodo: {df_train['timestamp'].min()} -> {df_train['timestamp'].max()}")

print(f"\n   Validation: {len(df_val):,} registros ({len(df_val)/n*100:.1f}%)")
print(f"   Periodo: {df_val['timestamp'].min()} -> {df_val['timestamp'].max()}")

print(f"\n   Test: {len(df_test):,} registros ({len(df_test)/n*100:.1f}%)")
print(f"   Periodo: {df_test['timestamp'].min()} -> {df_test['timestamp'].max()}")

# Guardar archivos
print("\n[3] Guardando archivos...")
output_dir = Path('data/processed')

df_train.to_csv(output_dir / 'data_train.csv', index=False)
print(f"   OK - data_train.csv guardado")

df_val.to_csv(output_dir / 'data_validation.csv', index=False)
print(f"   OK - data_validation.csv guardado")

df_test.to_csv(output_dir / 'data_test.csv', index=False)
print(f"   OK - data_test.csv guardado")

# Verificar continuidad temporal
print("\n[4] Verificando continuidad temporal...")
ultimo_train = df_train['timestamp'].max()
primero_val = df_val['timestamp'].min()
ultimo_val = df_val['timestamp'].max()
primero_test = df_test['timestamp'].min()

gap_train_val = (primero_val - ultimo_train).total_seconds() / 3600
gap_val_test = (primero_test - ultimo_val).total_seconds() / 3600

print(f"   Gap train->val: {gap_train_val:.1f} horas")
print(f"   Gap val->test: {gap_val_test:.1f} horas")

if gap_train_val <= 2 and gap_val_test <= 2:
    print("   OK - Continuidad temporal verificada")
else:
    print("   ADVERTENCIA - Hay gaps temporales grandes")

print("\n" + "=" * 80)
print("DIVISION COMPLETADA")
print("=" * 80)
