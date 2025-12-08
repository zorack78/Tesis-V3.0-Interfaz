"""
Script para comparar features usadas por scatter_plots vs interfaz
"""

print('='*80)
print('COMPARACION DE FEATURES')
print('='*80)

# Features de scatter_plots (features.txt viejo - sin EMAs)
print('\n[1] Features de scatter_plots_test.py:')
with open('models/forecasting/features.txt', 'r') as f:
    features_scatter = f.read().strip().split('\n')
print(f'    Total: {len(features_scatter)} features')
print(f'    Primeras 5: {features_scatter[:5]}')

# Features de interfaz (dataset_features_completo.csv - con EMAs)
import pandas as pd
df = pd.read_csv('data/processed/dataset_features_completo.csv', nrows=5)
exclude = ['timestamp', 'Q_net_m3h', 'sist_Qin_m3h', 'sist_Vtotal_m3']
import numpy as np
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features_interfaz = [col for col in numeric_cols if col not in exclude]

print(f'\n[2] Features de interfaz (dataset_features_completo.csv):')
print(f'    Total: {len(features_interfaz)} features')
print(f'    Primeras 5: {features_interfaz[:5]}')

# Diferencias
print(f'\n[3] Diferencias:')
print(f'    scatter_plots: {len(features_scatter)} features')
print(f'    interfaz:      {len(features_interfaz)} features')
print(f'    Diferencia:    {len(features_interfaz) - len(features_scatter)} features')

# Identificar features adicionales
set_scatter = set(features_scatter)
set_interfaz = set(features_interfaz)

features_extra = set_interfaz - set_scatter
if features_extra:
    print(f'\n[4] Features en interfaz pero NO en scatter_plots ({len(features_extra)}):')
    for feat in sorted(features_extra)[:20]:  # Primeras 20
        print(f'    - {feat}')

features_faltantes = set_scatter - set_interfaz
if features_faltantes:
    print(f'\n[5] Features en scatter_plots pero NO en interfaz ({len(features_faltantes)}):')
    for feat in sorted(features_faltantes)[:20]:
        print(f'    - {feat}')

print('\n' + '='*80)
print('CONCLUSION:')
print('='*80)
print('scatter_plots usa features.txt (164 features sin EMAs)')
print('interfaz usa dataset_features_completo.csv (172 features con EMAs)')
print('Por eso las metricas son DIFERENTES!')
print('='*80)
