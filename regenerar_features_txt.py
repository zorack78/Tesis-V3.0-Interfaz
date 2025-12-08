"""
Script para regenerar features.txt con las features actuales del dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('='*80)
print('REGENERAR FEATURES.TXT')
print('='*80)

# Cargar dataset
df = pd.read_csv('data/processed/dataset_features_completo.csv', nrows=5)

# Excluir columnas que no son features
exclude = ['timestamp', 'Q_net_m3h', 'sist_Qin_m3h', 'sist_Vtotal_m3']

# Obtener solo columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Features = numéricas - excluidas
features = [col for col in numeric_cols if col not in exclude]

print(f'\nFeatures encontradas: {len(features)}')
print(f'Primeras 10:')
for i, feat in enumerate(features[:10], 1):
    print(f'  {i}. {feat}')

# Guardar a features.txt
output_path = Path('models/forecasting/features.txt')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    for feat in features:
        f.write(f'{feat}\n')

print(f'\n✅ Archivo guardado: {output_path}')
print(f'   Total features: {len(features)}')

# Verificar si hay EMAs
emas = [f for f in features if 'ema' in f.lower()]
if emas:
    print(f'\n⚠️  ADVERTENCIA: Se encontraron {len(emas)} EMAs:')
    for ema in emas:
        print(f'     - {ema}')
    print('\n   Las EMAs NO mejoraron el rendimiento en pruebas previas.')
    print('   Considera eliminarlas del pipeline.')
else:
    print('\n✅ No se encontraron EMAs (correcto para evitar data leakage)')

print('\n' + '='*80)
