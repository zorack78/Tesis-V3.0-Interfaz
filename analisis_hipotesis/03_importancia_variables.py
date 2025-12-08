# -*- coding: utf-8 -*-
"""
Analisis 3: Importancia de Variables
- Feature importance del modelo XGBoost
- Analisis por categorias
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

print("=" * 80)
print("ANALISIS 3: IMPORTANCIA DE VARIABLES")
print("=" * 80)

# Configuracion
plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar modelo
print("\n[1] Cargando modelo XGBoost...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]
print(f"   OK - Modelo cargado con {len(features)} features")

# 2. Extraer importancia de features
print("\n[2] Extrayendo importancia de variables...")
importancia = modelo.feature_importances_
df_importancia = pd.DataFrame({
    'Variable': features,
    'Importancia': importancia
}).sort_values('Importancia', ascending=False)

print("\nTOP 10 VARIABLES MAS IMPORTANTES:")
for i, row in df_importancia.head(10).iterrows():
    print(f"   {row['Variable']:40s} {row['Importancia']:.4f}")

# 3. Grafico 1: Top 20 variables
print("\n[3] Generando grafico de top 20 variables...")

fig, ax = plt.subplots(figsize=(12, 10))

top_20 = df_importancia.head(20)
colors = plt.cm.Set3(range(len(top_20)))

ax.barh(range(len(top_20)), top_20['Importancia'], color=colors)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Variable'], fontsize=10)
ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Variables Mas Importantes', 
             fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Anadir valores
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax.text(row['Importancia'] + 0.001, i, f"{row['Importancia']:.4f}", 
           va='center', fontsize=9)

plt.tight_layout()
output_file = OUTPUT_DIR / '03_top20_variables.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   OK - Guardado: {output_file}")
plt.close()

# 4. Guardar tabla completa
output_csv = OUTPUT_DIR / '03_importancia_completa.csv'
df_importancia.to_csv(output_csv, index=False)
print(f"\nOK - Tabla guardada: {output_csv}")

# 5. Interpretacion
print("\n" + "=" * 80)
print("INTERPRETACION:")
print("=" * 80)

top_10_sum = df_importancia.head(10)['Importancia'].sum()
print(f"\nLas top 10 variables explican {top_10_sum:.1%} de la importancia")

print("\nTop 5 variables individuales:")
for i, row in df_importancia.head(5).iterrows():
    print(f"   {i+1}. {row['Variable']} = {row['Importancia']:.4f}")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
