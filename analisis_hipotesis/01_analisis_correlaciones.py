# -*- coding: utf-8 -*-
"""
Analisis 1: Mapa de Calor de Correlaciones
- Relaciones entre variables clave
- Matriz de correlacion
- Interpretacion automatica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("ANALISIS 1: MAPA DE CALOR DE CORRELACIONES")
print("=" * 80)

# Configuracion
plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos
print("\n[1] Cargando datos...")
df = pd.read_csv('../data/processed/dataset_completo_sincronizado.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   OK - Cargado: {len(df):,} registros")
print(f"   Periodo: {df['timestamp'].min()} -> {df['timestamp'].max()}")

# 2. Seleccionar variables de interes
print("\n[2] Seleccionando variables de interes...")
# Limpiar nombres de columnas (quitar espacios)
df.columns = df.columns.str.strip()
variables = ['Qin_m3_hr', 'Q_net_m3h', 'Volumen_Total_m3', 'temp', 'HR', 'mmhr']
df_corr = df[variables].copy()
print(f"   Variables seleccionadas: {', '.join(variables)}")

# 3. Calcular correlaciones
print("\n[3] Calculando correlaciones...")
df_corr_clean = df_corr.dropna()
print(f"   Registros validos: {len(df_corr_clean):,} de {len(df_corr):,}")

correlaciones = df_corr_clean.corr()
print("\n[Matriz de correlacion calculada]")

# 4. Generar mapa de calor
print("\n[4] Generando mapa de calor...")

fig, ax = plt.subplots(figsize=(12, 10))

# Mascara triangular
mask = np.triu(np.ones_like(correlaciones, dtype=bool))

# Mapa de calor
sns.heatmap(correlaciones, 
            mask=mask,
            annot=True, 
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1,
            ax=ax)

ax.set_title('Mapa de Calor de Correlaciones - Variables Clave', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
output_file = OUTPUT_DIR / '01_mapa_calor_correlaciones.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   OK - Guardado: {output_file}")
plt.close()

# 5. Identificar correlaciones mas fuertes
print("\n[5] Identificando correlaciones mas fuertes...")

# Extraer correlaciones sin diagonal
corr_abs = correlaciones.abs()
np.fill_diagonal(corr_abs.values, 0)

# Top correlaciones
top_corr = []
for i in range(len(correlaciones)):
    for j in range(i+1, len(correlaciones)):
        top_corr.append({
            'var1': correlaciones.index[i],
            'var2': correlaciones.columns[j],
            'correlacion': correlaciones.iloc[i, j]
        })

df_top_corr = pd.DataFrame(top_corr)
df_top_corr = df_top_corr.reindex(df_top_corr['correlacion'].abs().sort_values(ascending=False).index)
df_top_corr = df_top_corr.head(10)

print(f"\nTOP 10 CORRELACIONES MAS FUERTES:")
for i, row in df_top_corr.iterrows():
    corr_val = row['correlacion']
    if abs(corr_val) >= 0.7:
        interpretacion = "Fuerte"
    elif abs(corr_val) >= 0.4:
        interpretacion = "Moderada"
    else:
        interpretacion = "Debil"
    
    if corr_val > 0:
        interpretacion += " positiva"
    else:
        interpretacion += " negativa"
    
    print(f"   {list(df_top_corr.index).index(i)+1}. {row['var1']} <-> {row['var2']}: {row['correlacion']:.4f} ({interpretacion})")

# 6. Guardar tabla
output_csv = OUTPUT_DIR / '01_correlaciones_completas.csv'
correlaciones.to_csv(output_csv)
print(f"\nOK - Correlaciones guardadas: {output_csv}")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
