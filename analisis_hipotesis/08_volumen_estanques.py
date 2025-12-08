# -*- coding: utf-8 -*-
"""
Analisis 8: Volumen en Estanques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("ANALISIS 8: VOLUMEN EN ESTANQUES")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/descriptivo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos
print("\n[1] Cargando datos...")
df = pd.read_csv('../data/processed/data_train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['fecha'] = df['timestamp'].dt.date
df['hora'] = df['timestamp'].dt.hour
print(f"   OK - Cargado: {len(df):,} registros")

# Crear figura con 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Análisis de Volumen en Estanques del Sistema',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# Subplot 1: Serie temporal - Evolucion del volumen
# ============================================================================
print("\n[2] Analizando evolucion temporal del volumen...")
ax1 = axes[0]

# Volumen por hora
ax1.plot(df['timestamp'], df['sist_Vtotal_m3'], 
         color='steelblue', linewidth=0.5, alpha=0.7, label='Volumen Horario')

# Media movil 24 horas
df['vol_ma_24h'] = df['sist_Vtotal_m3'].rolling(window=24, center=True).mean()
ax1.plot(df['timestamp'], df['vol_ma_24h'], 
         color='darkblue', linewidth=2, label='Media Móvil 24h')

# Estadisticas
vol_mean = df['sist_Vtotal_m3'].mean()
vol_std = df['sist_Vtotal_m3'].std()

ax1.axhline(vol_mean, color='red', linestyle='--', linewidth=2, 
            label=f'Media: {vol_mean:,.0f} m³')
ax1.axhline(vol_mean + vol_std, color='orange', linestyle=':', linewidth=1.5, 
            alpha=0.7, label=f'+1std: {vol_mean + vol_std:,.0f} m3')
ax1.axhline(vol_mean - vol_std, color='orange', linestyle=':', linewidth=1.5, 
            alpha=0.7, label=f'-1std: {vol_mean - vol_std:,.0f} m3')

ax1.set_xlabel('Fecha', fontsize=11, fontweight='bold')
ax1.set_ylabel('Volumen Total (m³)', fontsize=11, fontweight='bold')
ax1.set_title('Evolución Temporal del Volumen Total en Estanques (2024-2025)',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='upper right')
ax1.tick_params(axis='x', rotation=45)

# Formato numeros en eje Y con separadores de miles
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

print(f"   Media: {vol_mean:,.0f} m3")
print(f"   Desv. Std: {vol_std:,.0f} m3")
print(f"   Min: {df['sist_Vtotal_m3'].min():,.0f} m3")
print(f"   Max: {df['sist_Vtotal_m3'].max():,.0f} m3")

# Calcular % de llenado (asumiendo capacidad total)
# Leer capacidad desde archivo si existe, sino usar maximo observado
try:
    with open('../data/processed/sistema_info_v3.json', 'r') as f:
        import json
        info = json.load(f)
        capacidad_total = info.get('capacidad_total_m3', df['sist_Vtotal_m3'].max())
except:
    capacidad_total = df['sist_Vtotal_m3'].max()

pct_llenado = (vol_mean / capacidad_total) * 100
print(f"   Capacidad total: {capacidad_total:,.0f} m³")
print(f"   % Llenado promedio: {pct_llenado:.1f}%")

# ============================================================================
# Subplot 2: Distribucion de volumen por hora del dia
# ============================================================================
print("\n[3] Analizando patron horario del volumen...")
ax2 = axes[1]

# Calcular estadisticas por hora
volumen_hora = df.groupby('hora')['sist_Vtotal_m3'].agg(['mean', 'std', 'min', 'max']).reset_index()

# Grafico de area con rango
ax2.fill_between(volumen_hora['hora'], 
                  volumen_hora['min'], 
                  volumen_hora['max'],
                  alpha=0.2, color='steelblue', label='Rango (Min-Max)')

ax2.plot(volumen_hora['hora'], volumen_hora['mean'], 
         marker='o', linewidth=2, markersize=6, color='darkblue', label='Media')

# Banda de confianza (+/- 1 std)
ax2.fill_between(volumen_hora['hora'],
                  volumen_hora['mean'] - volumen_hora['std'],
                  volumen_hora['mean'] + volumen_hora['std'],
                  alpha=0.4, color='lightblue', label='+/- 1std')

ax2.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax2.set_ylabel('Volumen Total (m³)', fontsize=11, fontweight='bold')
ax2.set_title('Patrón Horario del Volumen en Estanques',
              fontsize=12, fontweight='bold')
ax2.set_xticks(range(0, 24, 2))
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Formato numeros en eje Y
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Identificar hora con mayor y menor volumen
hora_max_vol = volumen_hora.loc[volumen_hora['mean'].idxmax(), 'hora']
hora_min_vol = volumen_hora.loc[volumen_hora['mean'].idxmin(), 'hora']
vol_max = volumen_hora['mean'].max()
vol_min = volumen_hora['mean'].min()

ax2.axvline(hora_max_vol, color='green', linestyle='--', alpha=0.5)
ax2.axvline(hora_min_vol, color='red', linestyle='--', alpha=0.5)

# Texto explicativo
textstr = f'Volumen máximo: {int(hora_max_vol)}:00 hrs ({vol_max:,.0f} m³)\n'
textstr += f'Volumen mínimo: {int(hora_min_vol)}:00 hrs ({vol_min:,.0f} m³)\n'
textstr += f'Variación: {vol_max - vol_min:,.0f} m³ ({((vol_max - vol_min) / vol_min * 100):.1f}%)'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

print(f"   Hora con max volumen: {int(hora_max_vol)}:00 hrs ({vol_max:,.0f} m³)")
print(f"   Hora con min volumen: {int(hora_min_vol)}:00 hrs ({vol_min:,.0f} m³)")
print(f"   Variacion horaria: {vol_max - vol_min:,.0f} m³")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_volumen_estanques.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '04_volumen_estanques.png'}")
plt.close()

# 4. Guardar datos
print("\n[4] Guardando datos...")
volumen_hora.to_csv(OUTPUT_DIR / '04_volumen_por_hora.csv', index=False, float_format='%.2f')
print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
