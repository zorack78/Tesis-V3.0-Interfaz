# -*- coding: utf-8 -*-
"""
Analisis 5: Estadisticas Descriptivas de Variables Clave
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("ANALISIS 5: ESTADISTICAS DESCRIPTIVAS")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/descriptivo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos
print("\n[1] Cargando datos...")
df = pd.read_csv('../data/processed/data_train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   OK - Cargado: {len(df):,} registros")
print(f"   Periodo: {df['timestamp'].min()} -> {df['timestamp'].max()}")

# Calcular demanda (Qout = Qin - Q_net)
df['Demanda_m3h'] = df['sist_Qin_m3h'] - df['Q_net_m3h']
print(f"   Demanda calculada: media={df['Demanda_m3h'].mean():.1f} m3/hr")

# 2. Variables de interes
variables = {
    'Demanda_m3h': 'Demanda de Agua (m3/hr)',
    'clima_temp_c': 'Temperatura (C)',
    'clima_HR_pct': 'Humedad Relativa (%)',
    'clima_lluvia_mmhr': 'Precipitacion (mm/hr)',
    'sist_Qin_m3h': 'Produccion - Caudal Entrada (m3/hr)',
    'sist_Vtotal_m3': 'Volumen Estanques (m3)'
}

print("\n[2] Calculando estadisticas descriptivas...")

# Crear figura con 6 subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Estadísticas Descriptivas - Variables Clave del Sistema', 
             fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()

estadisticas_all = []

for idx, (var, titulo) in enumerate(variables.items()):
    ax = axes[idx]
    
    # Datos validos
    data = df[var].dropna()
    
    # Calcular estadisticas
    stats = {
        'Variable': titulo,
        'Media': data.mean(),
        'Mediana': data.median(),
        'Desv. Std': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'P25': data.quantile(0.25),
        'P75': data.quantile(0.75),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'Registros': len(data)
    }
    estadisticas_all.append(stats)
    
    # Histograma
    ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Lineas verticales para media y mediana
    ax.axvline(stats['Media'], color='red', linestyle='--', linewidth=2, 
               label=f"Media: {stats['Media']:.1f}")
    ax.axvline(stats['Mediana'], color='green', linestyle='--', linewidth=2,
               label=f"Mediana: {stats['Mediana']:.1f}")
    
    # Titulos y etiquetas
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel('Valor', fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Texto con estadisticas clave
    textstr = f'Std = {stats["Desv. Std"]:.1f}\n'
    textstr += f'Min = {stats["Min"]:.1f}\n'
    textstr += f'Max = {stats["Max"]:.1f}\n'
    textstr += f'IQR = {stats["IQR"]:.1f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    print(f"   [{idx+1}/6] {titulo}")
    print(f"       Media={stats['Media']:.2f}, Mediana={stats['Mediana']:.2f}, Std={stats['Desv. Std']:.2f}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_estadisticas_descriptivas.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '01_estadisticas_descriptivas.png'}")
plt.close()

# 3. Guardar tabla de estadisticas
df_stats = pd.DataFrame(estadisticas_all)
df_stats.to_csv(OUTPUT_DIR / '01_estadisticas_tabla.csv', index=False, float_format='%.2f')
print(f"   OK - Tabla guardada: {OUTPUT_DIR / '01_estadisticas_tabla.csv'}")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
