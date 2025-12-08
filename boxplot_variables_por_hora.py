"""
Boxplots por Hora del Día para Variables Principales del Sistema

Genera visualizaciones de boxplot mostrando la distribución de las variables
principales (Qin, Volumen Total, Q_flujo) agrupadas por hora del día.

Autor: Sistema de Análisis de Demanda de Agua
Fecha: Diciembre 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas de datos
DATA_RAW = Path('data/raw')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("GENERACIÓN DE BOXPLOTS POR HORA DEL DÍA")
print("="*70)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("\n[1/4] Cargando datos...")

# Q_flujo
df_flujo = pd.read_csv(DATA_RAW / 'BD_Q_net_x_Hr_m3h.csv')
df_flujo.columns = df_flujo.columns.str.strip()
df_flujo['Timestamp'] = pd.to_datetime(df_flujo['timestamp'], utc=True)
df_flujo = df_flujo.rename(columns={'Q_flujo_m3hr': 'Q_flujo'})
print(f"   ✓ Q_flujo: {len(df_flujo):,} registros")

# Qin
df_qin = pd.read_csv(DATA_RAW / 'BD_Qin_m3_UTC.csv')
df_qin.columns = df_qin.columns.str.strip()
df_qin['Timestamp'] = pd.to_datetime(df_qin['timestamp'], utc=True)
df_qin = df_qin.rename(columns={'Qin': 'Qin'})
print(f"   ✓ Qin: {len(df_qin):,} registros")

# Volumen Total
df_vol = pd.read_csv(DATA_RAW / 'BD_VolTotal_X_Hr_m3_UTC.csv')
df_vol.columns = df_vol.columns.str.strip()
df_vol['Timestamp'] = pd.to_datetime(df_vol['timestamp'], utc=True)
df_vol = df_vol.rename(columns={'Volumen_Total_m3': 'Vol_Total'})
print(f"   ✓ Volumen Total: {len(df_vol):,} registros")

# ============================================================================
# MERGE Y PREPARACIÓN
# ============================================================================

print("\n[2/4] Preparando datos...")

# Merge de los tres datasets
df = df_qin[['Timestamp', 'Qin']].copy()
df = df.merge(df_vol[['Timestamp', 'Vol_Total']], on='Timestamp', how='inner')
df = df.merge(df_flujo[['Timestamp', 'Q_flujo']], on='Timestamp', how='inner')

print(f"   ✓ Dataset combinado: {len(df):,} registros")

# Extraer hora del día
df['Hora'] = df['Timestamp'].dt.hour

# Calcular Demanda
df['Demanda'] = df['Qin'] - df['Q_flujo']

print(f"   ✓ Período: {df['Timestamp'].min()} a {df['Timestamp'].max()}")
print(f"   ✓ Variables creadas: Hora (0-23), Demanda")

# ============================================================================
# VISUALIZACIÓN 1: BOXPLOTS INDIVIDUALES (4 SUBPLOTS)
# ============================================================================

print("\n[3/4] Generando visualización 1: Boxplots individuales...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')

# Título general
fig.suptitle('Distribución Horaria de Variables del Sistema de Distribución',
             fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')

# --- Subplot 1: Qin ---
ax1 = axes[0, 0]
bp1 = ax1.boxplot([df[df['Hora'] == h]['Qin'].dropna() for h in range(24)],
                   positions=range(24),
                   widths=0.6,
                   patch_artist=True,
                   showfliers=True,
                   notch=True,
                   boxprops=dict(facecolor='#3498DB', alpha=0.7, edgecolor='#2C3E50', linewidth=1.5),
                   whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                   capprops=dict(color='#2C3E50', linewidth=1.5),
                   medianprops=dict(color='#E74C3C', linewidth=2.5),
                   flierprops=dict(marker='o', markerfacecolor='#95A5A6', markersize=4, alpha=0.5))

ax1.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax1.set_ylabel('Producción (m³/h)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Producción de Agua (Qin)', fontsize=13, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(range(0, 24, 2))
ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)

# Estadísticas
qin_mean = df['Qin'].mean()
qin_std = df['Qin'].std()
ax1.axhline(qin_mean, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Media: {qin_mean:,.0f} m³/h')
ax1.legend(loc='upper right', fontsize=9)

# --- Subplot 2: Volumen Total ---
ax2 = axes[0, 1]
bp2 = ax2.boxplot([df[df['Hora'] == h]['Vol_Total'].dropna() for h in range(24)],
                   positions=range(24),
                   widths=0.6,
                   patch_artist=True,
                   showfliers=True,
                   notch=True,
                   boxprops=dict(facecolor='#2ECC71', alpha=0.7, edgecolor='#2C3E50', linewidth=1.5),
                   whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                   capprops=dict(color='#2C3E50', linewidth=1.5),
                   medianprops=dict(color='#E74C3C', linewidth=2.5),
                   flierprops=dict(marker='o', markerfacecolor='#95A5A6', markersize=4, alpha=0.5))

ax2.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax2.set_ylabel('Volumen Almacenado (m³)', fontsize=11, fontweight='bold')
ax2.set_title('(B) Volumen Total en Estanques', fontsize=13, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(range(0, 24, 2))
ax2.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)

vol_mean = df['Vol_Total'].mean()
ax2.axhline(vol_mean, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Media: {vol_mean:,.0f} m³')
ax2.legend(loc='upper right', fontsize=9)

# --- Subplot 3: Q_flujo ---
ax3 = axes[1, 0]
bp3 = ax3.boxplot([df[df['Hora'] == h]['Q_flujo'].dropna() for h in range(24)],
                   positions=range(24),
                   widths=0.6,
                   patch_artist=True,
                   showfliers=True,
                   notch=True,
                   boxprops=dict(facecolor='#F39C12', alpha=0.7, edgecolor='#2C3E50', linewidth=1.5),
                   whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                   capprops=dict(color='#2C3E50', linewidth=1.5),
                   medianprops=dict(color='#E74C3C', linewidth=2.5),
                   flierprops=dict(marker='o', markerfacecolor='#95A5A6', markersize=4, alpha=0.5))

ax3.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax3.set_ylabel('Tasa de Cambio de Volumen (m³/h)', fontsize=11, fontweight='bold')
ax3.set_title('(C) Q_flujo (ΔV/Δt)', fontsize=13, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(range(0, 24, 2))
ax3.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
ax3.axhline(0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Equilibrio (Q_flujo = 0)')
ax3.legend(loc='upper right', fontsize=9)

# --- Subplot 4: Demanda ---
ax4 = axes[1, 1]
bp4 = ax4.boxplot([df[df['Hora'] == h]['Demanda'].dropna() for h in range(24)],
                   positions=range(24),
                   widths=0.6,
                   patch_artist=True,
                   showfliers=True,
                   notch=True,
                   boxprops=dict(facecolor='#9B59B6', alpha=0.7, edgecolor='#2C3E50', linewidth=1.5),
                   whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                   capprops=dict(color='#2C3E50', linewidth=1.5),
                   medianprops=dict(color='#E74C3C', linewidth=2.5),
                   flierprops=dict(marker='o', markerfacecolor='#95A5A6', markersize=4, alpha=0.5))

ax4.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax4.set_ylabel('Demanda Observada (m³/h)', fontsize=11, fontweight='bold')
ax4.set_title('(D) Demanda de Agua (D_obs = Qin - Q_flujo)', fontsize=13, fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xticks(range(0, 24, 2))
ax4.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)

dem_mean = df['Demanda'].mean()
ax4.axhline(dem_mean, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Media: {dem_mean:,.0f} m³/h')
ax4.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_variables_por_hora.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'boxplot_variables_por_hora.pdf', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: boxplot_variables_por_hora.png/pdf")

# ============================================================================
# VISUALIZACIÓN 2: BOXPLOT COMBINADO CON SEABORN
# ============================================================================

print("\n[4/4] Generando visualización 2: Boxplot comparativo...")

# Preparar datos en formato largo
df_long = pd.melt(df, 
                   id_vars=['Hora'], 
                   value_vars=['Qin', 'Vol_Total', 'Q_flujo', 'Demanda'],
                   var_name='Variable', 
                   value_name='Valor')

# Normalizar para visualización conjunta (solo para este gráfico)
df_norm = df_long.copy()
for var in ['Qin', 'Vol_Total', 'Q_flujo', 'Demanda']:
    mask = df_norm['Variable'] == var
    valores = df_norm.loc[mask, 'Valor']
    df_norm.loc[mask, 'Valor_Norm'] = (valores - valores.mean()) / valores.std()

# Crear figura
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')

fig.suptitle('Análisis Comparativo de Variables por Hora del Día',
             fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')

# --- Panel Superior: Valores Absolutos (solo variables de escala similar) ---
df_subset = df_long[df_long['Variable'].isin(['Qin', 'Q_flujo', 'Demanda'])].copy()

sns.boxplot(data=df_subset, x='Hora', y='Valor', hue='Variable', ax=ax1,
            palette={'Qin': '#3498DB', 'Q_flujo': '#F39C12', 'Demanda': '#9B59B6'},
            showfliers=False, notch=True, linewidth=1.5)

ax1.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
ax1.set_ylabel('Valor (m³/h)', fontsize=12, fontweight='bold')
ax1.set_title('(A) Comparación de Flujos: Producción, Cambio de Volumen y Demanda',
              fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
ax1.legend(title='Variable', fontsize=10, title_fontsize=11, loc='upper left')
ax1.axhline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.5)

# --- Panel Inferior: Valores Normalizados (todas las variables) ---
sns.boxplot(data=df_norm, x='Hora', y='Valor_Norm', hue='Variable', ax=ax2,
            palette={'Qin': '#3498DB', 'Vol_Total': '#2ECC71', 
                     'Q_flujo': '#F39C12', 'Demanda': '#9B59B6'},
            showfliers=False, notch=True, linewidth=1.5)

ax2.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
ax2.set_ylabel('Valor Normalizado (σ)', fontsize=12, fontweight='bold')
ax2.set_title('(B) Patrones Normalizados: Todas las Variables',
              fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend(title='Variable', fontsize=10, title_fontsize=11, loc='upper left')
ax2.axhline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_comparativo_por_hora.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'boxplot_comparativo_por_hora.pdf', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: boxplot_comparativo_por_hora.png/pdf")

# ============================================================================
# ESTADÍSTICAS POR HORA
# ============================================================================

print("\n" + "="*70)
print("ESTADÍSTICAS POR HORA DEL DÍA")
print("="*70)

stats_hora = df.groupby('Hora').agg({
    'Qin': ['mean', 'std', 'min', 'max'],
    'Vol_Total': ['mean', 'std', 'min', 'max'],
    'Q_flujo': ['mean', 'std', 'min', 'max'],
    'Demanda': ['mean', 'std', 'min', 'max']
}).round(2)

# Guardar estadísticas
stats_hora.to_csv(OUTPUT_DIR / 'estadisticas_por_hora.csv')
print("\n✓ Estadísticas guardadas en: estadisticas_por_hora.csv")

# Mostrar resumen
print("\nRESUMEN DE PATRONES IDENTIFICADOS:")
print("-" * 70)

# Hora de mayor/menor demanda
hora_max_dem = df.groupby('Hora')['Demanda'].mean().idxmax()
hora_min_dem = df.groupby('Hora')['Demanda'].mean().idxmin()
print(f"• Demanda MÁXIMA promedio: {hora_max_dem:02d}:00 hrs")
print(f"• Demanda MÍNIMA promedio: {hora_min_dem:02d}:00 hrs")

# Hora de mayor/menor volumen
hora_max_vol = df.groupby('Hora')['Vol_Total'].mean().idxmax()
hora_min_vol = df.groupby('Hora')['Vol_Total'].mean().idxmin()
print(f"• Volumen MÁXIMO promedio: {hora_max_vol:02d}:00 hrs")
print(f"• Volumen MÍNIMO promedio: {hora_min_vol:02d}:00 hrs")

# Análisis Q_flujo
qflujo_por_hora = df.groupby('Hora')['Q_flujo'].mean()
horas_llenado = qflujo_por_hora[qflujo_por_hora > 0].index.tolist()
horas_vaciado = qflujo_por_hora[qflujo_por_hora < 0].index.tolist()

print(f"• Horas de LLENADO (Q_flujo > 0): {[f'{h:02d}:00' for h in horas_llenado]}")
print(f"• Horas de VACIADO (Q_flujo < 0): {[f'{h:02d}:00' for h in horas_vaciado]}")

print("\n" + "="*70)
print("PROCESO COMPLETADO")
print("="*70)
print("\nArchivos generados en outputs/:")
print("  • boxplot_variables_por_hora.png/pdf")
print("  • boxplot_comparativo_por_hora.png/pdf")
print("  • estadisticas_por_hora.csv")
print("="*70)
