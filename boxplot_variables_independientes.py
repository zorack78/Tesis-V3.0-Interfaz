"""
Boxplots Independientes por Hora del Día - Variables del Sistema

Genera 3 visualizaciones independientes (una por cada CSV) mostrando
la distribución horaria de cada variable principal del sistema.

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
print("GENERACIÓN DE BOXPLOTS INDEPENDIENTES POR HORA DEL DÍA")
print("="*70)

# ============================================================================
# GRÁFICO 1: Q_FLUJO
# ============================================================================

print("\n[1/3] Procesando BD_Q_net_x_Hr_m3h_LIMPIO.csv...")

df_qnet = pd.read_csv(DATA_RAW / 'BD_Q_net_x_Hr_m3h_LIMPIO.csv')
df_qnet.columns = df_qnet.columns.str.strip()
df_qnet['timestamp'] = pd.to_datetime(df_qnet['timestamp'], utc=True)
df_qnet['Hora'] = df_qnet['timestamp'].dt.hour

print(f"   ✓ Registros: {len(df_qnet):,}")
print(f"   ✓ Columnas: {df_qnet.columns.tolist()}")
print(f"   ✓ Período: {df_qnet['timestamp'].min()} a {df_qnet['timestamp'].max()}")

# Crear figura
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#F8F9FA')

# Preparar datos por hora
data_by_hour = [df_qnet[df_qnet['Hora'] == h]['Q_net_m3h'].dropna() for h in range(24)]

# Boxplot
bp = ax.boxplot(data_by_hour,
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

# Configuración
ax.set_xlabel('Hora del Día', fontsize=13, fontweight='bold')
ax.set_ylabel('Q_net (m³/h)', fontsize=13, fontweight='bold')
ax.set_title('Distribución Horaria de Q_net (Tasa de Cambio de Volumen)\n' +
             'BD_Q_net_x_Hr_m3h_LIMPIO.csv',
             fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)

# Línea de equilibrio
ax.axhline(0, color='red', linestyle='-', linewidth=2, alpha=0.8, 
          label='Equilibrio (Q_net = 0)\nPositivo: Llenado | Negativo: Vaciado')

# Estadísticas
media = df_qnet['Q_net_m3h'].mean()
std = df_qnet['Q_net_m3h'].std()
ax.axhline(media, color='#3498DB', linestyle='--', linewidth=1.5, alpha=0.7, 
          label=f'Media: {media:,.1f} m³/h')

# Leyenda
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Info adicional
stats_text = (f'Media: {media:,.1f} m³/h | Desv.Est: {std:,.1f} m³/h\n'
              f'Mín: {df_qnet["Q_net_m3h"].min():,.1f} m³/h | '
              f'Máx: {df_qnet["Q_net_m3h"].max():,.1f} m³/h')
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic', color='#7F8C8D')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_Q_flujo_por_hora.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'boxplot_Q_flujo_por_hora.pdf', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: boxplot_Q_flujo_por_hora.png/pdf")

# ============================================================================
# GRÁFICO 2: QIN (PRODUCCIÓN)
# ============================================================================

print("\n[2/3] Procesando BD_Qin_m3_UTC.csv...")

df_qin = pd.read_csv(DATA_RAW / 'BD_Qin_m3_UTC.csv')
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'], utc=True)
df_qin['Hora'] = df_qin['timestamp'].dt.hour

print(f"   ✓ Registros: {len(df_qin):,}")
print(f"   ✓ Columnas: {df_qin.columns.tolist()}")
print(f"   ✓ Período: {df_qin['timestamp'].min()} a {df_qin['timestamp'].max()}")

# Crear figura
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#F8F9FA')

# Preparar datos por hora
data_by_hour = [df_qin[df_qin['Hora'] == h]['Qin'].dropna() for h in range(24)]

# Boxplot
bp = ax.boxplot(data_by_hour,
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

# Configuración
ax.set_xlabel('Hora del Día', fontsize=13, fontweight='bold')
ax.set_ylabel('Producción (m³/h)', fontsize=13, fontweight='bold')
ax.set_title('Distribución Horaria de Producción de Agua (Qin)\n' +
             'BD_Qin_m3_UTC.csv',
             fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)

# Estadísticas
media = df_qin['Qin'].mean()
std = df_qin['Qin'].std()
mediana = df_qin['Qin'].median()

ax.axhline(media, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, 
          label=f'Media: {media:,.1f} m³/h')
ax.axhline(mediana, color='#2ECC71', linestyle='--', linewidth=1.5, alpha=0.7, 
          label=f'Mediana: {mediana:,.1f} m³/h')

# Leyenda
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Info adicional
stats_text = (f'Media: {media:,.1f} m³/h | Mediana: {mediana:,.1f} m³/h | Desv.Est: {std:,.1f} m³/h\n'
              f'Mín: {df_qin["Qin"].min():,.1f} m³/h | '
              f'Máx: {df_qin["Qin"].max():,.1f} m³/h')
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic', color='#7F8C8D')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_Qin_por_hora.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'boxplot_Qin_por_hora.pdf', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: boxplot_Qin_por_hora.png/pdf")

# ============================================================================
# GRÁFICO 3: VOLUMEN TOTAL
# ============================================================================

print("\n[3/3] Procesando BD_VolTotal_X_Hr_m3_UTC.csv...")

df_vol = pd.read_csv(DATA_RAW / 'BD_VolTotal_X_Hr_m3_UTC.csv')
df_vol.columns = df_vol.columns.str.strip()
df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], utc=True)
df_vol['Hora'] = df_vol['timestamp'].dt.hour

print(f"   ✓ Registros: {len(df_vol):,}")
print(f"   ✓ Columnas: {df_vol.columns.tolist()}")
print(f"   ✓ Período: {df_vol['timestamp'].min()} a {df_vol['timestamp'].max()}")

# Crear figura
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#F8F9FA')

# Preparar datos por hora
data_by_hour = [df_vol[df_vol['Hora'] == h]['Volumen_Total_m3'].dropna() for h in range(24)]

# Boxplot
bp = ax.boxplot(data_by_hour,
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

# Configuración
ax.set_xlabel('Hora del Día', fontsize=13, fontweight='bold')
ax.set_ylabel('Volumen Almacenado (m³)', fontsize=13, fontweight='bold')
ax.set_title('Distribución Horaria de Volumen Total en Estanques\n' +
             'BD_VolTotal_X_Hr_m3_UTC.csv',
             fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)

# Estadísticas
media = df_vol['Volumen_Total_m3'].mean()
std = df_vol['Volumen_Total_m3'].std()
mediana = df_vol['Volumen_Total_m3'].median()

ax.axhline(media, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, 
          label=f'Media: {media:,.0f} m³')
ax.axhline(mediana, color='#3498DB', linestyle='--', linewidth=1.5, alpha=0.7, 
          label=f'Mediana: {mediana:,.0f} m³')

# Leyenda
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Info adicional
stats_text = (f'Media: {media:,.0f} m³ | Mediana: {mediana:,.0f} m³ | Desv.Est: {std:,.0f} m³\n'
              f'Mín: {df_vol["Volumen_Total_m3"].min():,.0f} m³ | '
              f'Máx: {df_vol["Volumen_Total_m3"].max():,.0f} m³')
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic', color='#7F8C8D')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_VolTotal_por_hora.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'boxplot_VolTotal_por_hora.pdf', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: boxplot_VolTotal_por_hora.png/pdf")

# ============================================================================
# RESUMEN DE PATRONES
# ============================================================================

print("\n" + "="*70)
print("ANÁLISIS DE PATRONES HORARIOS")
print("="*70)

# Q_net
qnet_mean = df_qnet.groupby('Hora')['Q_net_m3h'].mean()
hora_max_llenado = qnet_mean.idxmax()
hora_max_vaciado = qnet_mean.idxmin()

print("\n1. Q_NET (Tasa de Cambio de Volumen):")
print(f"   • Hora de MÁXIMO llenado: {hora_max_llenado:02d}:00 ({qnet_mean[hora_max_llenado]:,.1f} m³/h)")
print(f"   • Hora de MÁXIMO vaciado: {hora_max_vaciado:02d}:00 ({qnet_mean[hora_max_vaciado]:,.1f} m³/h)")
horas_llenado = qnet_mean[qnet_mean > 0].index.tolist()
horas_vaciado = qnet_mean[qnet_mean < 0].index.tolist()
print(f"   • Horas típicas de llenado: {[f'{h:02d}:00' for h in horas_llenado]}")
print(f"   • Horas típicas de vaciado: {[f'{h:02d}:00' for h in horas_vaciado]}")

# Qin
qin_mean = df_qin.groupby('Hora')['Qin'].mean()
hora_max_prod = qin_mean.idxmax()
hora_min_prod = qin_mean.idxmin()

print("\n2. QIN (Producción):")
print(f"   • Hora de MÁXIMA producción: {hora_max_prod:02d}:00 ({qin_mean[hora_max_prod]:,.1f} m³/h)")
print(f"   • Hora de MÍNIMA producción: {hora_min_prod:02d}:00 ({qin_mean[hora_min_prod]:,.1f} m³/h)")
print(f"   • Variabilidad: {df_qin['Qin'].std():,.1f} m³/h (coef. var: {df_qin['Qin'].std()/df_qin['Qin'].mean()*100:.1f}%)")

# Volumen
vol_mean = df_vol.groupby('Hora')['Volumen_Total_m3'].mean()
hora_max_vol = vol_mean.idxmax()
hora_min_vol = vol_mean.idxmin()

print("\n3. VOLUMEN TOTAL (Almacenamiento):")
print(f"   • Hora de MÁXIMO volumen: {hora_max_vol:02d}:00 ({vol_mean[hora_max_vol]:,.0f} m³)")
print(f"   • Hora de MÍNIMO volumen: {hora_min_vol:02d}:00 ({vol_mean[hora_min_vol]:,.0f} m³)")
print(f"   • Amplitud diaria promedio: {vol_mean.max() - vol_mean.min():,.0f} m³")

print("\n" + "="*70)
print("PROCESO COMPLETADO")
print("="*70)
print("\nArchivos generados en outputs/:")
print("  1. boxplot_Q_flujo_por_hora.png/pdf")
print("  2. boxplot_Qin_por_hora.png/pdf")
print("  3. boxplot_VolTotal_por_hora.png/pdf")
print("="*70)
