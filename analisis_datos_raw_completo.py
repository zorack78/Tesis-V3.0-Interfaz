"""
Análisis Completo de Datos RAW - Sistema de Agua Potable
=========================================================

Genera todas las visualizaciones principales del análisis descriptivo:
1. Series temporales (Qin, Volumen Total, Demanda observada)
2. Mapas de presencia/frecuencia por hora
3. Boxplots por hora
4. Distribuciones KDE
5. Composición de capacidad por estanque
6. Porcentaje de llenado por estanque (series temporales)

Autor: Sistema de Predicción de Demanda
Fecha: Diciembre 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Directorios
BASE_DIR = 'c:/Users/socce/Downloads/rafa/Tesis3.0-Interfaz'
DATA_DIR = f'{BASE_DIR}/data/raw'
OUTPUT_DIR = f'{BASE_DIR}/outputs/analisis_datos_raw'

print("="*80)
print("ANÁLISIS COMPLETO DE DATOS RAW - SISTEMA DE AGUA POTABLE")
print("="*80)
print()

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("📂 Cargando datos RAW...")

# 1. Qin (producción)
df_qin = pd.read_csv(f'{DATA_DIR}/BD_Qin_m3_Local.csv')
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
df_qin = df_qin.sort_values('timestamp')
print(f"   ✓ Qin: {len(df_qin):,} registros ({df_qin['timestamp'].min()} - {df_qin['timestamp'].max()})")

# 2. Volumen Total
df_vol = pd.read_csv(f'{DATA_DIR}/BD_VolTotal_X_Hr_m3_Local.csv')
df_vol.columns = df_vol.columns.str.strip()
df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'])
df_vol = df_vol.sort_values('timestamp')
print(f"   ✓ Volumen Total: {len(df_vol):,} registros ({df_vol['timestamp'].min()} - {df_vol['timestamp'].max()})")

# 3. Q_net (flujo neto del sistema = ΔVol/Δt)
df_qnet = pd.read_csv(f'{DATA_DIR}/BD_Q_net_x_Hr_m3h_LIMPIO.csv')
df_qnet.columns = df_qnet.columns.str.strip()
df_qnet['timestamp'] = pd.to_datetime(df_qnet['timestamp'])
df_qnet = df_qnet.sort_values('timestamp')
print(f"   ✓ Q_net: {len(df_qnet):,} registros ({df_qnet['timestamp'].min()} - {df_qnet['timestamp'].max()})")

# 4. Volumen por estanque
df_tanks = pd.read_csv(f'{DATA_DIR}/Vol_X_TK_Hr_m3_Local.csv')
df_tanks.columns = df_tanks.columns.str.strip()
df_tanks['timestamp'] = pd.to_datetime(df_tanks['timestamp'])
df_tanks = df_tanks.sort_values('timestamp')
print(f"   ✓ Volumen por estanque: {len(df_tanks):,} registros")

# 5. Capacidad de estanques
df_capacity = pd.read_csv(f'{DATA_DIR}/BD_Capacidad_89Tks_m3.csv')
df_capacity.columns = df_capacity.columns.str.strip()
print(f"   ✓ Capacidad estanques: {len(df_capacity)} estanques")

# Calcular Demanda observada (D_obs = Qin - Q_net)
df_merged = pd.merge(df_qin, df_qnet, on='timestamp', how='inner', suffixes=('', '_qnet'))
if 'Qin' in df_merged.columns and 'Q_net' in df_merged.columns:
    df_merged['D_obs'] = df_merged['Qin'] - df_merged['Q_net']
elif 'Qin_m3h' in df_merged.columns and 'Q_net_m3h' in df_merged.columns:
    df_merged['D_obs'] = df_merged['Qin_m3h'] - df_merged['Q_net_m3h']
else:
    # Intentar con cualquier columna numérica disponible
    qin_col = [c for c in df_merged.columns if 'qin' in c.lower()][0]
    qnet_col = [c for c in df_merged.columns if 'q_net' in c.lower() or 'qnet' in c.lower()][0]
    df_merged['D_obs'] = df_merged[qin_col] - df_merged[qnet_col]

df_merged['hora'] = df_merged['timestamp'].dt.hour
df_merged['dia_semana'] = df_merged['timestamp'].dt.dayofweek

print("\n   ✓ Demanda calculada (D_obs = Qin - Q_net)")
print(f"     Rango: {df_merged['D_obs'].min():.0f} - {df_merged['D_obs'].max():.0f} m³/h")
print(f"     Media: {df_merged['D_obs'].mean():.0f} m³/h")

print("\n✅ Datos cargados correctamente\n")

# ============================================================================
# 1. SERIES TEMPORALES PRINCIPALES
# ============================================================================

print("📊 Generando series temporales principales...")

# Detectar columnas de datos
qin_col = [c for c in df_qin.columns if c != 'timestamp'][0]
vol_col = [c for c in df_vol.columns if c != 'timestamp'][0]

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Serie 1: Qin (Producción)
axes[0].plot(df_qin['timestamp'], df_qin[qin_col], 
             color='steelblue', linewidth=0.8, alpha=0.7)
axes[0].set_title('Serie Temporal: Producción (Qin)', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Qin (m³/h)', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(df_qin[qin_col].mean(), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Media: {df_qin[qin_col].mean():.0f} m³/h')
axes[0].legend(loc='upper right')

# Serie 2: Volumen Total
axes[1].plot(df_vol['timestamp'], df_vol[vol_col], 
             color='green', linewidth=0.8, alpha=0.7)
axes[1].set_title('Serie Temporal: Volumen Total en Estanques', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Volumen (m³)', fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(df_vol[vol_col].mean(), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Media: {df_vol[vol_col].mean():.0f} m³')
axes[1].legend(loc='upper right')

# Serie 3: Demanda observada
axes[2].plot(df_merged['timestamp'], df_merged['D_obs'], 
             color='orange', linewidth=0.8, alpha=0.7)
axes[2].set_title('Serie Temporal: Demanda Observada (D_obs = Qin - Q_flujo)', 
                  fontweight='bold', fontsize=14)
axes[2].set_ylabel('Demanda (m³/h)', fontweight='bold')
axes[2].set_xlabel('Fecha', fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(df_merged['D_obs'].mean(), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Media: {df_merged["D_obs"].mean():.0f} m³/h')
axes[2].legend(loc='upper right')

# Añadir información de período
fecha_inicio = df_merged['timestamp'].min().strftime('%Y-%m-%d')
fecha_fin = df_merged['timestamp'].max().strftime('%Y-%m-%d')
fig.text(0.5, 0.02, f'Período de análisis: {fecha_inicio} a {fecha_fin}', 
         ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(f'{OUTPUT_DIR}/series_temporales/series_temporales_principales.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/series_temporales/series_temporales_principales.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: series_temporales_principales.png/.pdf")

# ============================================================================
# 2. MAPAS DE PRESENCIA Y FRECUENCIA POR HORA
# ============================================================================

print("\n📊 Generando mapas de presencia y frecuencia por hora...")

# Mapa para Qin
df_qin_hora = df_qin.copy()
df_qin_hora['fecha'] = df_qin_hora['timestamp'].dt.date
df_qin_hora['hora'] = df_qin_hora['timestamp'].dt.hour

# Matriz de presencia (1 si hay datos, 0 si no)
presencia_qin = df_qin_hora.groupby(['fecha', 'hora']).size().unstack(fill_value=0)
presencia_qin = (presencia_qin > 0).astype(int)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Heatmap Qin
im1 = axes[0].imshow(presencia_qin.values, aspect='auto', cmap='YlGnBu', interpolation='nearest')
axes[0].set_title('Mapa de Presencia de Datos: Producción (Qin)', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Hora del día', fontweight='bold')
axes[0].set_ylabel('Fecha', fontweight='bold')
axes[0].set_xticks(range(24))
axes[0].set_xticklabels(range(24))
# Mostrar solo algunas fechas en el eje Y
y_ticks = np.linspace(0, len(presencia_qin)-1, min(20, len(presencia_qin))).astype(int)
axes[0].set_yticks(y_ticks)
axes[0].set_yticklabels([str(presencia_qin.index[i]) for i in y_ticks], fontsize=8)
plt.colorbar(im1, ax=axes[0], label='Presencia (1=datos, 0=sin datos)')

# Estadísticas de cobertura
cobertura_por_hora = presencia_qin.sum(axis=0) / len(presencia_qin) * 100
axes[0].text(0.02, 0.98, f'Cobertura promedio: {cobertura_por_hora.mean():.1f}%', 
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Mapa para Volumen Total
df_vol_hora = df_vol.copy()
df_vol_hora['fecha'] = df_vol_hora['timestamp'].dt.date
df_vol_hora['hora'] = df_vol_hora['timestamp'].dt.hour

presencia_vol = df_vol_hora.groupby(['fecha', 'hora']).size().unstack(fill_value=0)
presencia_vol = (presencia_vol > 0).astype(int)

im2 = axes[1].imshow(presencia_vol.values, aspect='auto', cmap='YlGnBu', interpolation='nearest')
axes[1].set_title('Mapa de Presencia de Datos: Volumen Total', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Hora del día', fontweight='bold')
axes[1].set_ylabel('Fecha', fontweight='bold')
axes[1].set_xticks(range(24))
axes[1].set_xticklabels(range(24))
y_ticks = np.linspace(0, len(presencia_vol)-1, min(20, len(presencia_vol))).astype(int)
axes[1].set_yticks(y_ticks)
axes[1].set_yticklabels([str(presencia_vol.index[i]) for i in y_ticks], fontsize=8)
plt.colorbar(im2, ax=axes[1], label='Presencia (1=datos, 0=sin datos)')

cobertura_vol = presencia_vol.sum(axis=0) / len(presencia_vol) * 100
axes[1].text(0.02, 0.98, f'Cobertura promedio: {cobertura_vol.mean():.1f}%', 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mapas_frecuencia/mapas_presencia_por_hora.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/mapas_frecuencia/mapas_presencia_por_hora.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: mapas_presencia_por_hora.png/.pdf")

# Gráfico de barras de frecuencia por hora
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Frecuencia Qin
freq_qin = df_qin_hora.groupby('hora').size()
axes[0].bar(freq_qin.index, freq_qin.values, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title('Frecuencia de Registros por Hora: Qin', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Hora del día', fontweight='bold')
axes[0].set_ylabel('Número de registros', fontweight='bold')
axes[0].set_xticks(range(24))
axes[0].grid(True, alpha=0.3, axis='y')

# Frecuencia Volumen
freq_vol = df_vol_hora.groupby('hora').size()
axes[1].bar(freq_vol.index, freq_vol.values, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title('Frecuencia de Registros por Hora: Volumen Total', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Hora del día', fontweight='bold')
axes[1].set_ylabel('Número de registros', fontweight='bold')
axes[1].set_xticks(range(24))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mapas_frecuencia/frecuencia_por_hora.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/mapas_frecuencia/frecuencia_por_hora.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: frecuencia_por_hora.png/.pdf")

# ============================================================================
# 3. BOXPLOTS POR HORA
# ============================================================================

print("\n📊 Generando boxplots por hora...")

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Boxplot Qin
df_qin_hora_val = df_qin.copy()
df_qin_hora_val['hora'] = df_qin_hora_val['timestamp'].dt.hour

datos_qin = [df_qin_hora_val[df_qin_hora_val['hora']==h][qin_col].values for h in range(24)]
bp1 = axes[0].boxplot(datos_qin, positions=range(24), widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='blue', linewidth=1.5),
                       capprops=dict(color='blue', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))

axes[0].set_title('Distribución Horaria: Producción (Qin)', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Hora del día', fontweight='bold')
axes[0].set_ylabel('Qin (m³/h)', fontweight='bold')
axes[0].set_xticks(range(24))
axes[0].grid(True, alpha=0.3, axis='y')

# Línea de mediana conectando las horas
medianas_qin = [np.median(d) if len(d) > 0 else np.nan for d in datos_qin]
axes[0].plot(range(24), medianas_qin, 'k--', linewidth=2, alpha=0.7, label='Mediana')
axes[0].legend(loc='upper right')

# Boxplot D_obs (Demanda observada)
datos_dobs = [df_merged[df_merged['hora']==h]['D_obs'].values for h in range(24)]
bp2 = axes[1].boxplot(datos_dobs, positions=range(24), widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightyellow', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='orange', linewidth=1.5),
                       capprops=dict(color='orange', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))

axes[1].set_title('Distribución Horaria: Demanda Observada (D_obs)', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Hora del día', fontweight='bold')
axes[1].set_ylabel('D_obs (m³/h)', fontweight='bold')
axes[1].set_xticks(range(24))
axes[1].grid(True, alpha=0.3, axis='y')

medianas_dobs = [np.median(d) if len(d) > 0 else np.nan for d in datos_dobs]
axes[1].plot(range(24), medianas_dobs, 'k--', linewidth=2, alpha=0.7, label='Mediana')
axes[1].legend(loc='upper right')

# Resaltar horas pico y valle
axes[1].axvspan(2, 5, alpha=0.1, color='blue', label='Valle nocturno')
axes[1].axvspan(7, 9, alpha=0.1, color='red', label='Pico matutino')
axes[1].axvspan(19, 21, alpha=0.1, color='orange', label='Pico vespertino')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/boxplots_horarios/boxplots_por_hora.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/boxplots_horarios/boxplots_por_hora.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: boxplots_por_hora.png/.pdf")

# ============================================================================
# 4. DISTRIBUCIONES KDE
# ============================================================================

print("\n📊 Generando distribuciones KDE...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# KDE Qin
axes[0].hist(df_qin[qin_col].dropna(), bins=50, density=True, alpha=0.6, 
             color='steelblue', edgecolor='black', label='Histograma')
df_qin[qin_col].dropna().plot.kde(ax=axes[0], color='darkblue', linewidth=3, label='KDE')
axes[0].set_title('Distribución de Densidad: Producción (Qin)', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Qin (m³/h)', fontweight='bold')
axes[0].set_ylabel('Densidad', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right')

# Estadísticas
mean_qin = df_qin[qin_col].mean()
median_qin = df_qin[qin_col].median()
std_qin = df_qin[qin_col].std()
axes[0].axvline(mean_qin, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_qin:.0f}')
axes[0].axvline(median_qin, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_qin:.0f}')

stats_text = f'Media: {mean_qin:.0f} m³/h\nMediana: {median_qin:.0f} m³/h\nDesv. Est.: {std_qin:.0f} m³/h'
axes[0].text(0.65, 0.95, stats_text, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# KDE D_obs
axes[1].hist(df_merged['D_obs'].dropna(), bins=50, density=True, alpha=0.6, 
             color='orange', edgecolor='black', label='Histograma')
df_merged['D_obs'].dropna().plot.kde(ax=axes[1], color='darkorange', linewidth=3, label='KDE')
axes[1].set_title('Distribución de Densidad: Demanda Observada (D_obs)', 
                  fontweight='bold', fontsize=14)
axes[1].set_xlabel('D_obs (m³/h)', fontweight='bold')
axes[1].set_ylabel('Densidad', fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')

mean_dobs = df_merged['D_obs'].mean()
median_dobs = df_merged['D_obs'].median()
std_dobs = df_merged['D_obs'].std()
axes[1].axvline(mean_dobs, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_dobs:.0f}')
axes[1].axvline(median_dobs, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_dobs:.0f}')

stats_text = f'Media: {mean_dobs:.0f} m³/h\nMediana: {median_dobs:.0f} m³/h\nDesv. Est.: {std_dobs:.0f} m³/h'
axes[1].text(0.65, 0.95, stats_text, transform=axes[1].transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/distribuciones_kde/distribuciones_kde.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/distribuciones_kde/distribuciones_kde.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: distribuciones_kde.png/.pdf")

# ============================================================================
# 5. COMPOSICIÓN DE CAPACIDAD POR ESTANQUE
# ============================================================================

print("\n📊 Generando gráfico de composición de capacidad por estanque...")

# Preparar datos de capacidad
if 'Capacity_m3' in df_capacity.columns:
    cap_col = 'Capacity_m3'
elif 'Capacidad_m3' in df_capacity.columns:
    cap_col = 'Capacidad_m3'
else:
    cap_col = [c for c in df_capacity.columns if 'cap' in c.lower()][0]

if 'Tank_ID' in df_capacity.columns:
    tank_col = 'Tank_ID'
elif 'Estanque_ID' in df_capacity.columns:
    tank_col = 'Estanque_ID'
else:
    tank_col = df_capacity.columns[0]

# Ordenar por capacidad
df_capacity_sorted = df_capacity.sort_values(cap_col, ascending=False)

# Tomar top 10 estanques y agrupar el resto
top_n = 10
if len(df_capacity_sorted) > top_n:
    top_tanks = df_capacity_sorted.head(top_n)
    otros_capacidad = df_capacity_sorted.iloc[top_n:][cap_col].sum()
    
    labels = list(top_tanks[tank_col].astype(str)) + ['Otros']
    sizes = list(top_tanks[cap_col]) + [otros_capacidad]
else:
    labels = list(df_capacity_sorted[tank_col].astype(str))
    sizes = list(df_capacity_sorted[cap_col])

# Crear gráfico circular
fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.Set3(range(len(sizes)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                    colors=colors, startangle=90,
                                    textprops=dict(color="black", fontsize=10))

# Mejorar legibilidad de porcentajes
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax.set_title('Composición de Capacidad por Estanque\n(Top 10 + Otros)', 
             fontweight='bold', fontsize=14, pad=20)

# Añadir información total
capacidad_total = df_capacity[cap_col].sum()
ax.text(0.5, -0.1, f'Capacidad Total: {capacidad_total:,.0f} m³ | Número de estanques: {len(df_capacity)}',
        ha='center', transform=ax.transAxes, fontsize=11, style='italic')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/capacidad_estanques/composicion_capacidad_estanques.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/capacidad_estanques/composicion_capacidad_estanques.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: composicion_capacidad_estanques.png/.pdf")

# Gráfico de barras de capacidad (alternativa)
fig, ax = plt.subplots(figsize=(14, 8))
df_capacity_sorted_top = df_capacity_sorted.head(20)
ax.barh(range(len(df_capacity_sorted_top)), df_capacity_sorted_top[cap_col], 
        color='steelblue', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(df_capacity_sorted_top)))
ax.set_yticklabels(df_capacity_sorted_top[tank_col].astype(str), fontsize=9)
ax.set_xlabel('Capacidad (m³)', fontweight='bold')
ax.set_ylabel('Estanque ID', fontweight='bold')
ax.set_title('Capacidad Individual por Estanque (Top 20)', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/capacidad_estanques/capacidad_barras_top20.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/capacidad_estanques/capacidad_barras_top20.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: capacidad_barras_top20.png/.pdf")

# ============================================================================
# 6. PORCENTAJE DE LLENADO POR ESTANQUE (SERIES TEMPORALES)
# ============================================================================

print("\n📊 Generando series temporales de porcentaje de llenado por estanque...")

# Identificar columnas de volumen por estanque
volume_cols = [c for c in df_tanks.columns if c not in ['timestamp', 'Timestamp']]

# Calcular porcentajes de llenado
df_fill_pct = df_tanks.copy()

# Merge con capacidades
capacity_dict = dict(zip(df_capacity[tank_col].astype(str), df_capacity[cap_col]))

for col in volume_cols:
    tank_id = col.replace('Vol_', '').replace('_m3', '')
    if tank_id in capacity_dict:
        df_fill_pct[f'{col}_pct'] = (df_fill_pct[col] / capacity_dict[tank_id]) * 100

# Columnas de porcentaje
pct_cols = [c for c in df_fill_pct.columns if '_pct' in c]

# Seleccionar estanques más relevantes (mayor variabilidad o capacidad)
if len(pct_cols) > 15:
    # Calcular variabilidad
    variabilidad = {}
    for col in pct_cols:
        variabilidad[col] = df_fill_pct[col].std()
    
    # Top 15 por variabilidad
    top_pct_cols = sorted(variabilidad.items(), key=lambda x: x[1], reverse=True)[:15]
    pct_cols_selected = [c[0] for c in top_pct_cols]
else:
    pct_cols_selected = pct_cols

# Gráfico de series temporales
fig, ax = plt.subplots(figsize=(16, 10))

for col in pct_cols_selected:
    tank_name = col.replace('_pct', '').replace('Vol_', '')
    ax.plot(df_fill_pct['timestamp'], df_fill_pct[col], 
            linewidth=1.5, alpha=0.7, label=tank_name)

ax.set_title('Evolución del Porcentaje de Llenado por Estanque\n(Estanques con mayor variabilidad)', 
             fontweight='bold', fontsize=14)
ax.set_xlabel('Fecha', fontweight='bold')
ax.set_ylabel('Porcentaje de llenado (%)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Capacidad máxima')
ax.axhline(50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='50% capacidad')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/llenado_estanques/series_temporales_llenado.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/llenado_estanques/series_temporales_llenado.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: series_temporales_llenado.png/.pdf")

# Heatmap de porcentajes de llenado
fig, ax = plt.subplots(figsize=(16, 10))

# Crear matriz para heatmap (submuestrear si hay muchos datos)
df_sample = df_fill_pct.copy()
if len(df_sample) > 1000:
    step = len(df_sample) // 1000
    df_sample = df_sample.iloc[::step]

matriz_llenado = df_sample[pct_cols_selected].T.values

im = ax.imshow(matriz_llenado, aspect='auto', cmap='RdYlGn', 
               interpolation='nearest', vmin=0, vmax=100)

ax.set_title('Mapa de Calor: Porcentaje de Llenado por Estanque', 
             fontweight='bold', fontsize=14)
ax.set_xlabel('Tiempo (muestras)', fontweight='bold')
ax.set_ylabel('Estanque', fontweight='bold')

# Etiquetas de estanques
tank_labels = [c.replace('_pct', '').replace('Vol_', '') for c in pct_cols_selected]
ax.set_yticks(range(len(tank_labels)))
ax.set_yticklabels(tank_labels, fontsize=9)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Porcentaje de llenado (%)')
cbar.ax.set_ylabel('Porcentaje de llenado (%)', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/llenado_estanques/heatmap_llenado.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/llenado_estanques/heatmap_llenado.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: heatmap_llenado.png/.pdf")

# Boxplot de distribución de llenado por estanque
fig, ax = plt.subplots(figsize=(14, 8))

datos_boxplot = [df_fill_pct[col].dropna().values for col in pct_cols_selected]
tank_labels_short = [c.replace('_pct', '').replace('Vol_', '')[:10] for c in pct_cols_selected]

bp = ax.boxplot(datos_boxplot, labels=tank_labels_short, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='green', linewidth=1.5),
                capprops=dict(color='green', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))

ax.set_title('Distribución de Porcentaje de Llenado por Estanque', 
             fontweight='bold', fontsize=14)
ax.set_xlabel('Estanque ID', fontweight='bold')
ax.set_ylabel('Porcentaje de llenado (%)', fontweight='bold')
ax.set_xticklabels(tank_labels_short, rotation=45, ha='right', fontsize=9)
ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Capacidad máxima')
ax.axhline(50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='50% capacidad')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/llenado_estanques/boxplot_llenado_por_estanque.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/llenado_estanques/boxplot_llenado_por_estanque.pdf', 
            bbox_inches='tight')
plt.close()

print("   ✓ Guardado: boxplot_llenado_por_estanque.png/.pdf")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETO FINALIZADO")
print("="*80)
print(f"\n📁 Todas las visualizaciones guardadas en: {OUTPUT_DIR}")
print("\nEstructura de carpetas generadas:")
print("├── series_temporales/")
print("│   └── series_temporales_principales.png/.pdf")
print("├── mapas_frecuencia/")
print("│   ├── mapas_presencia_por_hora.png/.pdf")
print("│   └── frecuencia_por_hora.png/.pdf")
print("├── boxplots_horarios/")
print("│   └── boxplots_por_hora.png/.pdf")
print("├── distribuciones_kde/")
print("│   └── distribuciones_kde.png/.pdf")
print("├── capacidad_estanques/")
print("│   ├── composicion_capacidad_estanques.png/.pdf")
print("│   └── capacidad_barras_top20.png/.pdf")
print("└── llenado_estanques/")
print("    ├── series_temporales_llenado.png/.pdf")
print("    ├── heatmap_llenado.png/.pdf")
print("    └── boxplot_llenado_por_estanque.png/.pdf")

print("\n📊 Total de visualizaciones generadas: 18 archivos (9 PNG + 9 PDF)")
print("\n" + "="*80)
