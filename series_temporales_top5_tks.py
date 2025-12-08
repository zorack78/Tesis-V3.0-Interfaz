"""
Series Temporales de Top 5 Estanques por Capacidad

Genera visualizaciones de series temporales para los 5 estanques
con mayor capacidad de almacenamiento:
- Vigia
- Lyon
- Rodriguez
- Eduardo_Aguirre_1
- San_Guillermo

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

# Rutas
DATA_RAW = Path('data/raw')
OUTPUT_DIR = Path('outputs/series_temporales_top5_tks')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("SERIES TEMPORALES - TOP 5 ESTANQUES POR CAPACIDAD")
print("="*80)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("\n[1/4] Cargando datos de volúmenes por estanque...")

df = pd.read_csv(DATA_RAW / 'Vol_X_TK_Hr_m3_UTC.csv')
df.columns = df.columns.str.strip()
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

print(f"   ✓ Registros: {len(df):,}")
print(f"   ✓ Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
print(f"   ✓ Total de estanques: {len(df.columns) - 1}")

# Estanques Top 5
top5_tanks = ['Vigia', 'Lyon', 'Rodriguez', 'Eduardo_Aguirre_1', 'San_Guillermo']

# Verificar que existen
for tank in top5_tanks:
    if tank not in df.columns:
        print(f"   ⚠ ADVERTENCIA: {tank} no encontrado en el archivo")
    else:
        print(f"   ✓ {tank} encontrado")

# ============================================================================
# VISUALIZACIÓN 1: SERIES TEMPORALES COMPLETAS (5 SUBPLOTS)
# ============================================================================

print("\n[2/4] Generando series temporales individuales...")

fig, axes = plt.subplots(5, 1, figsize=(18, 16))
fig.patch.set_facecolor('#F8F9FA')

colores = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

for idx, (tank, color) in enumerate(zip(top5_tanks, colores)):
    ax = axes[idx]
    
    # Serie temporal
    ax.plot(df['timestamp'], df[tank], color=color, linewidth=1, alpha=0.8)
    
    # Media móvil 24 horas
    ma24 = df[tank].rolling(window=24, center=True).mean()
    ax.plot(df['timestamp'], ma24, color='red', linewidth=2, 
           label='Media Móvil 24h', alpha=0.9)
    
    # Estadísticas
    media = df[tank].mean()
    std = df[tank].std()
    ax.axhline(media, color='black', linestyle='--', linewidth=1.5,
              alpha=0.7, label=f'Media: {media:,.0f} m³')
    
    # Configuración
    ax.set_ylabel('Volumen (m³)', fontsize=12, fontweight='bold')
    ax.set_title(f'({chr(65+idx)}) Estanque {tank.replace("_", " ")}',
                fontsize=13, fontweight='bold', pad=10, color='#2C3E50')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Solo xlabel en el último
    if idx == len(top5_tanks) - 1:
        ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    
    # Info de rango
    minimo = df[tank].min()
    maximo = df[tank].max()
    ax.text(0.02, 0.98, f'Min: {minimo:,.0f} m³\nMax: {maximo:,.0f} m³\nΔ: {maximo-minimo:,.0f} m³',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Series Temporales de Top 5 Estanques por Capacidad\n' +
             'Enero 2024 - Septiembre 2025',
             fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'series_temporales_top5_individuales.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'series_temporales_top5_individuales.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: series_temporales_top5_individuales.png/pdf")

# ============================================================================
# VISUALIZACIÓN 2: SERIES SUPERPUESTAS (COMPARACIÓN)
# ============================================================================

print("\n[3/4] Generando comparación superpuesta...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
fig.patch.set_facecolor('#F8F9FA')

# Panel superior: Valores absolutos
for tank, color in zip(top5_tanks, colores):
    ax1.plot(df['timestamp'], df[tank], color=color, linewidth=1.5,
            label=tank.replace('_', ' '), alpha=0.8)

ax1.set_ylabel('Volumen (m³)', fontsize=13, fontweight='bold')
ax1.set_title('(A) Series Temporales Superpuestas - Valores Absolutos',
             fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=11, ncol=2)

# Panel inferior: Valores normalizados (% de capacidad)
capacidades = {
    'Vigia': 13000,  # Aproximado según capacidades típicas
    'Lyon': 12000,
    'Rodriguez': 11000,
    'Eduardo_Aguirre_1': 10500,
    'San_Guillermo': 11500
}

for tank, color in zip(top5_tanks, colores):
    # Normalizar a % de capacidad (asumiendo capacidad máxima observada)
    max_cap = df[tank].max()
    pct = (df[tank] / max_cap) * 100
    ax2.plot(df['timestamp'], pct, color=color, linewidth=1.5,
            label=tank.replace('_', ' '), alpha=0.8)

ax2.set_xlabel('Fecha', fontsize=13, fontweight='bold')
ax2.set_ylabel('% de Capacidad Máxima Observada', fontsize=13, fontweight='bold')
ax2.set_title('(B) Series Temporales Normalizadas - Porcentaje de Llenado',
             fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=11, ncol=2)
ax2.set_ylim([0, 105])

# Líneas de referencia
ax2.axhline(100, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Capacidad Máxima')
ax2.axhline(50, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='50% Capacidad')
ax2.axhline(25, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5, label='25% Capacidad')

fig.suptitle('Comparación de Series Temporales - Top 5 Estanques',
             fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'series_temporales_top5_comparacion.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'series_temporales_top5_comparacion.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: series_temporales_top5_comparacion.png/pdf")

# ============================================================================
# VISUALIZACIÓN 3: HEATMAP TEMPORAL
# ============================================================================

print("\n[4/4] Generando heatmap temporal...")

# Agregar columnas temporales
df['Fecha'] = df['timestamp'].dt.date
df['Hora'] = df['timestamp'].dt.hour

fig, axes = plt.subplots(5, 1, figsize=(18, 20))
fig.patch.set_facecolor('#F8F9FA')

for idx, (tank, color) in enumerate(zip(top5_tanks, colores)):
    ax = axes[idx]
    
    # Crear pivot para heatmap (primeros 60 días)
    df_subset = df[df['timestamp'] < df['timestamp'].min() + pd.Timedelta(days=60)]
    pivot = df_subset.pivot_table(values=tank, index='Hora', columns='Fecha', aggfunc='mean')
    
    # Heatmap
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
    
    # Configuración
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
    ax.set_ylabel('Hora del Día', fontsize=11, fontweight='bold')
    
    # Etiquetas de fecha cada 5 días
    fecha_labels = [pivot.columns[i] for i in range(0, len(pivot.columns), 5)]
    fecha_positions = list(range(0, len(pivot.columns), 5))
    ax.set_xticks(fecha_positions)
    ax.set_xticklabels([str(f) for f in fecha_labels], rotation=45, ha='right', fontsize=8)
    
    ax.set_title(f'({chr(65+idx)}) Heatmap: {tank.replace("_", " ")} - Primeros 60 días',
                fontsize=12, fontweight='bold', pad=10, color='#2C3E50')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01, fraction=0.046)
    cbar.set_label('Volumen (m³)', fontsize=10)
    
    if idx == len(top5_tanks) - 1:
        ax.set_xlabel('Fecha', fontsize=11, fontweight='bold')

fig.suptitle('Heatmaps Temporales - Variación Hora×Día (Primeros 60 días)\n' +
             'Top 5 Estanques por Capacidad',
             fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'series_temporales_top5_heatmap.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'series_temporales_top5_heatmap.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("   ✓ Guardado: series_temporales_top5_heatmap.png/pdf")

# ============================================================================
# ESTADÍSTICAS Y EXPORT CSV
# ============================================================================

print("\nGenerando resumen estadístico...")

# Calcular estadísticas para cada estanque
stats_list = []
for tank in top5_tanks:
    stats = {
        'Estanque': tank.replace('_', ' '),
        'Media_m3': df[tank].mean(),
        'Mediana_m3': df[tank].median(),
        'Desv_Std_m3': df[tank].std(),
        'Min_m3': df[tank].min(),
        'Max_m3': df[tank].max(),
        'Rango_m3': df[tank].max() - df[tank].min(),
        'CV_%': (df[tank].std() / df[tank].mean()) * 100,
        'Q1_m3': df[tank].quantile(0.25),
        'Q3_m3': df[tank].quantile(0.75),
        'IQR_m3': df[tank].quantile(0.75) - df[tank].quantile(0.25)
    }
    stats_list.append(stats)

stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(OUTPUT_DIR / 'estadisticas_top5_estanques.csv', index=False)
print("   ✓ Guardado: estadisticas_top5_estanques.csv")

# Exportar series temporales completas para los top 5
df_export = df[['timestamp'] + top5_tanks].copy()
df_export.to_csv(OUTPUT_DIR / 'series_temporales_top5_datos.csv', index=False)
print("   ✓ Guardado: series_temporales_top5_datos.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("ESTADÍSTICAS DE LOS TOP 5 ESTANQUES")
print("="*80)
print("\n" + stats_df.to_string(index=False))

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"\nTodos los archivos guardados en: {OUTPUT_DIR.absolute()}")
print("\nArchivos generados:")
print("  📊 series_temporales_top5_individuales.png/pdf")
print("     - 5 subplots con series individuales y medias móviles")
print("\n  📊 series_temporales_top5_comparacion.png/pdf")
print("     - Panel A: Series superpuestas en valores absolutos")
print("     - Panel B: Series normalizadas (% de capacidad)")
print("\n  📊 series_temporales_top5_heatmap.png/pdf")
print("     - 5 heatmaps mostrando variación Hora×Día (primeros 60 días)")
print("\n  📄 estadisticas_top5_estanques.csv")
print("     - Resumen estadístico completo de cada estanque")
print("\n  📄 series_temporales_top5_datos.csv")
print("     - Series temporales completas exportadas")
print("\nTotal: 3 visualizaciones (6 archivos PNG+PDF) + 2 archivos CSV")
print("="*80)
