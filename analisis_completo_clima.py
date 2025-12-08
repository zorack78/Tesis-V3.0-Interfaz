"""
Análisis Completo de Variables Climáticas

Genera análisis exhaustivos para datos climáticos:
- Distribuciones estadísticas (histogramas + KDE)
- Boxplots por hora del día
- Análisis exploratorio con correlaciones
- Patrones temporales y estacionales

Variables: Temperatura (temp), Humedad Relativa (HR), Precipitación (mmhr)

Outputs organizados en: outputs/analisis_clima/

Autor: Sistema de Análisis de Demanda de Agua
Fecha: Diciembre 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas
DATA_RAW = Path('data/raw')
OUTPUT_DIR = Path('outputs/analisis_clima')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectorios
(OUTPUT_DIR / 'distribuciones').mkdir(exist_ok=True)
(OUTPUT_DIR / 'boxplots').mkdir(exist_ok=True)
(OUTPUT_DIR / 'exploratorio').mkdir(exist_ok=True)
(OUTPUT_DIR / 'temporal').mkdir(exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS COMPLETO DE VARIABLES CLIMÁTICAS")
print("="*80)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("\n[1/5] Cargando datos climáticos...")

df_clima = pd.read_csv(DATA_RAW / 'BD_Clima2024a202509_Local.csv')
df_clima.columns = df_clima.columns.str.strip()
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima['Hora'] = df_clima['timestamp'].dt.hour
df_clima['Dia_Semana'] = df_clima['timestamp'].dt.dayofweek
df_clima['Mes'] = df_clima['timestamp'].dt.month
df_clima['Fecha'] = df_clima['timestamp'].dt.date

print(f"   ✓ Registros: {len(df_clima):,}")
print(f"   ✓ Columnas: {df_clima.columns.tolist()}")
print(f"   ✓ Período: {df_clima['timestamp'].min()} a {df_clima['timestamp'].max()}")
print(f"   ✓ Variables climáticas: temp (°C), HR (%), mmhr (mm/h)")
print(f"   ✓ Zona horaria: Hora Local de Valparaíso, Chile")

# ============================================================================
# ANÁLISIS 1: DISTRIBUCIONES ESTADÍSTICAS
# ============================================================================

print("\n[2/5] Generando distribuciones estadísticas...")

variables = {
    'temp': ('Temperatura (°C)', '#E74C3C'),
    'HR': ('Humedad Relativa (%)', '#3498DB'),
    'mmhr': ('Precipitación (mm/h)', '#2ECC71')
}

for var, (nombre, color) in variables.items():
    print(f"\n  Analizando {nombre}...")
    
    data = df_clima[var].dropna()
    
    # Estadísticas
    media = data.mean()
    mediana = data.median()
    std = data.std()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    minimo = data.min()
    maximo = data.max()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    print(f"    Media: {media:.2f} | Mediana: {mediana:.2f} | Std: {std:.2f}")
    print(f"    Min: {minimo:.2f} | Max: {maximo:.2f} | Skew: {skewness:.3f}")
    
    # Histograma + KDE + Normal
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Histograma con gradiente
    n, bins, patches = ax.hist(data, bins=60, density=True, alpha=0.6,
                                edgecolor='black', linewidth=0.5)
    
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # KDE
    kde_x = np.linspace(data.min(), data.max(), 500)
    kde = stats.gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2.5, label='KDE', alpha=0.8)
    
    # Normal teórica
    norm_x = np.linspace(data.min(), data.max(), 500)
    norm_y = stats.norm.pdf(norm_x, media, std)
    ax.plot(norm_x, norm_y, 'b--', linewidth=2, label='Normal', alpha=0.7)
    
    # Líneas estadísticas
    ax.axvline(media, color='red', linestyle='-', linewidth=2,
              label=f'Media: {media:.1f}', alpha=0.8)
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2,
              label=f'Mediana: {mediana:.1f}', alpha=0.8)
    
    ax.set_xlabel(nombre, fontsize=13, fontweight='bold')
    ax.set_ylabel('Densidad de Probabilidad', fontsize=13, fontweight='bold')
    ax.set_title(f'Distribución Estadística: {nombre}\n' +
                 f'Histograma + KDE + Curva Normal',
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    stats_text = (f'n = {len(data):,} | μ = {media:.2f} | σ = {std:.2f}\n'
                  f'Skew = {skewness:.3f} | Kurt = {kurtosis:.3f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribuciones' / f'{var}_distribucion.png',
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig(OUTPUT_DIR / 'distribuciones' / f'{var}_distribucion.pdf',
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    
    print(f"    ✓ Guardado: {var}_distribucion.png/pdf")

# ============================================================================
# ANÁLISIS 2: BOXPLOTS POR HORA DEL DÍA
# ============================================================================

print("\n[3/5] Generando boxplots por hora del día...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.patch.set_facecolor('#F8F9FA')

colores = ['#E74C3C', '#3498DB', '#2ECC71']
vars_plot = ['temp', 'HR', 'mmhr']
nombres_plot = ['Temperatura (°C)', 'Humedad Relativa (%)', 'Precipitación (mm/h)']

for idx, (var, nombre, color) in enumerate(zip(vars_plot, nombres_plot, colores)):
    ax = axes[idx]
    
    data_by_hour = [df_clima[df_clima['Hora'] == h][var].dropna() for h in range(24)]
    
    bp = ax.boxplot(data_by_hour,
                    positions=range(24),
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    notch=True,
                    boxprops=dict(facecolor=color, alpha=0.7, 
                                 edgecolor='#2C3E50', linewidth=1.5),
                    whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                    capprops=dict(color='#2C3E50', linewidth=1.5),
                    medianprops=dict(color='#E74C3C', linewidth=2.5),
                    flierprops=dict(marker='o', markerfacecolor='#95A5A6',
                                   markersize=3, alpha=0.5))
    
    ax.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
    ax.set_ylabel(nombre, fontsize=12, fontweight='bold')
    ax.set_title(f'({chr(65+idx)}) Distribución Horaria: {nombre}',
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
    
    # Línea de media
    media = df_clima[var].mean()
    ax.axhline(media, color='red', linestyle='--', linewidth=1.5,
              alpha=0.7, label=f'Media: {media:.1f}')
    ax.legend(loc='upper right', fontsize=10)

fig.suptitle('Variación Horaria de Variables Climáticas',
             fontsize=16, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplots' / 'clima_boxplots_por_hora.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'boxplots' / 'clima_boxplots_por_hora.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("  ✓ Guardado: clima_boxplots_por_hora.png/pdf")

# ============================================================================
# ANÁLISIS 3: MATRIZ DE CORRELACIÓN Y PAIRPLOT
# ============================================================================

print("\n[4/5] Generando análisis exploratorio...")

# Matriz de correlación
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#F8F9FA')

corr_matrix = df_clima[['temp', 'HR', 'mmhr']].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('Matriz de Correlación - Variables Climáticas',
            fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
ax.set_xticklabels(['Temperatura', 'Humedad', 'Precipitación'], fontsize=11)
ax.set_yticklabels(['Temperatura', 'Humedad', 'Precipitación'], fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'exploratorio' / 'correlacion_clima.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'exploratorio' / 'correlacion_clima.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("  ✓ Guardado: correlacion_clima.png/pdf")

# Pairplot (scatter matrix)
fig, axes = plt.subplots(3, 3, figsize=(14, 14))
fig.patch.set_facecolor('#F8F9FA')

variables_pair = ['temp', 'HR', 'mmhr']
nombres_pair = ['Temp (°C)', 'HR (%)', 'Precip (mm/h)']

for i, var_y in enumerate(variables_pair):
    for j, var_x in enumerate(variables_pair):
        ax = axes[i, j]
        
        if i == j:  # Diagonal: histogramas
            ax.hist(df_clima[var_x].dropna(), bins=40, alpha=0.7,
                   color=colores[i], edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Frecuencia', fontsize=9)
        else:  # Off-diagonal: scatter plots
            # Muestreo para rendimiento
            sample = df_clima.sample(min(5000, len(df_clima)))
            ax.scatter(sample[var_x], sample[var_y], alpha=0.3, s=10,
                      color=colores[i])
            
            # Correlación
            corr = df_clima[[var_x, var_y]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'r={corr:.3f}',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if i == len(variables_pair) - 1:
            ax.set_xlabel(nombres_pair[j], fontsize=10, fontweight='bold')
        else:
            ax.set_xlabel('')
        
        if j == 0:
            ax.set_ylabel(nombres_pair[i], fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        ax.grid(True, alpha=0.3)

fig.suptitle('Pairplot - Relaciones entre Variables Climáticas',
            fontsize=16, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'exploratorio' / 'pairplot_clima.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'exploratorio' / 'pairplot_clima.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("  ✓ Guardado: pairplot_clima.png/pdf")

# ============================================================================
# ANÁLISIS 4: PATRONES TEMPORALES
# ============================================================================

print("\n[5/5] Generando análisis de patrones temporales...")

# Promedios por hora, día y mes
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.patch.set_facecolor('#F8F9FA')

# Por hora del día
for idx, (var, nombre, color) in enumerate(zip(vars_plot, nombres_plot, colores)):
    ax = axes[0, idx]
    
    hourly = df_clima.groupby('Hora')[var].agg(['mean', 'std', 'count'])
    hourly['se'] = hourly['std'] / np.sqrt(hourly['count'])
    hourly['ci'] = 1.96 * hourly['se']
    
    ax.plot(hourly.index, hourly['mean'], 'o-', color=color,
           linewidth=2, markersize=6, label='Media')
    ax.fill_between(hourly.index,
                    hourly['mean'] - hourly['ci'],
                    hourly['mean'] + hourly['ci'],
                    alpha=0.3, color=color, label='IC 95%')
    
    ax.set_xlabel('Hora del Día', fontsize=10, fontweight='bold')
    ax.set_ylabel(nombre, fontsize=10, fontweight='bold')
    ax.set_title(f'Patrón Horario: {nombre}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, 24, 3))

# Por día de la semana
dias_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
for idx, (var, nombre, color) in enumerate(zip(vars_plot, nombres_plot, colores)):
    ax = axes[1, idx]
    
    weekly = df_clima.groupby('Dia_Semana')[var].agg(['mean', 'std', 'count'])
    weekly['se'] = weekly['std'] / np.sqrt(weekly['count'])
    weekly['ci'] = 1.96 * weekly['se']
    
    ax.bar(weekly.index, weekly['mean'], color=color, alpha=0.7,
          edgecolor='black', linewidth=1)
    ax.errorbar(weekly.index, weekly['mean'], yerr=weekly['ci'],
               fmt='none', ecolor='black', capsize=5, linewidth=1.5)
    
    ax.set_xlabel('Día de la Semana', fontsize=10, fontweight='bold')
    ax.set_ylabel(nombre, fontsize=10, fontweight='bold')
    ax.set_title(f'Patrón Semanal: {nombre}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(7))
    ax.set_xticklabels(dias_nombres, rotation=45)

# Por mes
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
for idx, (var, nombre, color) in enumerate(zip(vars_plot, nombres_plot, colores)):
    ax = axes[2, idx]
    
    monthly = df_clima.groupby('Mes')[var].agg(['mean', 'std', 'count'])
    monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
    monthly['ci'] = 1.96 * monthly['se']
    
    ax.plot(monthly.index, monthly['mean'], 'o-', color=color,
           linewidth=2.5, markersize=8)
    ax.fill_between(monthly.index,
                    monthly['mean'] - monthly['ci'],
                    monthly['mean'] + monthly['ci'],
                    alpha=0.3, color=color)
    
    ax.set_xlabel('Mes', fontsize=10, fontweight='bold')
    ax.set_ylabel(nombre, fontsize=10, fontweight='bold')
    ax.set_title(f'Patrón Mensual: {nombre}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([meses_nombres[m-1] for m in monthly.index], rotation=45)

fig.suptitle('Patrones Temporales de Variables Climáticas\n' +
             'Hora del Día | Día de la Semana | Mes del Año',
            fontsize=16, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal' / 'patrones_temporales_clima.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'temporal' / 'patrones_temporales_clima.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("  ✓ Guardado: patrones_temporales_clima.png/pdf")

# Serie temporal completa
fig, axes = plt.subplots(3, 1, figsize=(16, 10))
fig.patch.set_facecolor('#F8F9FA')

for idx, (var, nombre, color) in enumerate(zip(vars_plot, nombres_plot, colores)):
    ax = axes[idx]
    
    # Agregación diaria para suavizado
    daily = df_clima.groupby('Fecha')[var].mean()
    
    ax.plot(daily.index, daily.values, color=color, linewidth=1, alpha=0.8)
    
    # Media móvil 7 días
    ma7 = daily.rolling(window=7, center=True).mean()
    ax.plot(daily.index, ma7.values, color='red', linewidth=2,
           label='Media Móvil 7 días', alpha=0.9)
    
    ax.set_xlabel('Fecha', fontsize=11, fontweight='bold')
    ax.set_ylabel(nombre, fontsize=11, fontweight='bold')
    ax.set_title(f'({chr(65+idx)}) Serie Temporal: {nombre}',
                fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

fig.suptitle('Series Temporales Completas - Variables Climáticas',
            fontsize=16, fontweight='bold', y=0.995, color='#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal' / 'series_temporales_clima.png',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(OUTPUT_DIR / 'temporal' / 'series_temporales_clima.pdf',
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("  ✓ Guardado: series_temporales_clima.png/pdf")

# ============================================================================
# RESUMEN ESTADÍSTICO Y EXPORT CSV
# ============================================================================

print("\nGenerando resumen estadístico...")

stats_summary = df_clima[['temp', 'HR', 'mmhr']].describe().T
stats_summary['CV_%'] = (stats_summary['std'] / stats_summary['mean'] * 100)
stats_summary['Skew'] = df_clima[['temp', 'HR', 'mmhr']].skew()
stats_summary['Kurt'] = df_clima[['temp', 'HR', 'mmhr']].kurtosis()

stats_summary.to_csv(OUTPUT_DIR / 'estadisticas_clima_resumen.csv')
print("  ✓ Guardado: estadisticas_clima_resumen.csv")

# Correlaciones
corr_matrix.to_csv(OUTPUT_DIR / 'matriz_correlacion_clima.csv')
print("  ✓ Guardado: matriz_correlacion_clima.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"\nTodos los archivos guardados en: {OUTPUT_DIR.absolute()}")
print("\nEstructura de carpetas:")
print("  📁 distribuciones/    - Histogramas + KDE para cada variable (6 archivos)")
print("  📁 boxplots/          - Boxplots por hora del día (2 archivos)")
print("  📁 exploratorio/      - Correlaciones y pairplot (4 archivos)")
print("  📁 temporal/          - Patrones temporales y series (4 archivos)")
print("  📄 estadisticas_clima_resumen.csv")
print("  📄 matriz_correlacion_clima.csv")
print("\nTotal: 16 visualizaciones (PNG + PDF) + 2 archivos CSV")
print("="*80)
