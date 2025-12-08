"""
Análisis de Distribución Estadística para Variables del Sistema

Genera histogramas y gráficos KDE (Kernel Density Estimation) para cada
variable principal del sistema de distribución de agua potable.

Outputs organizados en: outputs/distribuciones_estadisticas/

Autor: Sistema de Análisis de Demanda de Agua
Fecha: Diciembre 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas
DATA_RAW = Path('data/raw')
OUTPUT_DIR = Path('outputs/distribuciones_estadisticas')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS DE DISTRIBUCIÓN ESTADÍSTICA - VARIABLES DEL SISTEMA")
print("="*80)

# ============================================================================
# FUNCIÓN PARA ANÁLISIS COMPLETO DE DISTRIBUCIÓN
# ============================================================================

def analizar_distribucion(df, columna, nombre_variable, color_principal, archivo_base):
    """
    Genera análisis completo de distribución con múltiples visualizaciones
    
    Parameters:
    -----------
    df : DataFrame
        Datos a analizar
    columna : str
        Nombre de la columna a analizar
    nombre_variable : str
        Nombre descriptivo para títulos
    color_principal : str
        Color hex para visualizaciones
    archivo_base : str
        Nombre base para archivos de salida
    """
    
    data = df[columna].dropna()
    
    # Calcular estadísticas
    media = data.mean()
    mediana = data.median()
    std = data.std()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    minimo = data.min()
    maximo = data.max()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Test de normalidad
    shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(media, std))
    
    print(f"\n{'='*70}")
    print(f"ESTADÍSTICAS: {nombre_variable}")
    print(f"{'='*70}")
    print(f"  N observaciones:    {len(data):,}")
    print(f"  Media:              {media:,.2f}")
    print(f"  Mediana:            {mediana:,.2f}")
    print(f"  Desv. Estándar:     {std:,.2f}")
    print(f"  Mínimo:             {minimo:,.2f}")
    print(f"  Q1 (25%):           {q1:,.2f}")
    print(f"  Q3 (75%):           {q3:,.2f}")
    print(f"  Máximo:             {maximo:,.2f}")
    print(f"  IQR:                {iqr:,.2f}")
    print(f"  Rango:              {maximo - minimo:,.2f}")
    print(f"  Coef. Variación:    {(std/media)*100:.2f}%")
    print(f"  Asimetría (Skew):   {skewness:.3f}")
    print(f"  Curtosis:           {kurtosis:.3f}")
    print(f"\n  Test Shapiro-Wilk:  stat={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")
    print(f"  Test Kolmogorov-S:  stat={ks_stat:.4f}, p-value={ks_p:.4e}")
    
    # ========================================================================
    # VISUALIZACIÓN 1: HISTOGRAMA + KDE + CURVA NORMAL
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Histograma
    n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.6, 
                                color=color_principal, edgecolor='black', 
                                linewidth=0.5, label='Histograma')
    
    # Colorear barras por altura (gradiente)
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # KDE (Kernel Density Estimation)
    kde_x = np.linspace(data.min(), data.max(), 500)
    kde = stats.gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2.5, label='KDE (Densidad)', alpha=0.8)
    
    # Curva normal teórica
    norm_x = np.linspace(data.min(), data.max(), 500)
    norm_y = stats.norm.pdf(norm_x, media, std)
    ax.plot(norm_x, norm_y, 'b--', linewidth=2, label='Distribución Normal', alpha=0.7)
    
    # Líneas verticales de estadísticas
    ax.axvline(media, color='red', linestyle='-', linewidth=2, 
              label=f'Media: {media:,.1f}', alpha=0.8)
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, 
              label=f'Mediana: {mediana:,.1f}', alpha=0.8)
    ax.axvline(q1, color='orange', linestyle=':', linewidth=1.5, 
              label=f'Q1: {q1:,.1f}', alpha=0.6)
    ax.axvline(q3, color='orange', linestyle=':', linewidth=1.5, 
              label=f'Q3: {q3:,.1f}', alpha=0.6)
    
    # Configuración
    ax.set_xlabel(nombre_variable, fontsize=13, fontweight='bold')
    ax.set_ylabel('Densidad de Probabilidad', fontsize=13, fontweight='bold')
    ax.set_title(f'Distribución Estadística: {nombre_variable}\n' +
                 f'Histograma + KDE + Curva Normal',
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Texto con estadísticas
    stats_text = (f'n = {len(data):,} | μ = {media:,.1f} | σ = {std:,.1f}\n'
                  f'Skew = {skewness:.3f} | Kurt = {kurtosis:.3f}\n'
                  f'Shapiro p-value = {shapiro_p:.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{archivo_base}_histograma_kde.png', 
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig(OUTPUT_DIR / f'{archivo_base}_histograma_kde.pdf', 
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    
    print(f"  ✓ Guardado: {archivo_base}_histograma_kde.png/pdf")
    
    # ========================================================================
    # VISUALIZACIÓN 2: Q-Q PLOT + BOXPLOT + VIOLIN PLOT
    # ========================================================================
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#F8F9FA')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # --- Panel 1: Q-Q Plot ---
    ax1 = fig.add_subplot(gs[0, 0])
    stats.probplot(data, dist="norm", plot=ax1)
    ax1.set_title('(A) Q-Q Plot (vs. Distribución Normal)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Boxplot Horizontal ---
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([data], vert=False, widths=0.5, patch_artist=True,
                      showfliers=True, notch=True,
                      boxprops=dict(facecolor=color_principal, alpha=0.7, 
                                   edgecolor='#2C3E50', linewidth=1.5),
                      whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                      capprops=dict(color='#2C3E50', linewidth=1.5),
                      medianprops=dict(color='#E74C3C', linewidth=2.5),
                      flierprops=dict(marker='o', markerfacecolor='#95A5A6', 
                                     markersize=3, alpha=0.5))
    ax2.set_xlabel(nombre_variable, fontsize=11, fontweight='bold')
    ax2.set_title('(B) Boxplot - Identificación de Outliers', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Marcar media en boxplot
    ax2.axvline(media, color='red', linestyle='--', linewidth=1.5, 
               label=f'Media: {media:,.1f}')
    ax2.legend(fontsize=9)
    
    # --- Panel 3: Violin Plot ---
    ax3 = fig.add_subplot(gs[1, 0])
    parts = ax3.violinplot([data], vert=False, widths=0.7,
                            showmeans=True, showmedians=True, showextrema=True)
    
    # Colorear violin plot
    for pc in parts['bodies']:
        pc.set_facecolor(color_principal)
        pc.set_alpha(0.7)
    
    ax3.set_xlabel(nombre_variable, fontsize=11, fontweight='bold')
    ax3.set_title('(C) Violin Plot - Densidad + Distribución', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # --- Panel 4: Tabla de Estadísticas ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Datos de la tabla
    tabla_data = [
        ['Estadística', 'Valor'],
        ['─'*20, '─'*15],
        ['N observaciones', f'{len(data):,}'],
        ['Media (μ)', f'{media:,.2f}'],
        ['Mediana', f'{mediana:,.2f}'],
        ['Desv. Estándar (σ)', f'{std:,.2f}'],
        ['Varianza', f'{std**2:,.2f}'],
        ['Coef. Variación', f'{(std/media)*100:.2f}%'],
        ['', ''],
        ['Mínimo', f'{minimo:,.2f}'],
        ['Q1 (25%)', f'{q1:,.2f}'],
        ['Q2 (50%)', f'{mediana:,.2f}'],
        ['Q3 (75%)', f'{q3:,.2f}'],
        ['Máximo', f'{maximo:,.2f}'],
        ['IQR', f'{iqr:,.2f}'],
        ['Rango', f'{maximo - minimo:,.2f}'],
        ['', ''],
        ['Asimetría (Skew)', f'{skewness:.3f}'],
        ['Curtosis', f'{kurtosis:.3f}'],
        ['', ''],
        ['Shapiro-Wilk p', f'{shapiro_p:.4f}'],
        ['Kolmogorov-S p', f'{ks_p:.4f}'],
    ]
    
    # Crear tabla
    table = ax4.table(cellText=tabla_data, cellLoc='left',
                      loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Estilizar tabla
    for (i, j), cell in table._cells.items():
        if i == 0:  # Header
            cell.set_facecolor('#3498DB')
            cell.set_text_props(weight='bold', color='white')
        elif i == 1:  # Separador
            cell.set_facecolor('#ECF0F1')
        elif tabla_data[i][0] == '':  # Filas vacías
            cell.set_facecolor('#F8F9FA')
        else:
            cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F8F9FA')
    
    ax4.set_title('(D) Resumen Estadístico', fontsize=12, fontweight='bold', pad=20)
    
    # Título general
    fig.suptitle(f'Análisis de Distribución Completo: {nombre_variable}',
                 fontsize=16, fontweight='bold', y=0.98, color='#2C3E50')
    
    plt.savefig(OUTPUT_DIR / f'{archivo_base}_analisis_completo.png', 
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig(OUTPUT_DIR / f'{archivo_base}_analisis_completo.pdf', 
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    
    print(f"  ✓ Guardado: {archivo_base}_analisis_completo.png/pdf")
    
    # ========================================================================
    # VISUALIZACIÓN 3: HISTOGRAMA POR PERCENTILES (COLOREADO)
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Calcular percentiles para colorear
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    valores_percentiles = [data.quantile(p/100) for p in percentiles]
    
    # Histograma con muchos bins para detalle
    counts, bins, patches = ax.hist(data, bins=100, alpha=0.8, 
                                     edgecolor='black', linewidth=0.3)
    
    # Colorear bins por percentil
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6', '#34495E']
    
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        
        for j in range(len(valores_percentiles) - 1):
            if valores_percentiles[j] <= bin_center < valores_percentiles[j+1]:
                patch.set_facecolor(colors[j])
                break
    
    # Líneas de percentiles
    for i, (p, v) in enumerate(zip(percentiles, valores_percentiles)):
        if p in [0, 100]:
            continue
        ax.axvline(v, color=colors[percentiles.index(p)], linestyle='--', 
                  linewidth=1.5, alpha=0.7)
        ax.text(v, ax.get_ylim()[1] * 0.95, f'P{p}\n{v:,.0f}',
               ha='center', fontsize=8, bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    # Configuración
    ax.set_xlabel(nombre_variable, fontsize=13, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=13, fontweight='bold')
    ax.set_title(f'Histograma por Percentiles: {nombre_variable}\n' +
                 f'Distribución de Frecuencias con Marcadores de Percentiles',
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='P0-P10 (Muy bajo)'),
        Patch(facecolor='#E67E22', label='P10-P25 (Bajo)'),
        Patch(facecolor='#F39C12', label='P25-P50 (Medio-bajo)'),
        Patch(facecolor='#2ECC71', label='P50-P75 (Medio-alto)'),
        Patch(facecolor='#3498DB', label='P75-P90 (Alto)'),
        Patch(facecolor='#9B59B6', label='P90-P100 (Muy alto)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{archivo_base}_histograma_percentiles.png', 
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig(OUTPUT_DIR / f'{archivo_base}_histograma_percentiles.pdf', 
                dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    
    print(f"  ✓ Guardado: {archivo_base}_histograma_percentiles.png/pdf")
    
    # Exportar estadísticas a CSV
    stats_df = pd.DataFrame({
        'Estadística': ['N', 'Media', 'Mediana', 'Desv_Std', 'Varianza', 
                       'CV_%', 'Min', 'Q1', 'Q3', 'Max', 'IQR', 'Rango',
                       'Skewness', 'Kurtosis', 'Shapiro_stat', 'Shapiro_p',
                       'KS_stat', 'KS_p'],
        'Valor': [len(data), media, mediana, std, std**2, (std/media)*100,
                 minimo, q1, q3, maximo, iqr, maximo-minimo,
                 skewness, kurtosis, shapiro_stat, shapiro_p, ks_stat, ks_p]
    })
    stats_df.to_csv(OUTPUT_DIR / f'{archivo_base}_estadisticas.csv', index=False)
    print(f"  ✓ Guardado: {archivo_base}_estadisticas.csv")

# ============================================================================
# ANÁLISIS 1: Q_FLUJO
# ============================================================================

print("\n[1/3] Analizando BD_Q_net_x_Hr_m3h.csv...")

df_flujo = pd.read_csv(DATA_RAW / 'BD_Q_net_x_Hr_m3h.csv')
df_flujo.columns = df_flujo.columns.str.strip()
print(f"  Registros cargados: {len(df_flujo):,}")

analizar_distribucion(
    df=df_flujo,
    columna='Q_flujo_m3hr',
    nombre_variable='Q_flujo (Tasa de Cambio de Volumen) [m³/h]',
    color_principal='#F39C12',
    archivo_base='1_Q_flujo'
)

# ============================================================================
# ANÁLISIS 2: QIN (PRODUCCIÓN)
# ============================================================================

print("\n[2/3] Analizando BD_Qin_m3_UTC.csv...")

df_qin = pd.read_csv(DATA_RAW / 'BD_Qin_m3_UTC.csv')
df_qin.columns = df_qin.columns.str.strip()
print(f"  Registros cargados: {len(df_qin):,}")

analizar_distribucion(
    df=df_qin,
    columna='Qin',
    nombre_variable='Qin (Producción de Agua) [m³/h]',
    color_principal='#3498DB',
    archivo_base='2_Qin'
)

# ============================================================================
# ANÁLISIS 3: VOLUMEN TOTAL
# ============================================================================

print("\n[3/3] Analizando BD_VolTotal_X_Hr_m3_UTC.csv...")

df_vol = pd.read_csv(DATA_RAW / 'BD_VolTotal_X_Hr_m3_UTC.csv')
df_vol.columns = df_vol.columns.str.strip()
print(f"  Registros cargados: {len(df_vol):,}")

analizar_distribucion(
    df=df_vol,
    columna='Volumen_Total_m3',
    nombre_variable='Volumen Total (Almacenamiento) [m³]',
    color_principal='#2ECC71',
    archivo_base='3_Volumen_Total'
)

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"\nTodos los archivos guardados en: {OUTPUT_DIR.absolute()}")
print("\nArchivos generados por variable (3 visualizaciones + 1 CSV):")
print("  • *_histograma_kde.png/pdf - Histograma + KDE + Curva Normal")
print("  • *_analisis_completo.png/pdf - Q-Q Plot + Boxplot + Violin + Tabla")
print("  • *_histograma_percentiles.png/pdf - Histograma coloreado por percentiles")
print("  • *_estadisticas.csv - Resumen estadístico en formato CSV")
print("\nTotal: 9 visualizaciones (18 archivos PNG+PDF) + 3 archivos CSV")
print("="*80)
