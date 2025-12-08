"""
SCRIPT SIMPLIFICADO - Completa las figuras 4.6, 4.7 y 4.8
==========================================================
Ejecuta solo las figuras faltantes del análisis exploratorio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['savefig.dpi'] = 300

BASE_DIR = 'c:/Users/socce/Downloads/rafa/Tesis3.0-Interfaz'
DATA_DIR = f'{BASE_DIR}/data/raw'
OUTPUT_DIR = f'{BASE_DIR}/outputs/analisis_exploratorio'

print("\n" + "="*80)
print("COMPLETANDO FIGURAS 4.6, 4.7 y 4.8")
print("="*80)

# Cargar datos
print("\n📂 Cargando datos...")
df_qin = pd.read_csv(f'{DATA_DIR}/BD_Qin_m3_UTC.csv')
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
qin_col = [c for c in df_qin.columns if c != 'timestamp'][0]
df_qin = df_qin.rename(columns={qin_col: 'Qin'})

df_vol = pd.read_csv(f'{DATA_DIR}/BD_VolTotal_X_Hr_m3_Local.csv')
df_vol.columns = df_vol.columns.str.strip()
df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'])
vol_col = [c for c in df_vol.columns if c != 'timestamp'][0]
df_vol = df_vol.rename(columns={vol_col: 'Volumen_Total'})

df_qnet = pd.read_csv(f'{DATA_DIR}/BD_Q_net_x_Hr_m3h_LIMPIO.csv')
df_qnet.columns = df_qnet.columns.str.strip()
df_qnet['timestamp'] = pd.to_datetime(df_qnet['timestamp'])
# Ya viene con Q_net_m3h estandarizado

df_clima = pd.read_csv(f'{DATA_DIR}/BD_Clima2024a202509_Local.csv')
df_clima.columns = df_clima.columns.str.strip()
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima = df_clima.rename(columns={'temp': 'Temperatura', 'HR': 'Humedad', 'mmhr': 'Precipitacion'})

# Merge
df = df_qin.merge(df_vol, on='timestamp', how='inner')
df = df.merge(df_qnet, on='timestamp', how='inner')
df = df.merge(df_clima, on='timestamp', how='inner')
df['D_obs'] = df['Qin'] - df['Q_net_m3h']

print(f"✓ Dataset listo: {len(df):,} registros\n")

# ==========================================================================
# FIGURA 4.6: RELACIÓN PRODUCCIÓN-DEMANDA
# ==========================================================================

print("📊 Generando Figura 4.6...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Serie temporal dual-axis
ax1 = fig.add_subplot(gs[0, :])
ax2 = ax1.twinx()

df_plot = df.iloc[::24]  # Cada 24 horas
line1 = ax1.plot(df_plot['timestamp'], df_plot['Qin'], 
                color='steelblue', linewidth=2, alpha=0.8, label='Producción (Qin)')
line2 = ax2.plot(df_plot['timestamp'], df_plot['D_obs'], 
                color='orange', linewidth=2, alpha=0.8, label='Demanda (D_obs)')

ax1.set_xlabel('Fecha', fontweight='bold')
ax1.set_ylabel('Producción Qin (m³/h)', fontweight='bold', color='steelblue')
ax2.set_ylabel('Demanda D_obs (m³/h)', fontweight='bold', color='orange')
ax1.set_title('Series Temporales: Producción vs Demanda', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='orange')
ax1.grid(True, alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

# Scatter plot con regresión
ax3 = fig.add_subplot(gs[1, 0])
df_scatter = df.sample(min(5000, len(df)))
ax3.scatter(df_scatter['Qin'], df_scatter['D_obs'], 
           alpha=0.3, s=10, color='steelblue', edgecolors='none')

# Regresión con sklearn (más estable) - eliminar NaN
from sklearn.linear_model import LinearRegression
df_clean = df[['Qin', 'D_obs']].dropna()
X = df_clean[['Qin']].values
y = df_clean['D_obs'].values
reg = LinearRegression().fit(X, y)
x_reg = np.linspace(df_clean['Qin'].min(), df_clean['Qin'].max(), 100).reshape(-1, 1)
y_reg = reg.predict(x_reg)
ax3.plot(x_reg, y_reg, "r--", linewidth=3, 
        label=f'y = {reg.coef_[0]:.3f}x + {reg.intercept_:.1f}')

ax3.plot([df_clean['Qin'].min(), df_clean['Qin'].max()], 
         [df_clean['Qin'].min(), df_clean['Qin'].max()], 
         'k--', linewidth=2, alpha=0.5, label='y = x (identidad)')

corr = df_clean['Qin'].corr(df_clean['D_obs'])
ax3.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax3.set_xlabel('Producción Qin (m³/h)', fontweight='bold')
ax3.set_ylabel('Demanda D_obs (m³/h)', fontweight='bold')
ax3.set_title('Relación Producción-Demanda', fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# Histograma 2D
ax4 = fig.add_subplot(gs[1, 1])
h = ax4.hist2d(df_clean['Qin'], df_clean['D_obs'], bins=50, cmap='YlOrRd', cmin=1)
plt.colorbar(h[3], ax=ax4, label='Frecuencia')

ax4.set_xlabel('Producción Qin (m³/h)', fontweight='bold')
ax4.set_ylabel('Demanda D_obs (m³/h)', fontweight='bold')
ax4.set_title('Densidad Conjunta', fontweight='bold')

fig.suptitle('Figura 4.6: Análisis de la Relación entre Producción y Demanda', 
             fontweight='bold', fontsize=15, y=0.995)

plt.savefig(f'{OUTPUT_DIR}/fig_4_6_produccion_demanda.png', bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig_4_6_produccion_demanda.pdf', bbox_inches='tight')
plt.close()

print("✓ Figura 4.6 guardada")

# ==========================================================================
# FIGURA 4.7: ANÁLISIS DE OUTLIERS
# ==========================================================================

print("\n📊 Generando Figura 4.7...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
variables_outliers = ['D_obs', 'Qin', 'Temperatura', 'Volumen_Total']

for idx, var in enumerate(variables_outliers):
    ax = axes[idx // 2, idx % 2]
    
    # Boxplot + violinplot
    parts = ax.violinplot([df[var].dropna()], positions=[0], widths=0.7,
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(sns.color_palette("Set2")[idx])
        pc.set_alpha(0.6)
    
    bp = ax.boxplot([df[var].dropna()], positions=[0], widths=0.3,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                    medianprops=dict(color='red', linewidth=3),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', 
                                   markersize=4, alpha=0.5))
    
    # Outliers método IQR
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
    pct_outliers = len(outliers) / len(df) * 100
    
    stats_text = (f'Media: {df[var].mean():.1f}\n'
                 f'Mediana: {df[var].median():.1f}\n'
                 f'Desv.Est: {df[var].std():.1f}\n'
                 f'Outliers: {len(outliers)} ({pct_outliers:.2f}%)\n'
                 f'Rango IQR: [{lower_bound:.1f}, {upper_bound:.1f}]')
    
    ax.text(0.55, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_ylabel(var, fontweight='bold')
    ax.set_title(f'{var}', fontweight='bold')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figura 4.7: Análisis de Outliers (Boxplot + Violinplot + Método IQR)', 
             fontweight='bold', fontsize=15, y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{OUTPUT_DIR}/fig_4_7_outliers.png', bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig_4_7_outliers.pdf', bbox_inches='tight')
plt.close()

print("✓ Figura 4.7 guardada")

# ==========================================================================
# FIGURA 4.8: TESTS DE NORMALIDAD
# ==========================================================================

print("\n📊 Generando Figura 4.8...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
variables_norm = ['D_obs', 'Qin', 'Q_net_m3h', 'Temperatura', 'Humedad', 'Volumen_Total']

for idx, var in enumerate(variables_norm):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Q-Q plot
    stats.probplot(df[var].dropna(), dist="norm", plot=ax)
    
    # Tests de normalidad
    sample_data = df[var].dropna().sample(min(5000, len(df[var].dropna())))
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
    ks_stat, ks_p = stats.kstest(df[var].dropna(), 'norm', 
                                 args=(df[var].mean(), df[var].std()))
    
    if shapiro_p < 0.01:
        resultado = "NO Normal"
        color = 'red'
    else:
        resultado = "Normal"
        color = 'green'
    
    test_text = (f'Shapiro-Wilk:\n'
                f'  p-valor: {shapiro_p:.4f}\n'
                f'Kolmogorov-Smirnov:\n'
                f'  p-valor: {ks_p:.4f}\n'
                f'Resultado: {resultado}')
    
    ax.text(0.05, 0.95, test_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor=color, linewidth=2))
    
    ax.set_title(f'{var}', fontweight='bold')
    ax.set_xlabel('Cuantiles teóricos (Normal)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Cuantiles observados', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('Figura 4.8: Análisis de Normalidad - Q-Q Plots y Tests Estadísticos\n' +
             '(Justificación de uso de métodos no paramétricos y modelos ML robustos)', 
             fontweight='bold', fontsize=15, y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f'{OUTPUT_DIR}/fig_4_8_normalidad_qqplots.png', bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig_4_8_normalidad_qqplots.pdf', bbox_inches='tight')
plt.close()

print("✓ Figura 4.8 guardada")

print("\n" + "="*80)
print("✅ FIGURAS 4.6, 4.7 Y 4.8 COMPLETADAS")
print("="*80)
print(f"\n📁 Archivos guardados en: {OUTPUT_DIR}")
print("\n📄 Archivos generados:")
print("   • fig_4_6_produccion_demanda.png/.pdf")
print("   • fig_4_7_outliers.png/.pdf")
print("   • fig_4_8_normalidad_qqplots.png/.pdf")
print("\n" + "="*80)
