"""
Análisis de Ciclo Semanal (Lunes-Domingo) - Capítulo 4.1.3
Genera gráfica de barras de demanda promedio por día de la semana
Autor: Sistema Predictivo Gran Valparaíso
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración estilo gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("=" * 80)
print("ANÁLISIS DE CICLO SEMANAL (LUNES-DOMINGO)")
print("=" * 80)

# Cargar Volumen Total (almacenamiento)
df_vol = pd.read_csv('data/raw/BD_VolTotal_X_Hr_m3_UTC.csv')
print(f"\n[1] Volumen Total cargado: {len(df_vol):,} registros")

# Cargar Qin (producción)
df_qin = pd.read_csv('data/raw/BD_Qin_m3_UTC.csv')
print(f"    Qin cargado: {len(df_qin):,} registros")

# ============================================================================
# 2. PREPARAR DATOS
# ============================================================================

print("\n[2] PREPARACIÓN DE DATOS:")

# Limpiar nombres de columnas (eliminar espacios)
df_vol.columns = df_vol.columns.str.strip()
df_qin.columns = df_qin.columns.str.strip()

columnas_vol = df_vol.columns.tolist()
columnas_qin = df_qin.columns.tolist()

print(f"   Columnas Volumen: {columnas_vol}")
print(f"   Columnas Qin: {columnas_qin}")

# Identificar columna timestamp en Volumen
col_timestamp_vol = 'timestamp' if 'timestamp' in columnas_vol else columnas_vol[0]

# Identificar columna timestamp en Qin
col_timestamp_qin = 'timestamp' if 'timestamp' in columnas_qin else columnas_qin[0]

# Identificar columna de Volumen
if 'Volumen_Total_m3' in columnas_vol:
    col_vol = 'Volumen_Total_m3'
elif 'Volumen_Total' in columnas_vol:
    col_vol = 'Volumen_Total'
else:
    columnas_numericas_vol = df_vol.select_dtypes(include=[np.number]).columns.tolist()
    col_vol = columnas_numericas_vol[0] if columnas_numericas_vol else columnas_vol[1]

# Identificar columna de Qin
col_qin = 'Qin' if 'Qin' in columnas_qin else df_qin.select_dtypes(include=[np.number]).columns[0]

print(f"\n   Columnas identificadas:")
print(f"   - Timestamp Volumen: '{col_timestamp_vol}'")
print(f"   - Volumen Total: '{col_vol}'")
print(f"   - Timestamp Qin: '{col_timestamp_qin}'")
print(f"   - Qin: '{col_qin}'")

# Convertir timestamps a datetime
df_vol['timestamp'] = pd.to_datetime(df_vol[col_timestamp_vol], utc=True)
df_qin['timestamp'] = pd.to_datetime(df_qin[col_timestamp_qin], utc=True)

# Renombrar columnas para consistencia
df_vol = df_vol.rename(columns={col_vol: 'Volumen'})
df_qin = df_qin.rename(columns={col_qin: 'Qin'})

# Ordenar por timestamp
df_vol = df_vol.sort_values('timestamp').reset_index(drop=True)
df_qin = df_qin.sort_values('timestamp').reset_index(drop=True)

# Merge por timestamp
df = pd.merge(df_vol[['timestamp', 'Volumen']], 
              df_qin[['timestamp', 'Qin']], 
              on='timestamp', 
              how='inner')

print(f"\n   Registros después del merge: {len(df):,}")

# ============================================================================
# 3. CALCULAR DEMANDA REAL
# ============================================================================

print("\n[3] CÁLCULO DE DEMANDA REAL:")
print("   Fórmula: Demanda = Qin - ΔVolumen")
print("   Donde:")
print("      - Qin: Producción que entra al sistema (m³/h)")
print("      - ΔVolumen: Cambio en almacenamiento = Vol[t] - Vol[t-1] (m³/h)")
print("      - Demanda: Consumo real de la ciudad (m³/h)")

# Calcular ΔVolumen (cambio horario en almacenamiento)
df['delta_vol'] = df['Volumen'].diff()

# Calcular demanda real: Qin - ΔVolumen
df['Demanda'] = df['Qin'] - df['delta_vol']

# Eliminar primera fila (tiene NaN en delta_vol)
df = df.dropna(subset=['delta_vol']).reset_index(drop=True)

print(f"\n   Estadísticas Demanda:")
print(f"   - Media: {df['Demanda'].mean():,.2f} m³/h")
print(f"   - Mediana: {df['Demanda'].median():,.2f} m³/h")
print(f"   - Std: {df['Demanda'].std():,.2f} m³/h")
print(f"   - Min: {df['Demanda'].min():,.2f} m³/h")
print(f"   - Max: {df['Demanda'].max():,.2f} m³/h")

# Filtrar valores razonables (eliminar outliers extremos)
# Asumiendo demanda típica entre 0 y 30,000 m³/h
df_valid = df[(df['Demanda'] >= 0) & (df['Demanda'] <= 30000)].copy()
print(f"\n   Registros válidos (0-30,000 m³/h): {len(df_valid):,} ({len(df_valid)/len(df)*100:.2f}%)")

# ============================================================================
# 4. EXTRAER DÍA DE LA SEMANA
# ============================================================================

print("\n[4] EXTRACCIÓN DÍA DE LA SEMANA:")

# Extraer día de la semana (0=Lunes, 6=Domingo)
df_valid['dia_semana_num'] = df_valid['timestamp'].dt.dayofweek
df_valid['dia_semana_nombre'] = df_valid['timestamp'].dt.day_name()

# Mapeo a español
dias_map = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miércoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}
df_valid['dia_semana_es'] = df_valid['dia_semana_nombre'].map(dias_map)

# Contar observaciones por día
print(f"\n   Observaciones por día:")
conteo_dias = df_valid['dia_semana_es'].value_counts().sort_index()
for dia in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']:
    if dia in conteo_dias.index:
        print(f"   - {dia}: {conteo_dias[dia]:,} horas")

# ============================================================================
# 5. CALCULAR DEMANDA PROMEDIO POR DÍA DE LA SEMANA
# ============================================================================

print("\n[5] DEMANDA PROMEDIO POR DÍA DE LA SEMANA:")

# Calcular estadísticas por día
stats_dia = df_valid.groupby('dia_semana_num').agg({
    'Demanda': ['count', 'mean', 'std', 'median', 'min', 'max']
}).reset_index()

# Aplanar columnas multi-nivel
stats_dia.columns = ['dia_num', 'count', 'mean', 'std', 'median', 'min', 'max']

# Agregar nombres en español
dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
stats_dia['dia_nombre'] = [dias_orden[i] for i in stats_dia['dia_num']]

print(f"\n   {'Día':^15} | {'Media (m³/h)':>14} | {'Mediana':>12} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
print("   " + "-" * 90)

for _, row in stats_dia.iterrows():
    print(f"   {row['dia_nombre']:^15} | {row['mean']:>14,.0f} | {row['median']:>12,.0f} | "
          f"{row['std']:>10,.0f} | {row['min']:>10,.0f} | {row['max']:>10,.0f}")

# Identificar día con mayor y menor demanda
dia_max = stats_dia.loc[stats_dia['mean'].idxmax()]
dia_min = stats_dia.loc[stats_dia['mean'].idxmin()]

print(f"\n   HALLAZGOS CLAVE:")
print(f"   - Día con MAYOR demanda: {dia_max['dia_nombre']} ({dia_max['mean']:,.0f} m³/h)")
print(f"   - Día con MENOR demanda: {dia_min['dia_nombre']} ({dia_min['mean']:,.0f} m³/h)")
print(f"   - Diferencia: {dia_max['mean'] - dia_min['mean']:,.0f} m³/h "
      f"({(dia_max['mean'] / dia_min['mean'] - 1) * 100:.2f}%)")

# Comparar días laborales vs fin de semana
laborales = df_valid[df_valid['dia_semana_num'] < 5]['Demanda']  # L-V
fin_semana = df_valid[df_valid['dia_semana_num'] >= 5]['Demanda']  # S-D

print(f"\n   COMPARACIÓN LABORALES vs FIN DE SEMANA:")
print(f"   - Días laborales (L-V): {laborales.mean():,.0f} m³/h (media)")
print(f"   - Fin de semana (S-D): {fin_semana.mean():,.0f} m³/h (media)")
print(f"   - Diferencia: {laborales.mean() - fin_semana.mean():,.0f} m³/h "
      f"({(laborales.mean() / fin_semana.mean() - 1) * 100:.2f}%)")

# ============================================================================
# 6. TEST ESTADÍSTICO: ANOVA
# ============================================================================

print("\n[6] TEST ANOVA (Diferencias entre días):")

# Preparar grupos por día
grupos_dia = [df_valid[df_valid['dia_semana_num'] == i]['Demanda'].values 
              for i in range(7)]

# ANOVA de una vía
f_statistic, p_value = stats.f_oneway(*grupos_dia)

print(f"   - Estadístico F: {f_statistic:.2f}")
print(f"   - P-valor: {p_value:.6f}")
print(f"\n   INTERPRETACIÓN:")
if p_value < 0.05:
    print(f"   ✓ Existen diferencias SIGNIFICATIVAS entre días (p < 0.05)")
    print(f"   El efecto \"día de la semana\" es estadísticamente relevante.")
else:
    print(f"   ✗ No se detectan diferencias significativas (p ≥ 0.05)")

# Test Mann-Whitney: Laborales vs Fin de semana
u_statistic, p_value_mw = stats.mannwhitneyu(laborales, fin_semana, alternative='two-sided')

print(f"\n   Test Mann-Whitney (Laborales vs Fin de Semana):")
print(f"   - Estadístico U: {u_statistic:,.0f}")
print(f"   - P-valor: {p_value_mw:.6f}")
if p_value_mw < 0.05:
    print(f"   ✓ Diferencia SIGNIFICATIVA entre laborales y fin de semana (p < 0.05)")

# ============================================================================
# 7. CREAR GRÁFICO DE BARRAS
# ============================================================================

print("\n[7] GENERANDO GRÁFICO DE CICLO SEMANAL...")

# Crear figura
fig, ax = plt.subplots(figsize=(14, 8))

# Colores: días laborales en azul, fin de semana en naranja
colores = ['steelblue'] * 5 + ['coral'] * 2

# Gráfico de barras con error bars (std)
bars = ax.bar(
    stats_dia['dia_nombre'],
    stats_dia['mean'],
    yerr=stats_dia['std'],
    color=colores,
    alpha=0.75,
    edgecolor='black',
    linewidth=1.5,
    capsize=8,
    error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6}
)

# Línea horizontal con media global
media_global = df_valid['Demanda'].mean()
ax.axhline(media_global, color='red', linestyle='--', linewidth=2.5, 
           label=f'Media global: {media_global:,.0f} m³/h', alpha=0.7)

# Anotaciones en barras
for i, (bar, row) in enumerate(zip(bars, stats_dia.itertuples())):
    height = bar.get_height()
    
    # Valor de demanda sobre cada barra
    ax.text(bar.get_x() + bar.get_width()/2., height + row.std + 200,
            f'{row.mean:,.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Marcar día con mayor y menor demanda
    if row.dia_nombre == dia_max['dia_nombre']:
        ax.annotate(
            'MÁXIMO',
            xy=(bar.get_x() + bar.get_width()/2., height),
            xytext=(0, -30),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    elif row.dia_nombre == dia_min['dia_nombre']:
        ax.annotate(
            'MÍNIMO',
            xy=(bar.get_x() + bar.get_width()/2., height),
            xytext=(0, -30),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7)
        )

# Sombreado de fondo para distinguir laborales vs fin de semana
ax.axvspan(-0.5, 4.5, alpha=0.10, color='blue', label='Días laborales')
ax.axvspan(4.5, 6.5, alpha=0.10, color='orange', label='Fin de semana')

# Configuración ejes y título
ax.set_xlabel('Día de la Semana', fontsize=14, fontweight='bold')
ax.set_ylabel('Demanda Promedio (m³/h)', fontsize=14, fontweight='bold')
ax.set_title(
    'Figura 4.1.3: Ciclo Semanal de Demanda de Agua Potable - Gran Valparaíso\n' +
    f'Demanda promedio por día (N = {len(df_valid):,} horas, {len(df_valid)//24:,} días)',
    fontsize=15,
    fontweight='bold',
    pad=20
)

# Formato eje Y con separador de miles
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, axis='y')

# Leyenda
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Cuadro de texto con resultados estadísticos
textstr = '\n'.join([
    'Test ANOVA:',
    f'F = {f_statistic:.2f}',
    f'p-valor = {p_value:.4f}',
    '',
    'Mann-Whitney (L-V vs S-D):',
    f'p-valor = {p_value_mw:.4f}',
    '',
    'Conclusión:',
    'Diferencia significativa' if p_value < 0.05 else 'No significativa',
    'entre días de la semana' if p_value < 0.05 else ''
])

props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
ax.text(
    0.02, 0.98,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=props,
    family='monospace'
)

# Ajustar layout
plt.tight_layout()

# ============================================================================
# 8. GUARDAR GRÁFICO
# ============================================================================

output_path = 'outputs/figures/ciclo_semanal_demanda.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n   ✓ Gráfico guardado en: {output_path}")

# Versión PDF
output_path_pdf = 'outputs/figures/ciclo_semanal_demanda.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
print(f"   ✓ Versión PDF guardada en: {output_path_pdf}")

plt.show()

# ============================================================================
# 9. RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN EJECUTIVO - CICLO SEMANAL:")
print("=" * 80)

diferencia_pct = (laborales.mean() / fin_semana.mean() - 1) * 100

print(f"""
1. PATRÓN SEMANAL IDENTIFICADO:
   - Día con MAYOR demanda: {dia_max['dia_nombre']} ({dia_max['mean']:,.0f} m³/h)
   - Día con MENOR demanda: {dia_min['dia_nombre']} ({dia_min['mean']:,.0f} m³/h)
   - Variación máxima: {(dia_max['mean'] / dia_min['mean'] - 1) * 100:.2f}%

2. EFECTO "FIN DE SEMANA":
   - Días laborales (L-V): {laborales.mean():,.0f} m³/h
   - Fin de semana (S-D): {fin_semana.mean():,.0f} m³/h
   - Reducción fin de semana: {abs(diferencia_pct):.2f}%
   
3. SIGNIFICANCIA ESTADÍSTICA:
   - ANOVA: F={f_statistic:.2f}, p={p_value:.4f} {'✓ SIGNIFICATIVO' if p_value < 0.05 else '✗ NO significativo'}
   - Mann-Whitney: p={p_value_mw:.4f} {'✓ SIGNIFICATIVO' if p_value_mw < 0.05 else '✗ NO significativo'}

4. INTERPRETACIÓN:
   {'✓ Existe un patrón semanal estadísticamente significativo' if p_value < 0.05 else '✗ No hay evidencia de patrón semanal consistente'}
   {'✓ Los días laborales presentan mayor demanda que fines de semana' if diferencia_pct > 0 else '✗ No hay diferencia clara laborales vs fin de semana'}
   {'✓ Actividad comercial/industrial aumenta consumo en días hábiles' if diferencia_pct > 0 else ''}
   {'✓ Domingo típicamente registra el mínimo semanal' if dia_min['dia_nombre'] == 'Domingo' else ''}

5. VARIABILIDAD:
   - Std promedio: {stats_dia['std'].mean():,.0f} m³/h
   - Coeficiente variación: {(stats_dia['std'].mean() / stats_dia['mean'].mean()) * 100:.1f}%
   - Nota: Alta variabilidad indica que eventos especiales/clima pueden 
     superar el patrón semanal típico

6. IMPLICACIONES OPERATIVAS:
   - Ajustar producción según día de la semana
   - Lunes-Viernes: mantener capacidad elevada
   - Fin de semana: oportunidad para mantenimiento preventivo
   - Domingo: ventana óptima para intervenciones programadas

7. PARA EL MODELO PREDICTIVO:
   - Feature "dia_semana" es relevante (incluir en modelo)
   - Considerar interacciones día×hora (picos distintos L-V vs S-D)
   - Encoding cíclico recomendado: dia_semana_sin, dia_semana_cos
""")

print("=" * 80)
print("Análisis completado exitosamente.")
print("=" * 80)
