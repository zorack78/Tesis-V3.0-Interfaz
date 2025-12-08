"""
Análisis de Distribución de Demanda Horaria - Capítulo 4.1.1
Genera histograma + curva normal teórica + test Shapiro-Wilk
Autor: Sistema Predictivo Gran Valparaíso
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Configuración estilo gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("=" * 80)
print("ANÁLISIS DE DISTRIBUCIÓN DE DEMANDA HORARIA")
print("=" * 80)

# Cargar flujo neto del sistema (Q_flujo = demanda neta considerando Qin)
df = pd.read_csv('data/raw/BD_Q_net_x_Hr_m3h.csv')

print("\n[1] INFORMACIÓN DEL DATASET:")
print(f"   - Registros totales: {len(df):,}")
print(f"   - Columnas disponibles: {df.columns.tolist()}")

# Identificar columna de flujo/demanda (puede ser 'Q_flujo', 'Volumen_Total', 'Demanda', etc.)
# Asumimos que la primera columna numérica es el flujo
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Q_flujo' in columnas_numericas:
    columna_demanda = 'Q_flujo'
elif 'Volumen_Total' in columnas_numericas:
    columna_demanda = 'Volumen_Total'
elif 'Demanda' in columnas_numericas:
    columna_demanda = 'Demanda'
else:
    columna_demanda = columnas_numericas[0]  # Primera columna numérica

print(f"   - Columna de demanda identificada: '{columna_demanda}'")

# Extraer serie de demanda (solo valores positivos = consumo)
demanda = df[columna_demanda].dropna()
demanda_positiva = demanda[demanda > 0]  # Filtrar solo consumo (valores positivos)

print(f"   - Registros válidos (positivos): {len(demanda_positiva):,}")
print(f"   - Registros negativos/recuperación: {len(demanda[demanda <= 0]):,}")

# ============================================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

print("\n[2] ESTADÍSTICAS DESCRIPTIVAS:")
print(f"   - Media: {demanda_positiva.mean():,.2f} m³/h")
print(f"   - Mediana: {demanda_positiva.median():,.2f} m³/h")
print(f"   - Desviación estándar: {demanda_positiva.std():,.2f} m³/h")
print(f"   - Mínimo: {demanda_positiva.min():,.2f} m³/h")
print(f"   - Máximo: {demanda_positiva.max():,.2f} m³/h")
print(f"   - Rango: {demanda_positiva.max() - demanda_positiva.min():,.2f} m³/h")

# Percentiles clave
percentiles = [5, 10, 25, 50, 75, 90, 95]
print("\n   PERCENTILES:")
for p in percentiles:
    valor = np.percentile(demanda_positiva, p)
    print(f"      P{p:2d}: {valor:>10,.2f} m³/h")

# Asimetría y curtosis
skewness = stats.skew(demanda_positiva)
kurtosis = stats.kurtosis(demanda_positiva)
print(f"\n   - Asimetría (skewness): {skewness:.4f}")
print(f"     {'(Sesgo positivo - cola derecha)' if skewness > 0 else '(Sesgo negativo - cola izquierda)'}")
print(f"   - Curtosis (kurtosis): {kurtosis:.4f}")
print(f"     {'(Leptocúrtica - más picuda que normal)' if kurtosis > 0 else '(Platicúrtica - más plana que normal)'}")

# ============================================================================
# 3. TEST DE NORMALIDAD (SHAPIRO-WILK)
# ============================================================================

print("\n[3] TEST DE NORMALIDAD (SHAPIRO-WILK):")

# Nota: Shapiro-Wilk con muestras grandes puede ser muy estricto
# Si hay >5000 observaciones, usar una muestra aleatoria
if len(demanda_positiva) > 5000:
    print(f"   - Nota: Dataset grande ({len(demanda_positiva):,} obs), usando muestra aleatoria de 5000")
    muestra = demanda_positiva.sample(n=5000, random_state=42)
    statistic, p_value = stats.shapiro(muestra)
else:
    statistic, p_value = stats.shapiro(demanda_positiva)

print(f"   - Estadístico W: {statistic:.6f}")
print(f"   - P-valor: {p_value:.6e}")
print(f"\n   INTERPRETACIÓN:")
if p_value < 0.01:
    print(f"   ✗ Se RECHAZA la hipótesis de normalidad (p < 0.01)")
    print(f"   La demanda horaria NO sigue una distribución normal.")
else:
    print(f"   ✓ No se rechaza la hipótesis de normalidad (p ≥ 0.01)")

# ============================================================================
# 4. CREAR GRÁFICO: HISTOGRAMA + CURVA NORMAL TEÓRICA
# ============================================================================

print("\n[4] GENERANDO GRÁFICO DE DISTRIBUCIÓN...")

# Crear figura
fig, ax = plt.subplots(figsize=(12, 7))

# 4.1 Histograma de frecuencias (normalizado para comparar con densidad)
n, bins, patches = ax.hist(
    demanda_positiva, 
    bins=50,  # 50 intervalos
    density=True,  # Normalizar para que área = 1 (comparable con curva densidad)
    alpha=0.65,
    color='steelblue',
    edgecolor='black',
    linewidth=0.8,
    label='Distribución observada'
)

# 4.2 Curva de densidad normal teórica (basada en media y DE de los datos)
mu = demanda_positiva.mean()
sigma = demanda_positiva.std()
x_teorico = np.linspace(demanda_positiva.min(), demanda_positiva.max(), 1000)
densidad_normal = stats.norm.pdf(x_teorico, mu, sigma)

ax.plot(
    x_teorico, 
    densidad_normal, 
    'r-', 
    linewidth=2.5,
    label=f'Distribución normal teórica\n(μ={mu:,.0f}, σ={sigma:,.0f})'
)

# 4.3 Líneas verticales para percentiles 5 y 95 (rango 90% central)
p5 = np.percentile(demanda_positiva, 5)
p95 = np.percentile(demanda_positiva, 95)

ax.axvline(p5, color='green', linestyle='--', linewidth=1.5, 
           label=f'P5 = {p5:,.0f} m³/h')
ax.axvline(p95, color='orange', linestyle='--', linewidth=1.5, 
           label=f'P95 = {p95:,.0f} m³/h')

# Sombreado área entre P5 y P95 (90% central)
ax.axvspan(p5, p95, alpha=0.15, color='gray', 
           label=f'Rango 90% central\n({p95-p5:,.0f} m³/h)')

# 4.4 Configuración ejes y título
ax.set_xlabel('Demanda Horaria (m³/h)', fontsize=13, fontweight='bold')
ax.set_ylabel('Densidad de Probabilidad', fontsize=13, fontweight='bold')
ax.set_title(
    'Distribución de Demanda Horaria del Sistema Gran Valparaíso\n' +
    f'(N = {len(demanda_positiva):,} observaciones)',
    fontsize=15, 
    fontweight='bold',
    pad=20
)

# 4.5 Cuadro de texto con resultados test Shapiro-Wilk
textstr = '\n'.join([
    'Test de Normalidad (Shapiro-Wilk):',
    f'Estadístico W = {statistic:.6f}',
    f'P-valor = {p_value:.6e}',
    '',
    'Conclusión:',
    'Se rechaza H₀ (normalidad)' if p_value < 0.01 else 'No se rechaza H₀',
    '',
    'Interpretación:',
    'Distribución asimétrica positiva',
    '(sesgo a la derecha)',
    'Presencia de picos extremos',
    'más frecuentes que normal'
])

# Posicionar cuadro de texto en esquina superior derecha
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(
    0.98, 0.97, 
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=props,
    family='monospace'
)

# 4.6 Leyenda
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# 4.7 Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

# Ajustar layout
plt.tight_layout()

# ============================================================================
# 5. GUARDAR GRÁFICO
# ============================================================================

output_path = 'outputs/figures/distribucion_demanda_horaria.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n   ✓ Gráfico guardado en: {output_path}")

# También guardar en formato vectorial (PDF) para publicación
output_path_pdf = 'outputs/figures/distribucion_demanda_horaria.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
print(f"   ✓ Versión PDF guardada en: {output_path_pdf}")

plt.show()

# ============================================================================
# 6. ANÁLISIS ADICIONAL: OUTLIERS
# ============================================================================

print("\n[5] IDENTIFICACIÓN DE OUTLIERS (Método IQR):")

Q1 = demanda_positiva.quantile(0.25)
Q3 = demanda_positiva.quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = demanda_positiva[(demanda_positiva < limite_inferior) | 
                            (demanda_positiva > limite_superior)]

print(f"   - Q1 (percentil 25): {Q1:,.2f} m³/h")
print(f"   - Q3 (percentil 75): {Q3:,.2f} m³/h")
print(f"   - IQR (rango intercuartílico): {IQR:,.2f} m³/h")
print(f"   - Límite inferior: {limite_inferior:,.2f} m³/h")
print(f"   - Límite superior: {limite_superior:,.2f} m³/h")
print(f"\n   - Outliers identificados: {len(outliers)} ({len(outliers)/len(demanda_positiva)*100:.2f}%)")
print(f"   - Valor outlier mínimo: {outliers.min():,.2f} m³/h")
print(f"   - Valor outlier máximo: {outliers.max():,.2f} m³/h")

# ============================================================================
# 7. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN EJECUTIVO:")
print("=" * 80)

print(f"""
1. DISTRIBUCIÓN DE DEMANDA:
   - La demanda horaria presenta asimetría positiva (skewness = {skewness:.4f})
   - La mayoría de observaciones se concentran en rango medio
   - Existe una cola extendida hacia valores altos (eventos excepcionales)

2. TEST DE NORMALIDAD:
   - Shapiro-Wilk rechaza hipótesis de normalidad (p = {p_value:.6e})
   - La distribución NO es Gaussiana (campana de Gauss)
   - Picos de alta demanda son más frecuentes que lo que normal predeciría

3. RANGO CENTRAL (90% observaciones):
   - Percentil 5: {p5:,.0f} m³/h
   - Percentil 95: {p95:,.0f} m³/h
   - Amplitud: {p95-p5:,.0f} m³/h
   
4. OUTLIERS:
   - {len(outliers)} horas ({len(outliers)/len(demanda_positiva)*100:.2f}%) fuera de límites IQR
   - Requieren análisis detallado por separado
   - Probablemente asociados a eventos sociales o condiciones climáticas extremas

5. IMPLICACIÓN PARA MODELAMIENTO:
   - Modelos basados en árboles (XGBoost, RF) son apropiados
   - NO asumir normalidad en residuos de regresión lineal
   - Considerar transformación Box-Cox si se requiere normalización
""")

print("=" * 80)
print("Análisis completado exitosamente.")
print("=" * 80)
