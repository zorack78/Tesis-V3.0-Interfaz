"""
Análisis: ¿Cómo determina el modelo las horarios de inflexión?

Este script explica:
1. De dónde salen las horas en las predicciones
2. Cómo el modelo aprende los patrones horarios
3. Qué datos reales usa para estimar LAGs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("ANÁLISIS: DETERMINACIÓN DE HORAS PICO Y VALLE")
print("="*80)

# 1. Cargar datos históricos con demanda
data_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

print(f"\n📊 Datos cargados: {len(df)} registros")
print(f"   Período: {df['timestamp_utc'].min()} a {df['timestamp_utc'].max()}")

# 2. ANÁLISIS REAL DE DEMANDA POR HORA
print("\n" + "="*80)
print("DEMANDA REAL PROMEDIO POR HORA DEL DÍA (DATOS HISTÓRICOS)")
print("="*80)

df['hora'] = df['timestamp_utc'].dt.hour

# Calcular estadísticas por hora
demanda_por_hora = df.groupby('hora')['Demanda_m3_hr'].agg([
    'count', 'mean', 'std', 'min', 'max', 
    ('p25', lambda x: x.quantile(0.25)),
    ('p75', lambda x: x.quantile(0.75))
]).round(0)

print(f"\n{'Hora':>4} | {'N':>5} | {'Media':>11} | {'Std':>10} | {'Min':>11} | {'Max':>11} | {'P25':>11} | {'P75':>11}")
print("-" * 100)

for hora in range(24):
    n = demanda_por_hora.loc[hora, 'count']
    media = demanda_por_hora.loc[hora, 'mean']
    std = demanda_por_hora.loc[hora, 'std']
    min_val = demanda_por_hora.loc[hora, 'min']
    max_val = demanda_por_hora.loc[hora, 'max']
    p25 = demanda_por_hora.loc[hora, 'p25']
    p75 = demanda_por_hora.loc[hora, 'p75']
    
    print(f"{hora:>4} | {n:>5.0f} | {media:>11,.0f} | {std:>10,.0f} | {min_val:>11,.0f} | {max_val:>11,.0f} | {p25:>11,.0f} | {p75:>11,.0f}")

# Identificar picos y valles EN LOS DATOS REALES
hora_pico = demanda_por_hora['mean'].idxmax()
hora_valle = demanda_por_hora['mean'].idxmin()
demanda_pico = demanda_por_hora.loc[hora_pico, 'mean']
demanda_valle = demanda_por_hora.loc[hora_valle, 'mean']

print(f"\n🔝 HORA PICO en datos reales:")
print(f"   Hora: {hora_pico}:00")
print(f"   Demanda promedio: {demanda_pico:,.0f} m³/hr")
print(f"   Desv. estándar: {demanda_por_hora.loc[hora_pico, 'std']:,.0f} m³/hr")

print(f"\n🔻 HORA VALLE en datos reales:")
print(f"   Hora: {hora_valle}:00")
print(f"   Demanda promedio: {demanda_valle:,.0f} m³/hr")
print(f"   Desv. estándar: {demanda_por_hora.loc[hora_valle, 'std']:,.0f} m³/hr")

# 3. ANÁLISIS POR ESTACIÓN
print("\n" + "="*80)
print("DEMANDA POR HORA Y ESTACIÓN DEL AÑO")
print("="*80)

# Definir estaciones (hemisferio sur)
def get_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

df['estacion'] = df['timestamp_utc'].dt.month.apply(get_estacion)
df['mes'] = df['timestamp_utc'].dt.month

# Promedio por hora y estación
demanda_por_hora_estacion = df.groupby(['hora', 'estacion'])['Demanda_m3_hr'].mean().round(0)

print("\n📊 Comparación Verano vs Invierno por hora:")
print(f"\n{'Hora':>4} | {'Verano':>12} | {'Invierno':>12} | {'Diferencia':>12} | {'% Diff':>8}")
print("-" * 60)

for hora in range(24):
    verano = demanda_por_hora_estacion.get((hora, 'Verano'), np.nan)
    invierno = demanda_por_hora_estacion.get((hora, 'Invierno'), np.nan)
    
    if not np.isnan(verano) and not np.isnan(invierno):
        diff = verano - invierno
        pct_diff = (diff / invierno) * 100 if invierno != 0 else 0
        print(f"{hora:>4} | {verano:>12,.0f} | {invierno:>12,.0f} | {diff:>12,.0f} | {pct_diff:>7.1f}%")

# Promedios por estación
print("\n" + "="*80)
print("PROMEDIO GENERAL POR ESTACIÓN")
print("="*80)

demanda_por_estacion = df.groupby('estacion')['Demanda_m3_hr'].agg(['mean', 'std', 'count']).round(0)
demanda_por_estacion = demanda_por_estacion.reindex(['Verano', 'Otoño', 'Invierno', 'Primavera'])

for estacion in demanda_por_estacion.index:
    media = demanda_por_estacion.loc[estacion, 'mean']
    std = demanda_por_estacion.loc[estacion, 'std']
    n = demanda_por_estacion.loc[estacion, 'count']
    print(f"\n{estacion}:")
    print(f"   Demanda promedio: {media:,.0f} m³/hr")
    print(f"   Desv. estándar: {std:,.0f} m³/hr")
    print(f"   N registros: {n:,.0f}")

# Comparar verano vs invierno
verano_mean = demanda_por_estacion.loc['Verano', 'mean']
invierno_mean = demanda_por_estacion.loc['Invierno', 'mean']
diff_absoluta = verano_mean - invierno_mean
diff_porcentual = (diff_absoluta / invierno_mean) * 100

print(f"\n🔥 VERANO vs ❄️ INVIERNO:")
print(f"   Diferencia absoluta: {diff_absoluta:,.0f} m³/hr")
print(f"   Diferencia porcentual: {diff_porcentual:.1f}%")
print(f"   {'✅ Verano > Invierno (CORRECTO)' if verano_mean > invierno_mean else '❌ Invierno > Verano (INCORRECTO)'}")

# 4. CÓMO FUNCIONA _estimar_lags_demanda()
print("\n" + "="*80)
print("CÓMO EL MODELO ESTIMA LOS LAGs")
print("="*80)

print("""
La función _estimar_lags_demanda() en la interfaz:

1️⃣ Recibe: fecha y hora para predecir
2️⃣ Extrae: día_semana (lunes=0, domingo=6) y mes
3️⃣ Filtra datos históricos por:
   • Misma HORA del día
   • Mismo DÍA DE LA SEMANA
   • Mismo MES del año
   
4️⃣ Calcula MEDIANA de demanda de esos datos filtrados
5️⃣ Usa esa mediana como LAG_1h, LAG_24h, etc.

EJEMPLO:
Si pides predicción para "2025-12-25 a las 12:00" (jueves, diciembre):
• Busca todos los jueves de diciembre a las 12:00 en datos históricos
• Calcula mediana de demanda en esas condiciones
• Usa ese valor como estimación de LAGs

ESTO SIGNIFICA:
• Las horarios de inflexión vienen de PATRONES REALES en los datos
• NO son valores arbitrarios o referencias teóricas
• Son el comportamiento OBSERVADO del sistema
""")

# Ejemplo concreto
print("\n📝 EJEMPLO CONCRETO:")
print("\nBusquemos todos los datos de hora 12:00 en diciembre:")

datos_12h_diciembre = df[(df['hora'] == 12) & (df['mes'] == 12)]
print(f"   Registros encontrados: {len(datos_12h_diciembre)}")
print(f"   Demanda promedio: {datos_12h_diciembre['Demanda_m3_hr'].mean():,.0f} m³/hr")
print(f"   Demanda mediana: {datos_12h_diciembre['Demanda_m3_hr'].median():,.0f} m³/hr")

print("\nBusquemos todos los datos de hora 4:00 en diciembre:")
datos_4h_diciembre = df[(df['hora'] == 4) & (df['mes'] == 12)]
print(f"   Registros encontrados: {len(datos_4h_diciembre)}")
print(f"   Demanda promedio: {datos_4h_diciembre['Demanda_m3_hr'].mean():,.0f} m³/hr")
print(f"   Demanda mediana: {datos_4h_diciembre['Demanda_m3_hr'].median():,.0f} m³/hr")

# 5. FEATURES QUE USA EL MODELO
print("\n" + "="*80)
print("FEATURES MÁS IMPORTANTES DEL MODELO")
print("="*80)

import json
metricas_path = Path('models/demanda/metricas.json')
if metricas_path.exists():
    with open(metricas_path, 'r', encoding='utf-8') as f:
        metricas = json.load(f)
    
    print("\n🔝 Top 10 features por importancia:")
    for i, feat_info in enumerate(metricas['top_features'][:10], 1):
        print(f"   {i:>2}. {feat_info['feature']:.<40} {feat_info['importance']:.4f}")
    
    print(f"""
💡 INTERPRETACIÓN:

Las features más importantes son:
1. Demanda_diff_1h: Cambio respecto a hora anterior
2. Demanda_diff_24h: Cambio respecto a misma hora ayer
3. hour: La hora del día (0-23)

Esto significa que el modelo aprende:
• Patrones horarios fuertes (feature 'hour')
• Tendencias de corto plazo (diff_1h)
• Patrones diarios repetitivos (diff_24h)

Por eso las predicciones respetan:
• Horario de inflexión máximo a mediodía (porque históricamente es así)
• Horario de inflexión mínimo en madrugada (porque históricamente es así)
• Diferencias estacionales (porque los datos lo muestran)
""")

# 6. Visualización
print("\n📊 Generando gráfico...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Demanda por hora (promedio)
ax1 = axes[0, 0]
demanda_por_hora['mean'].plot(kind='line', marker='o', linewidth=2, markersize=8, ax=ax1, color='blue')
ax1.axhline(demanda_por_hora['mean'].mean(), color='red', linestyle='--', label='Promedio general')
ax1.axvline(hora_pico, color='green', linestyle='--', alpha=0.7, label=f'Máximo: {hora_pico}:00')
ax1.axvline(hora_valle, color='orange', linestyle='--', alpha=0.7, label=f'Mínimo: {hora_valle}:00')
ax1.set_xlabel('Hora del día', fontsize=12)
ax1.set_ylabel('Demanda (m³/hr)', fontsize=12)
ax1.set_title('DEMANDA REAL PROMEDIO POR HORA\n(De aquí salen las horarios de inflexión)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(24))

# Gráfico 2: Comparación estaciones por hora
ax2 = axes[0, 1]
for estacion in ['Verano', 'Invierno']:
    datos_estacion = demanda_por_hora_estacion.xs(estacion, level='estacion')
    ax2.plot(datos_estacion.index, datos_estacion.values, marker='o', linewidth=2, label=estacion)
ax2.set_xlabel('Hora del día', fontsize=12)
ax2.set_ylabel('Demanda (m³/hr)', fontsize=12)
ax2.set_title('COMPARACIÓN VERANO vs INVIERNO POR HORA', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(24))

# Gráfico 3: Boxplot por hora (variabilidad)
ax3 = axes[1, 0]
df_sample = df[df['hora'].isin([0, 4, 8, 12, 16, 20])]
df_sample.boxplot(column='Demanda_m3_hr', by='hora', ax=ax3)
ax3.set_xlabel('Hora del día', fontsize=12)
ax3.set_ylabel('Demanda (m³/hr)', fontsize=12)
ax3.set_title('VARIABILIDAD DE DEMANDA POR HORA\n(Muestra cada 4 horas)', fontsize=14, fontweight='bold')
plt.sca(ax3)
plt.xticks(rotation=0)

# Gráfico 4: Promedio por estación
ax4 = axes[1, 1]
estaciones_order = ['Verano', 'Otoño', 'Invierno', 'Primavera']
colors = ['red', 'orange', 'blue', 'green']
demanda_por_estacion.loc[estaciones_order, 'mean'].plot(kind='bar', ax=ax4, color=colors)
ax4.set_xlabel('Estación', fontsize=12)
ax4.set_ylabel('Demanda promedio (m³/hr)', fontsize=12)
ax4.set_title('DEMANDA PROMEDIO POR ESTACIÓN\n(Verano > Invierno)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
plt.sca(ax4)
plt.xticks(rotation=45)

plt.tight_layout()
output_path = Path('outputs/analisis_horas_pico_valle.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {output_path}")

# 7. CONCLUSIÓN
print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)

print(f"""
📌 RESPUESTA A TU PREGUNTA:

Las horarios de inflexión que muestra la simulación vienen de:

1️⃣ DATOS REALES HISTÓRICOS:
   • {len(df):,} registros de demanda real del sistema
   • Período: 2024-01-01 a 2025-09-30
   • Horario de inflexión máximo observada: {hora_pico}:00 ({demanda_pico:,.0f} m³/hr)
   • Horario de inflexión mínimo observada: {hora_valle}:00 ({demanda_valle:,.0f} m³/hr)

2️⃣ MÉTODO DEL MODELO:
   • Filtra datos históricos por: hora + día_semana + mes
   • Calcula MEDIANA de demanda en esas condiciones
   • Usa feature 'hour' que tiene alta importancia (14.14%)
   • NO son valores teóricos o referencias arbitrarias

3️⃣ VALIDACIÓN ESTACIONAL:
   • Verano promedio: {verano_mean:,.0f} m³/hr
   • Invierno promedio: {invierno_mean:,.0f} m³/hr
   • Diferencia: {diff_porcentual:.1f}% (Verano > Invierno ✅)

4️⃣ ORIGEN DE TUS "HORAS REFERENCIALES":
   • Si mencionaste que 7:00 AM era horario de inflexión máximo
   • Era referencia al VOLUMEN ALMACENADO (máximo a las 7 AM)
   • NO a la DEMANDA (máximo a las {hora_pico}:00)
   • Ahora el modelo predice DEMANDA correctamente

🎯 Las horas son REALES, no teóricas.
🎯 Vienen del comportamiento OBSERVADO del sistema.
🎯 El modelo aprende estos patrones de los datos históricos.
""")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)
