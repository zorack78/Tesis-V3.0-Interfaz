"""
Explicación detallada: ¿Cómo se calculan los promedios estacionales?

Este script muestra paso a paso el cálculo de:
- Verano promedio: 12,730 m³/hr
- Invierno promedio: 11,077 m³/hr
- Diferencia: 14.9%
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("EXPLICACIÓN: CÁLCULO DE PROMEDIOS ESTACIONALES")
print("="*80)

# 1. Cargar datos con demanda
print("\n📂 PASO 1: CARGAR DATOS")
print("-" * 80)

data_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

print(f"✅ Archivo cargado: {data_path}")
print(f"   Total de registros: {len(df):,}")
print(f"   Período: {df['timestamp_utc'].min().date()} a {df['timestamp_utc'].max().date()}")

# 2. Clasificar por estación
print("\n📅 PASO 2: CLASIFICAR REGISTROS POR ESTACIÓN")
print("-" * 80)

def get_estacion(mes):
    """
    Clasifica el mes en estación del año (Hemisferio Sur - Chile)
    """
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:  # 9, 10, 11
        return 'Primavera'

# Extraer mes y asignar estación
df['mes'] = df['timestamp_utc'].dt.month
df['estacion'] = df['mes'].apply(get_estacion)

print("\n📋 Regla de clasificación (Hemisferio Sur):")
print("   • Verano:     Diciembre (12), Enero (1), Febrero (2)")
print("   • Otoño:      Marzo (3), Abril (4), Mayo (5)")
print("   • Invierno:   Junio (6), Julio (7), Agosto (8)")
print("   • Primavera:  Septiembre (9), Octubre (10), Noviembre (11)")

# Contar registros por estación
registros_por_estacion = df['estacion'].value_counts().sort_index()

print("\n📊 Registros por estación:")
for estacion in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
    n = registros_por_estacion.get(estacion, 0)
    pct = (n / len(df)) * 100
    print(f"   {estacion:.<15} {n:>6,} registros ({pct:>5.1f}%)")

# 3. Mostrar ejemplos de registros
print("\n📝 PASO 3: EJEMPLOS DE REGISTROS POR ESTACIÓN")
print("-" * 80)

# Mostrar 3 ejemplos de verano
print("\n☀️ Ejemplos de VERANO (diciembre, enero, febrero):")
ejemplos_verano = df[df['estacion'] == 'Verano'][['timestamp_utc', 'mes', 'Demanda_m3_hr']].head(3)
for idx, row in ejemplos_verano.iterrows():
    print(f"   • {row['timestamp_utc']} (mes {row['mes']:>2}) → {row['Demanda_m3_hr']:>10,.0f} m³/hr")

# Mostrar 3 ejemplos de invierno
print("\n❄️ Ejemplos de INVIERNO (junio, julio, agosto):")
ejemplos_invierno = df[df['estacion'] == 'Invierno'][['timestamp_utc', 'mes', 'Demanda_m3_hr']].head(3)
for idx, row in ejemplos_invierno.iterrows():
    print(f"   • {row['timestamp_utc']} (mes {row['mes']:>2}) → {row['Demanda_m3_hr']:>10,.0f} m³/hr")

# 4. Calcular promedios
print("\n🧮 PASO 4: CALCULAR PROMEDIO DE DEMANDA POR ESTACIÓN")
print("-" * 80)

print("\nFórmula: PROMEDIO = SUMA(todas las demandas) / CANTIDAD de registros")

# Calcular para cada estación
estadisticas = df.groupby('estacion')['Demanda_m3_hr'].agg([
    'count',
    'sum', 
    'mean',
    'std',
    'min',
    'max'
]).round(2)

print("\n📊 Cálculo detallado por estación:\n")

for estacion in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
    if estacion in estadisticas.index:
        n = estadisticas.loc[estacion, 'count']
        suma = estadisticas.loc[estacion, 'sum']
        promedio = estadisticas.loc[estacion, 'mean']
        std = estadisticas.loc[estacion, 'std']
        minimo = estadisticas.loc[estacion, 'min']
        maximo = estadisticas.loc[estacion, 'max']
        
        print(f"{estacion}:")
        print(f"   Cantidad de registros:  {n:>10,.0f}")
        print(f"   Suma total:            {suma:>11,.0f} m³")
        print(f"   Promedio:              {promedio:>11,.2f} m³/hr")
        print(f"   Desviación estándar:   {std:>11,.2f} m³/hr")
        print(f"   Mínimo:                {minimo:>11,.2f} m³/hr")
        print(f"   Máximo:                {maximo:>11,.2f} m³/hr")
        print(f"   Cálculo: {suma:,.0f} ÷ {n:,.0f} = {promedio:,.2f}\n")

# 5. Comparación Verano vs Invierno
print("\n🔥 PASO 5: COMPARACIÓN VERANO vs INVIERNO")
print("=" * 80)

verano_promedio = estadisticas.loc['Verano', 'mean']
invierno_promedio = estadisticas.loc['Invierno', 'mean']
diferencia_absoluta = verano_promedio - invierno_promedio
diferencia_porcentual = (diferencia_absoluta / invierno_promedio) * 100

print(f"\n☀️ VERANO:")
print(f"   Promedio: {verano_promedio:,.2f} m³/hr")
print(f"   N registros: {estadisticas.loc['Verano', 'count']:,.0f}")

print(f"\n❄️ INVIERNO:")
print(f"   Promedio: {invierno_promedio:,.2f} m³/hr")
print(f"   N registros: {estadisticas.loc['Invierno', 'count']:,.0f}")

print(f"\n📊 DIFERENCIA:")
print(f"   Absoluta: {diferencia_absoluta:,.2f} m³/hr")
print(f"   Porcentual: {diferencia_porcentual:.2f}%")

print(f"\n🧮 Cálculo del porcentaje:")
print(f"   Fórmula: ((Verano - Invierno) / Invierno) × 100")
print(f"   = (({verano_promedio:,.2f} - {invierno_promedio:,.2f}) / {invierno_promedio:,.2f}) × 100")
print(f"   = ({diferencia_absoluta:,.2f} / {invierno_promedio:,.2f}) × 100")
print(f"   = {diferencia_porcentual:.2f}%")

# Verificación
print(f"\n✅ RESULTADO:")
print(f"   Verano es {diferencia_porcentual:.1f}% MAYOR que Invierno")
print(f"   {'✅ CORRECTO: Verano > Invierno' if verano_promedio > invierno_promedio else '❌ ERROR: Invierno > Verano'}")

# 6. Visualizar distribución
print("\n📈 PASO 6: DISTRIBUCIÓN DE DATOS")
print("-" * 80)

print("\n🔍 Verificación de datos por mes:")
print(f"\n{'Mes':>12} | {'Estación':>11} | {'N Registros':>12} | {'Demanda Prom':>14}")
print("-" * 60)

for mes in range(1, 13):
    datos_mes = df[df['mes'] == mes]
    if len(datos_mes) > 0:
        estacion = datos_mes['estacion'].iloc[0]
        n = len(datos_mes)
        promedio = datos_mes['Demanda_m3_hr'].mean()
        
        nombre_mes = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][mes]
        
        print(f"{nombre_mes:>12} | {estacion:>11} | {n:>12,} | {promedio:>11,.0f} m³/hr")

# 7. Estadísticas adicionales
print("\n📊 PASO 7: ESTADÍSTICAS ADICIONALES")
print("-" * 80)

print("\n🌡️ Análisis de variabilidad:")
coef_var_verano = (estadisticas.loc['Verano', 'std'] / estadisticas.loc['Verano', 'mean']) * 100
coef_var_invierno = (estadisticas.loc['Invierno', 'std'] / estadisticas.loc['Invierno', 'mean']) * 100

print(f"\nCoeficiente de Variación (CV = std/mean × 100):")
print(f"   Verano:   CV = {coef_var_verano:.1f}%  {'(más variable)' if coef_var_verano > coef_var_invierno else '(menos variable)'}")
print(f"   Invierno: CV = {coef_var_invierno:.1f}%  {'(más variable)' if coef_var_invierno > coef_var_verano else '(menos variable)'}")

# 8. Conclusión
print("\n" + "="*80)
print("💡 CONCLUSIÓN")
print("="*80)

print(f"""
Los valores reportados provienen de:

1️⃣ DATOS REALES:
   • {len(df):,} registros de demanda horaria
   • Período: 2024-2025 (21 meses)
   • Verano: {int(estadisticas.loc['Verano', 'count']):,} registros
   • Invierno: {int(estadisticas.loc['Invierno', 'count']):,} registros

2️⃣ MÉTODO DE CÁLCULO:
   • Clasificación por mes → estación
   • Promedio simple: SUMA(demandas) / N_registros
   • Sin ponderaciones ni ajustes
   • Incluye todas las horas del día (0-23)

3️⃣ RESULTADOS:
   • Verano:  {verano_promedio:,.0f} m³/hr (promedio de {int(estadisticas.loc['Verano', 'count']):,} registros)
   • Invierno: {invierno_promedio:,.0f} m³/hr (promedio de {int(estadisticas.loc['Invierno', 'count']):,} registros)
   • Diferencia: +{diferencia_porcentual:.1f}%

4️⃣ INTERPRETACIÓN:
   • En verano se consume {diferencia_porcentual:.1f}% MÁS agua que en invierno
   • Esto representa {diferencia_absoluta:,.0f} m³/hr adicionales
   • Por día: {diferencia_absoluta * 24:,.0f} m³ más en verano
   • Por mes (30 días): {diferencia_absoluta * 24 * 30:,.0f} m³ más en verano

5️⃣ VALIDACIÓN:
   {'✅ Los datos muestran el patrón esperado: Verano > Invierno' if verano_promedio > invierno_promedio else '❌ Los datos NO muestran el patrón esperado'}
   {'✅ Consistente con mayor consumo en temperaturas altas' if verano_promedio > invierno_promedio else ''}

📌 Estos NO son valores teóricos ni estimados.
📌 Son el comportamiento REAL OBSERVADO del sistema.
📌 El modelo aprende este patrón de los datos históricos.
""")

print("="*80)
print("FIN DE LA EXPLICACIÓN")
print("="*80)
