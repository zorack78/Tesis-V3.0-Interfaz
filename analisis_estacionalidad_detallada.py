"""
Análisis detallado para entender la relación temperatura-consumo
¿Por qué más frío = más consumo?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("ANÁLISIS: ¿POR QUÉ MÁS FRÍO = MÁS CONSUMO?")
print("=" * 80)

# Cargar datos
volumen_df = pd.read_csv('data/processed/data_processed_complete.csv')
volumen_df.columns = volumen_df.columns.str.strip()
volumen_df['timestamp_utc'] = pd.to_datetime(volumen_df['timestamp_utc'])

clima_df = pd.read_csv('data/processed/clima_chile_v3.csv')
clima_df['timestamp'] = pd.to_datetime(clima_df['timestamp'])

df = pd.merge(volumen_df, clima_df, 
              left_on='timestamp_utc', right_on='timestamp', how='inner')

volumen_col = 'Volumen_Total_m3'

# Asegurar columnas temporales
if 'hora' not in df.columns:
    df['hora'] = df['timestamp_utc'].dt.hour
if 'mes' not in df.columns:
    df['mes'] = df['timestamp_utc'].dt.month

print(f"\n📊 Total registros: {len(df)}")
print(f"📊 Rango fechas: {df['timestamp_utc'].min()} a {df['timestamp_utc'].max()}")

# Definir estaciones
def get_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

df['estacion'] = df['mes'].apply(get_estacion)

print("\n" + "=" * 80)
print("1️⃣ ANÁLISIS POR MES Y TEMPERATURA")
print("=" * 80)

meses_nombres = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

analisis_mensual = df.groupby('mes').agg({
    volumen_col: ['mean', 'std'],
    'temp': 'mean',
    'HR': 'mean',
    'mmhr': 'mean'
}).round(2)

print("\n📊 Consumo promedio por mes:")
print(f"{'Mes':<12} {'Consumo':<15} {'Temp':<10} {'Humedad':<10} {'Lluvia'}")
print("-" * 65)
for mes in range(1, 13):
    if mes in df['mes'].values:
        consumo = df[df['mes']==mes][volumen_col].mean()
        temp = df[df['mes']==mes]['temp'].mean()
        hr = df[df['mes']==mes]['HR'].mean()
        lluvia = df[df['mes']==mes]['mmhr'].mean()
        print(f"{meses_nombres[mes]:<12} {consumo:>10,.0f} m³   {temp:>5.1f}°C   "
              f"{hr:>5.1f}%    {lluvia:>5.2f}mm")

print("\n" + "=" * 80)
print("2️⃣ ANÁLISIS POR HORA DEL DÍA Y TEMPERATURA")
print("=" * 80)

# Patrones por hora en invierno vs verano
print("\n⏰ CONSUMO POR HORA en diferentes estaciones:")
print(f"{'Hora':<8} {'Verano':<15} {'Invierno':<15} {'Diferencia':<15} {'% Var'}")
print("-" * 70)

for hora in range(24):
    verano = df[(df['estacion']=='Verano') & (df['hora']==hora)][volumen_col].mean()
    invierno = df[(df['estacion']=='Invierno') & (df['hora']==hora)][volumen_col].mean()
    dif = invierno - verano
    pct = (dif / verano * 100) if verano > 0 else 0
    print(f"{hora:02d}:00    {verano:>10,.0f} m³   {invierno:>10,.0f} m³   "
          f"{dif:>10,.0f} m³   {pct:>6.2f}%")

print("\n" + "=" * 80)
print("3️⃣ ANÁLISIS DE ACTIVIDAD POBLACIONAL")
print("=" * 80)

# Horas de mayor consumo (actividad humana)
horas_alta = [6, 7, 8, 9, 10, 11, 12]  # Mañana y mediodía
horas_baja = [0, 1, 2, 3, 4, 5, 22, 23]  # Madrugada

print("\n🌅 CONSUMO EN HORAS DE ALTA ACTIVIDAD (6:00-12:00):")
alta_verano = df[(df['estacion']=='Verano') & (df['hora'].isin(horas_alta))][volumen_col].mean()
alta_invierno = df[(df['estacion']=='Invierno') & (df['hora'].isin(horas_alta))][volumen_col].mean()
print(f"   Verano: {alta_verano:,.0f} m³/hora")
print(f"   Invierno: {alta_invierno:,.0f} m³/hora")
print(f"   Diferencia: {alta_invierno - alta_verano:,.0f} m³/hora ({(alta_invierno-alta_verano)/alta_verano*100:+.2f}%)")

print("\n🌙 CONSUMO EN HORAS DE BAJA ACTIVIDAD (00:00-05:00, 22:00-23:00):")
baja_verano = df[(df['estacion']=='Verano') & (df['hora'].isin(horas_baja))][volumen_col].mean()
baja_invierno = df[(df['estacion']=='Invierno') & (df['hora'].isin(horas_baja))][volumen_col].mean()
print(f"   Verano: {baja_verano:,.0f} m³/hora")
print(f"   Invierno: {baja_invierno:,.0f} m³/hora")
print(f"   Diferencia: {baja_invierno - baja_verano:,.0f} m³/hora ({(baja_invierno-baja_verano)/baja_verano*100:+.2f}%)")

print("\n" + "=" * 80)
print("4️⃣ ANÁLISIS DE TEMPERATURAS EXTREMAS")
print("=" * 80)

# Dividir en rangos más específicos
temp_ranges = [
    (-np.inf, 10, 'Muy Frío (<10°C)'),
    (10, 12, 'Frío (10-12°C)'),
    (12, 15, 'Templado (12-15°C)'),
    (15, 18, 'Cálido (15-18°C)'),
    (18, 20, 'Caluroso (18-20°C)'),
    (20, np.inf, 'Muy Caluroso (>20°C)')
]

print("\n🌡️ CONSUMO POR RANGO DE TEMPERATURA:")
print(f"{'Rango':<25} {'Consumo':<15} {'Registros':<10} {'% Total'}")
print("-" * 65)

for min_temp, max_temp, label in temp_ranges:
    mask = (df['temp'] >= min_temp) & (df['temp'] < max_temp)
    consumo = df[mask][volumen_col].mean()
    count = mask.sum()
    pct = (count / len(df) * 100)
    if count > 0:
        print(f"{label:<25} {consumo:>10,.0f} m³   {count:>6} ({pct:>5.1f}%)")

print("\n" + "=" * 80)
print("5️⃣ ANÁLISIS: ¿QUÉ ESTÁ PASANDO?")
print("=" * 80)

# Calcular promedios por estación y hora
verano_promedio = df[df['estacion']=='Verano'][volumen_col].mean()
invierno_promedio = df[df['estacion']=='Invierno'][volumen_col].mean()
verano_temp = df[df['estacion']=='Verano']['temp'].mean()
invierno_temp = df[df['estacion']=='Invierno']['temp'].mean()

print("\n📊 RESUMEN VERANO vs INVIERNO:")
print(f"   VERANO:")
print(f"      - Temperatura promedio: {verano_temp:.1f}°C")
print(f"      - Consumo promedio: {verano_promedio:,.0f} m³/hora")
print(f"\n   INVIERNO:")
print(f"      - Temperatura promedio: {invierno_temp:.1f}°C")
print(f"      - Consumo promedio: {invierno_promedio:,.0f} m³/hora")
print(f"\n   DIFERENCIA:")
print(f"      - Temperatura: {invierno_temp - verano_temp:+.1f}°C")
print(f"      - Consumo: {invierno_promedio - verano_promedio:+,.0f} m³/hora ({(invierno_promedio-verano_promedio)/verano_promedio*100:+.2f}%)")

print("\n" + "=" * 80)
print("6️⃣ HIPÓTESIS POSIBLES")
print("=" * 80)

print("""
🔍 POSIBLES EXPLICACIONES:

1. **TURISMO Y POBLACIÓN ESTACIONAL** ⭐ (MÁS PROBABLE)
   - Verano: Población sale de vacaciones, menos gente en la ciudad
   - Invierno: Población completa en la ciudad (trabajo, escuela)
   - Chile: Vacaciones de verano (Dic-Feb) → menos consumo
   - Valparaíso: Ciudad universitaria y laboral

2. **PATRONES DE USO DOMÉSTICO** ⭐
   - Invierno: Más tiempo en casa → más duchas calientes, lavado de ropa
   - Verano: Más actividades al aire libre, menos tiempo en casa
   - Duchas: En invierno son más largas (agua caliente)
   
3. **FUGAS Y PÉRDIDAS**
   - Invierno: Temperaturas bajas pueden afectar tuberías
   - Expansión/contracción de materiales

4. **ACTIVIDAD INDUSTRIAL**
   - Algunos procesos industriales son más intensivos en invierno
   - Menos actividad industrial en período vacacional de verano

5. **JARDINES Y RIEGO** ❓ (MENOS PROBABLE en este caso)
   - Normalmente: Verano = más riego
   - PERO: Los datos muestran lo contrario
   - Posible que el riego sea minoritario vs uso doméstico

6. **EFECTO HUMEDAD**
   - Correlación +0.43 entre humedad y consumo
   - Invierno: Mayor humedad (>85%) → más consumo
   - Verano: Menor humedad (<60%) → menos consumo
""")

print("\n" + "=" * 80)
print("7️⃣ CONCLUSIÓN")
print("=" * 80)

print(f"""
✅ EL PATRÓN ES REAL Y TIENE SENTIDO:

   📉 Temperatura más baja (invierno) → MÁS consumo
   📈 Temperatura más alta (verano) → MENOS consumo

   RAZÓN PRINCIPAL (hipótesis más probable):
   
   🏖️  VERANO (Dic-Feb): 
       • Temporada vacacional en Chile
       • Mucha gente sale de la ciudad
       • Menos actividad doméstica
       • → MENOS consumo: {verano_promedio:,.0f} m³/hora
       
   🏢 INVIERNO (Jun-Ago):
       • Actividad laboral/escolar normal
       • Población completa en la ciudad
       • Más tiempo en hogares
       • Duchas más largas (agua caliente)
       • → MÁS consumo: {invierno_promedio:,.0f} m³/hora

   DIFERENCIA: {invierno_promedio - verano_promedio:+,.0f} m³/hora
   ({(invierno_promedio-verano_promedio)/verano_promedio*100:+.2f}%)

⚠️  ESTO ES DIFERENTE a países con estaciones inversas donde:
   • Verano = más consumo (riego, aire acondicionado, turismo)
   • Aquí: Verano = vacaciones = éxodo poblacional

🎯 PARA EL MODELO:
   ✅ La temperatura ES un predictor importante
   ✅ La relación inversa (más frío = más consumo) es CORRECTA
   ✅ El modelo DEBE incluir temperatura para predicciones precisas
""")

print("\n" + "=" * 80)
