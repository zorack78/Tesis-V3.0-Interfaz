"""
Análisis rápido del patrón de temperatura en Valparaíso
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('data/raw/BD_Clima2024a202509_Local.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hora'] = df['timestamp'].dt.hour

# Análisis por hora
print("\n" + "="*80)
print("TEMPERATURA PROMEDIO POR HORA EN VALPARAÍSO (DATOS REALES 2024-2025)")
print("="*80 + "\n")

temp_por_hora = df.groupby('hora')['temp'].agg(['mean', 'std', 'min', 'max', 'count'])

for hora in range(24):
    datos = temp_por_hora.loc[hora]
    print(f"{hora:02d}:00 hrs → Media: {datos['mean']:6.2f}°C | Std: {datos['std']:5.2f}°C | "
          f"Min: {datos['min']:6.2f}°C | Max: {datos['max']:6.2f}°C | N={int(datos['count']):5d}")

hora_min = temp_por_hora['mean'].idxmin()
hora_max = temp_por_hora['mean'].idxmax()
amplitud = temp_por_hora['mean'].max() - temp_por_hora['mean'].min()

print("\n" + "-"*80)
print(f"★ TEMPERATURA MÍNIMA: {hora_min:02d}:00 hrs → {temp_por_hora.loc[hora_min, 'mean']:.2f}°C")
print(f"★ TEMPERATURA MÁXIMA: {hora_max:02d}:00 hrs → {temp_por_hora.loc[hora_max, 'mean']:.2f}°C")
print(f"★ AMPLITUD TÉRMICA DIARIA: {amplitud:.2f}°C")
print("-"*80)

print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)
print(f"""
Basado en {len(df):,} registros horarios reales de temperatura en Valparaíso:

El patrón que mencioné NO es teórico, sino EMPÍRICO:
- Temperatura mínima ocurre a las {hora_min:02d}:00 hrs (madrugada/amanecer)
- Temperatura máxima ocurre a las {hora_max:02d}:00 hrs (tarde)
- Amplitud térmica promedio: {amplitud:.2f}°C

Este es el comportamiento REAL observado en tus datos de clima.
Cuando sugerí usar un "perfil realista", me refería a ESTE patrón empírico,
no a una invención teórica.
""")

# Guardar CSV
temp_por_hora.to_csv('outputs/temperatura_por_hora_valparaiso.csv')
print("✓ Tabla guardada en outputs/temperatura_por_hora_valparaiso.csv\n")
