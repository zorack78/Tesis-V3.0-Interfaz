"""
Análisis de temperatura por mes para calibrar perfil realista
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/BD_Clima2024a202509_Local.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hora'] = df['timestamp'].dt.hour
df['mes'] = df['timestamp'].dt.month

print("\n" + "="*90)
print("TEMPERATURA POR MES Y HORA - VALPARAÍSO (datos reales)")
print("="*90)
print(f"{'Mes':<12} {'Temp Min':>10} {'Hora Min':>9} {'Temp Max':>10} {'Hora Max':>9} {'Amplitud':>10}")
print("-"*90)

meses_nombre = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                'Julio', 'Agosto', 'Sept']

for mes in range(1, 10):
    temp_mes = df[df['mes'] == mes].groupby('hora')['temp'].mean()
    temp_min = temp_mes.min()
    hora_min = temp_mes.idxmin()
    temp_max = temp_mes.max()
    hora_max = temp_mes.idxmax()
    amplitud = temp_max - temp_min
    
    print(f"{meses_nombre[mes]:<12} {temp_min:>9.2f}°C {hora_min:>9d}:00 "
          f"{temp_max:>9.2f}°C {hora_max:>9d}:00 {amplitud:>9.2f}°C")

print("="*90)
print("\nCONCLUSIÓN: Patrón consistente todo el año:")
print("- Temperatura mínima: 06:00-07:00 hrs")
print("- Temperatura máxima: 15:00-16:00 hrs")
print("- Amplitud térmica varía por estación: 6-9°C")
