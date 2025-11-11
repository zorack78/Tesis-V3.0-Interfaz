"""
Análisis para entender:
1. ¿Qué representa Volumen_Total_m3?
2. ¿Es volumen almacenado o caudal consumido?
3. ¿Cómo se relaciona con la demanda real (m³/hr)?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

# Cargar datos
print("="*80)
print("ANÁLISIS: VOLUMEN TOTAL vs DEMANDA REAL")
print("="*80)

# 1. Cargar Volumen Total
vol_path = Path('data/raw/BD_VolTotal_X_Hr_m3_UTC.csv')
df_vol = pd.read_csv(vol_path)
df_vol.columns = df_vol.columns.str.strip()
df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'])
df_vol = df_vol.sort_values('timestamp').reset_index(drop=True)

print(f"\n📊 VOLUMEN TOTAL:")
print(f"   Registros: {len(df_vol)}")
print(f"   Rango: {df_vol['Volumen_Total_m3'].min():,.0f} - {df_vol['Volumen_Total_m3'].max():,.0f} m³")
print(f"   Promedio: {df_vol['Volumen_Total_m3'].mean():,.0f} m³")

# 2. Cargar Qin (caudal de entrada)
qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')
df_qin = pd.read_csv(qin_path)
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
df_qin = df_qin.sort_values('timestamp').reset_index(drop=True)

print(f"\n💧 CAUDAL DE ENTRADA (Qin):")
print(f"   Registros: {len(df_qin)}")
print(f"   Rango: {df_qin['Qin'].min():,.0f} - {df_qin['Qin'].max():,.0f} m³/hr")
print(f"   Promedio: {df_qin['Qin'].mean():,.0f} m³/hr")

# 3. Combinar datos
df = pd.merge(df_vol, df_qin, on='timestamp', how='inner')
df = df.sort_values('timestamp').reset_index(drop=True)

# 4. CALCULAR DEMANDA (diferencia de volumen)
# Si Volumen_Total es ALMACENAMIENTO, la demanda = Qin - ΔVolumen
df['delta_volumen'] = df['Volumen_Total_m3'].diff()
df['demanda_calculada'] = df['Qin'] - df['delta_volumen']

print(f"\n📈 ANÁLISIS DE DIFERENCIAS:")
print(f"   ΔVolumen promedio: {df['delta_volumen'].mean():,.0f} m³/hr")
print(f"   ΔVolumen min: {df['delta_volumen'].min():,.0f} m³/hr")
print(f"   ΔVolumen max: {df['delta_volumen'].max():,.0f} m³/hr")
print(f"   ΔVolumen std: {df['delta_volumen'].std():,.0f} m³/hr")

# 5. Analizar primer día completo
print("\n" + "="*80)
print("ANÁLISIS DEL 1 DE ENERO 2024 (Primer día completo)")
print("="*80)

df_dia1 = df[df['timestamp'].dt.date == pd.Timestamp('2024-01-01').date()].copy()

print(f"\n{'Hora':>4} | {'Volumen_Total':>14} | {'ΔVolumen':>11} | {'Qin':>11} | {'Demanda':>11}")
print("-" * 70)

for idx, row in df_dia1.iterrows():
    hora = row['timestamp'].hour
    vol = row['Volumen_Total_m3']
    delta = row['delta_volumen']
    qin = row['Qin']
    demanda = row['demanda_calculada']
    
    delta_str = f"{delta:,.0f}" if not pd.isna(delta) else "N/A"
    demanda_str = f"{demanda:,.0f}" if not pd.isna(demanda) else "N/A"
    
    print(f"{hora:>4} | {vol:>14,.0f} | {delta_str:>11} | {qin:>11,.0f} | {demanda_str:>11}")

# 6. Analizar comportamiento típico por hora
print("\n" + "="*80)
print("COMPORTAMIENTO PROMEDIO POR HORA DEL DÍA")
print("="*80)

df['hora'] = df['timestamp'].dt.hour
analisis_hora = df.groupby('hora').agg({
    'Volumen_Total_m3': ['mean', 'std', 'min', 'max'],
    'delta_volumen': ['mean', 'std'],
    'Qin': ['mean', 'std'],
    'demanda_calculada': ['mean', 'std']
}).round(0)

print("\n📊 VOLUMEN TOTAL PROMEDIO POR HORA:")
print(f"\n{'Hora':>4} | {'Vol_Mean':>11} | {'Vol_Std':>10} | {'ΔVol_Mean':>11} | {'Qin_Mean':>10} | {'Demanda':>11}")
print("-" * 80)

for hora in range(24):
    vol_mean = analisis_hora.loc[hora, ('Volumen_Total_m3', 'mean')]
    vol_std = analisis_hora.loc[hora, ('Volumen_Total_m3', 'std')]
    delta_mean = analisis_hora.loc[hora, ('delta_volumen', 'mean')]
    qin_mean = analisis_hora.loc[hora, ('Qin', 'mean')]
    demanda_mean = analisis_hora.loc[hora, ('demanda_calculada', 'mean')]
    
    print(f"{hora:>4} | {vol_mean:>11,.0f} | {vol_std:>10,.0f} | {delta_mean:>11,.0f} | {qin_mean:>10,.0f} | {demanda_mean:>11,.0f}")

# 7. INTERPRETACIÓN
print("\n" + "="*80)
print("🔍 INTERPRETACIÓN DE LOS DATOS")
print("="*80)

# Correlación entre Qin y ΔVolumen
corr_qin_delta = df[['Qin', 'delta_volumen']].corr().iloc[0, 1]
print(f"\n📊 Correlación Qin vs ΔVolumen: {corr_qin_delta:.3f}")

# Si ΔVolumen es positivo cuando Qin es bajo = el sistema se está vaciando (demanda)
# Si ΔVolumen es negativo cuando Qin es alto = el sistema se está llenando
demanda_vs_qin = df[['Qin', 'demanda_calculada']].corr().iloc[0, 1]
print(f"📊 Correlación Qin vs Demanda_Calculada: {demanda_vs_qin:.3f}")

# Analizar picos
print(f"\n🔝 HORA CON MAYOR VOLUMEN ALMACENADO:")
hora_max_vol = analisis_hora[('Volumen_Total_m3', 'mean')].idxmax()
vol_max = analisis_hora.loc[hora_max_vol, ('Volumen_Total_m3', 'mean')]
print(f"   Hora: {hora_max_vol}:00 → {vol_max:,.0f} m³")

print(f"\n🔻 HORA CON MENOR VOLUMEN ALMACENADO:")
hora_min_vol = analisis_hora[('Volumen_Total_m3', 'mean')].idxmin()
vol_min = analisis_hora.loc[hora_min_vol, ('Volumen_Total_m3', 'mean')]
print(f"   Hora: {hora_min_vol}:00 → {vol_min:,.0f} m³")

print(f"\n💧 HORA CON MAYOR DEMANDA CALCULADA:")
hora_max_demanda = analisis_hora[('demanda_calculada', 'mean')].idxmax()
demanda_max = analisis_hora.loc[hora_max_demanda, ('demanda_calculada', 'mean')]
print(f"   Hora: {hora_max_demanda}:00 → {demanda_max:,.0f} m³/hr")

print(f"\n💧 HORA CON MENOR DEMANDA CALCULADA:")
hora_min_demanda = analisis_hora[('demanda_calculada', 'mean')].idxmin()
demanda_min = analisis_hora.loc[hora_min_demanda, ('demanda_calculada', 'mean')]
print(f"   Hora: {hora_min_demanda}:00 → {demanda_min:,.0f} m³/hr")

# 8. CONCLUSIÓN
print("\n" + "="*80)
print("💡 CONCLUSIONES")
print("="*80)

print(f"""
1. VOLUMEN_TOTAL_M3 representa:
   {'✅ VOLUMEN ALMACENADO en los tanques del sistema' if vol_max > 130000 else '❓'}
   
2. El sistema tiene un patrón de:
   • Volumen MÁXIMO a las {hora_max_vol}:00 hrs → Sistema lleno para el día
   • Volumen MÍNIMO a las {hora_min_vol}:00 hrs → Sistema más vacío
   • Mayor DEMANDA a las {hora_max_demanda}:00 hrs → Consumo pico
   • Menor DEMANDA a las {hora_min_demanda}:00 hrs → Consumo bajo

3. Durante la noche/madrugada:
   {'✅ El sistema se LLENA (Qin > Demanda)' if analisis_hora.loc[3, ('delta_volumen', 'mean')] > 0 else '❌ El sistema se VACÍA'}
   
4. Durante el día:
   {'✅ El sistema se VACÍA (Demanda > Qin)' if analisis_hora.loc[14, ('delta_volumen', 'mean')] < 0 else '❌ El sistema se LLENA'}

5. PROBLEMA IDENTIFICADO:
   ❌ El modelo está prediciendo VOLUMEN ALMACENADO (m³)
   ❌ NO está prediciendo DEMANDA/CONSUMO (m³/hr)
   ✅ Para obtener DEMANDA necesitamos: Demanda = Qin - ΔVolumen
""")

# 9. Visualización
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Gráfico 1: Volumen Total vs Hora
axes[0].plot(analisis_hora.index, analisis_hora[('Volumen_Total_m3', 'mean')], 
             marker='o', linewidth=2, markersize=8, color='blue')
axes[0].fill_between(analisis_hora.index, 
                      analisis_hora[('Volumen_Total_m3', 'mean')] - analisis_hora[('Volumen_Total_m3', 'std')],
                      analisis_hora[('Volumen_Total_m3', 'mean')] + analisis_hora[('Volumen_Total_m3', 'std')],
                      alpha=0.3, color='blue')
axes[0].axvline(hora_max_vol, color='green', linestyle='--', label=f'Max Vol: {hora_max_vol}:00')
axes[0].axvline(hora_min_vol, color='red', linestyle='--', label=f'Min Vol: {hora_min_vol}:00')
axes[0].set_xlabel('Hora del día', fontsize=12)
axes[0].set_ylabel('Volumen Almacenado (m³)', fontsize=12)
axes[0].set_title('VOLUMEN TOTAL ALMACENADO POR HORA (Promedio)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: ΔVolumen (tasa de cambio)
axes[1].bar(analisis_hora.index, analisis_hora[('delta_volumen', 'mean')], 
            color=['green' if x > 0 else 'red' for x in analisis_hora[('delta_volumen', 'mean')]])
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Hora del día', fontsize=12)
axes[1].set_ylabel('ΔVolumen (m³/hr)', fontsize=12)
axes[1].set_title('CAMBIO EN VOLUMEN ALMACENADO (ΔVol) - Verde=Llenado, Rojo=Vaciado', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Gráfico 3: Demanda Calculada
axes[2].plot(analisis_hora.index, analisis_hora[('demanda_calculada', 'mean')], 
             marker='s', linewidth=2, markersize=8, color='red')
axes[2].fill_between(analisis_hora.index, 
                      analisis_hora[('demanda_calculada', 'mean')] - analisis_hora[('demanda_calculada', 'std')],
                      analisis_hora[('demanda_calculada', 'mean')] + analisis_hora[('demanda_calculada', 'std')],
                      alpha=0.3, color='red')
axes[2].axvline(hora_max_demanda, color='darkred', linestyle='--', label=f'Max Demanda: {hora_max_demanda}:00')
axes[2].set_xlabel('Hora del día', fontsize=12)
axes[2].set_ylabel('Demanda (m³/hr)', fontsize=12)
axes[2].set_title('DEMANDA CALCULADA (Qin - ΔVolumen)', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/analisis_volumen_vs_demanda.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: outputs/analisis_volumen_vs_demanda.png")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)
