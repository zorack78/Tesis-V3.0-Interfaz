"""
Script para crear la variable DEMANDA_m3_hr como nueva variable objetivo.

Demanda = Qin - ΔVolumen
- Qin: Caudal de entrada (m³/hr)
- ΔVolumen: Cambio en volumen almacenado (m³/hr)
- Demanda: Caudal consumido/demandado (m³/hr)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("CREACIÓN DE VARIABLE OBJETIVO: DEMANDA_m3_hr")
print("="*80)

# 1. Cargar datos procesados completos
data_path = Path('data/processed/data_processed_complete.csv')
print(f"\n📂 Cargando datos procesados: {data_path}")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
print(f"✅ Datos cargados: {len(df)} registros")
print(f"📊 Columnas: {len(df.columns)}")

# 2. Cargar Qin (datos originales)
qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')
print(f"\n📂 Cargando Qin: {qin_path}")

df_qin = pd.read_csv(qin_path)
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
print(f"✅ Qin cargado: {len(df_qin)} registros")

# 3. Asegurar timestamp como datetime en df
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
df = df.sort_values('timestamp_utc').reset_index(drop=True)

# 4. Merge con Qin
print(f"\n🔗 Combinando datos...")
df = pd.merge(df, df_qin, left_on='timestamp_utc', right_on='timestamp', how='left')
print(f"✅ Datos combinados: {len(df)} registros")

# 5. Verificar columnas
vol_col = 'Volumen_Total_m3'
qin_col = 'Qin'

if vol_col not in df.columns:
    print(f"❌ ERROR: No se encontró columna {vol_col}")
    exit(1)
    
if qin_col not in df.columns:
    print(f"❌ ERROR: No se encontró columna {qin_col}")
    exit(1)

print(f"\n🔍 Columnas confirmadas:")
print(f"   Volumen Total: {vol_col}")
print(f"   Qin (entrada): {qin_col}")

# 4. Calcular ΔVolumen
print(f"\n📐 Calculando ΔVolumen...")
df['delta_volumen_m3_hr'] = df[vol_col].diff()

# Primera fila no tiene ΔVolumen
print(f"   Primera fila sin ΔVolumen: {df['timestamp_utc'].iloc[0]}")

# 5. Calcular DEMANDA
print(f"\n🎯 Calculando DEMANDA = Qin - ΔVolumen...")
df['Demanda_m3_hr'] = df[qin_col] - df['delta_volumen_m3_hr']

# 6. Análisis de la nueva variable
print("\n" + "="*80)
print("ESTADÍSTICAS DE DEMANDA")
print("="*80)

# Excluir primera fila (NaN)
df_valid = df.dropna(subset=['Demanda_m3_hr'])

print(f"\n📊 Demanda_m3_hr:")
print(f"   Registros válidos: {len(df_valid)} de {len(df)}")
print(f"   Media: {df_valid['Demanda_m3_hr'].mean():,.2f} m³/hr")
print(f"   Mediana: {df_valid['Demanda_m3_hr'].median():,.2f} m³/hr")
print(f"   Std: {df_valid['Demanda_m3_hr'].std():,.2f} m³/hr")
print(f"   Min: {df_valid['Demanda_m3_hr'].min():,.2f} m³/hr")
print(f"   Max: {df_valid['Demanda_m3_hr'].max():,.2f} m³/hr")

# Percentiles
print(f"\n📈 Percentiles:")
for p in [5, 25, 50, 75, 95]:
    val = df_valid['Demanda_m3_hr'].quantile(p/100)
    print(f"   P{p}: {val:,.2f} m³/hr")

# 7. Análisis por hora del día
print("\n" + "="*80)
print("DEMANDA PROMEDIO POR HORA DEL DÍA")
print("="*80)

df_valid['hora'] = df_valid['timestamp_utc'].dt.hour
demanda_por_hora = df_valid.groupby('hora')['Demanda_m3_hr'].agg(['mean', 'std', 'min', 'max'])

print(f"\n{'Hora':>4} | {'Media':>11} | {'Std':>10} | {'Min':>11} | {'Max':>11}")
print("-" * 60)
for hora in range(24):
    media = demanda_por_hora.loc[hora, 'mean']
    std = demanda_por_hora.loc[hora, 'std']
    min_val = demanda_por_hora.loc[hora, 'min']
    max_val = demanda_por_hora.loc[hora, 'max']
    print(f"{hora:>4} | {media:>11,.0f} | {std:>10,.0f} | {min_val:>11,.0f} | {max_val:>11,.0f}")

# Identificar horarios de inflexión
hora_max = demanda_por_hora['mean'].idxmax()
hora_min = demanda_por_hora['mean'].idxmin()

print(f"\n🔝 Hora con MAYOR demanda: {hora_max}:00 → {demanda_por_hora.loc[hora_max, 'mean']:,.0f} m³/hr")
print(f"🔻 Hora con MENOR demanda: {hora_min}:00 → {demanda_por_hora.loc[hora_min, 'mean']:,.0f} m³/hr")

# 8. Verificar valores anómalos
print("\n" + "="*80)
print("VERIFICACIÓN DE VALORES ANÓMALOS")
print("="*80)

# Demandas extremadamente altas o negativas
demanda_negativa = df_valid[df_valid['Demanda_m3_hr'] < 0]
demanda_muy_alta = df_valid[df_valid['Demanda_m3_hr'] > 30000]

print(f"\n⚠️  Demandas negativas: {len(demanda_negativa)} registros ({len(demanda_negativa)/len(df_valid)*100:.2f}%)")
if len(demanda_negativa) > 0:
    print(f"   Media: {demanda_negativa['Demanda_m3_hr'].mean():,.0f} m³/hr")
    print(f"   Min: {demanda_negativa['Demanda_m3_hr'].min():,.0f} m³/hr")

print(f"\n⚠️  Demandas > 30,000 m³/hr: {len(demanda_muy_alta)} registros ({len(demanda_muy_alta)/len(df_valid)*100:.2f}%)")
if len(demanda_muy_alta) > 0:
    print(f"   Media: {demanda_muy_alta['Demanda_m3_hr'].mean():,.0f} m³/hr")
    print(f"   Max: {demanda_muy_alta['Demanda_m3_hr'].max():,.0f} m³/hr")

# 9. Comparar con Qin
print("\n" + "="*80)
print("COMPARACIÓN: DEMANDA vs QIN")
print("="*80)

print(f"\n📊 Qin (entrada):")
print(f"   Media: {df_valid[qin_col].mean():,.2f} m³/hr")
print(f"   Mediana: {df_valid[qin_col].median():,.2f} m³/hr")

print(f"\n📊 Demanda (salida):")
print(f"   Media: {df_valid['Demanda_m3_hr'].mean():,.2f} m³/hr")
print(f"   Mediana: {df_valid['Demanda_m3_hr'].median():,.2f} m³/hr")

print(f"\n🔄 Balance:")
balance = df_valid[qin_col].mean() - df_valid['Demanda_m3_hr'].mean()
print(f"   Qin - Demanda = {balance:,.2f} m³/hr")
print(f"   {'✅ Sistema en equilibrio' if abs(balance) < 100 else '⚠️  Sistema desbalanceado'}")

# 10. Guardar datos con nueva variable
print("\n" + "="*80)
print("GUARDANDO DATOS ACTUALIZADOS")
print("="*80)

# Guardar CSV completo
output_path = Path('data/processed/data_processed_con_demanda.csv')
df.to_csv(output_path, index=False)
print(f"✅ Guardado: {output_path}")
print(f"   Registros: {len(df)}")
print(f"   Columnas: {len(df.columns)}")

# Guardar solo registros válidos (sin NaN en Demanda)
output_valid_path = Path('data/processed/data_processed_demanda_valid.csv')
df_valid_full = df.dropna(subset=['Demanda_m3_hr']).reset_index(drop=True)
df_valid_full.to_csv(output_valid_path, index=False)
print(f"✅ Guardado (solo válidos): {output_valid_path}")
print(f"   Registros: {len(df_valid_full)}")

# 11. Crear archivo de información
info = {
    'variable_objetivo': 'Demanda_m3_hr',
    'formula': 'Qin - delta_volumen_m3_hr',
    'unidad': 'm3/hr',
    'descripcion': 'Caudal demandado/consumido por hora',
    'registros_totales': int(len(df)),
    'registros_validos': int(len(df_valid_full)),
    'estadisticas': {
        'media': float(df_valid['Demanda_m3_hr'].mean()),
        'mediana': float(df_valid['Demanda_m3_hr'].median()),
        'std': float(df_valid['Demanda_m3_hr'].std()),
        'min': float(df_valid['Demanda_m3_hr'].min()),
        'max': float(df_valid['Demanda_m3_hr'].max()),
        'p5': float(df_valid['Demanda_m3_hr'].quantile(0.05)),
        'p95': float(df_valid['Demanda_m3_hr'].quantile(0.95))
    },
    'hora_pico': {
        'hora': int(hora_max),
        'demanda_promedio': float(demanda_por_hora.loc[hora_max, 'mean'])
    },
    'hora_valle': {
        'hora': int(hora_min),
        'demanda_promedio': float(demanda_por_hora.loc[hora_min, 'mean'])
    },
    'comparacion_qin': {
        'qin_media': float(df_valid[qin_col].mean()),
        'demanda_media': float(df_valid['Demanda_m3_hr'].mean()),
        'balance': float(balance)
    }
}

info_path = Path('data/processed/demanda_info.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"✅ Info guardada: {info_path}")

print("\n" + "="*80)
print("✅ PROCESO COMPLETADO")
print("="*80)
print(f"""
RESUMEN:
• Variable creada: Demanda_m3_hr
• Fórmula: Qin - ΔVolumen
• Demanda media: {df_valid['Demanda_m3_hr'].mean():,.0f} m³/hr
• Horario de inflexión máximo: {hora_max}:00 ({demanda_por_hora.loc[hora_max, 'mean']:,.0f} m³/hr)
• Horario de inflexión mínimo: {hora_min}:00 ({demanda_por_hora.loc[hora_min, 'mean']:,.0f} m³/hr)
• Archivos generados:
  - {output_path}
  - {output_valid_path}
  - {info_path}
""")
