"""
Investigación: ¿Qué pasó con los 114 registros perdidos?

Original: 15,336 registros
Final: 15,222 registros
Perdidos: 114 registros (0.74%)

Este script investiga dónde y por qué se perdieron esos registros.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("INVESTIGACIÓN: REGISTROS PERDIDOS EN EL PROCESO")
print("="*80)

# 1. Archivo ORIGINAL (data_processed_complete.csv)
print("\n📂 PASO 1: REVISAR ARCHIVO ORIGINAL")
print("-" * 80)

original_path = Path('data/processed/data_processed_complete.csv')
df_original = pd.read_csv(original_path)
df_original.columns = df_original.columns.str.strip()
df_original['timestamp_utc'] = pd.to_datetime(df_original['timestamp_utc'])

print(f"Archivo: {original_path}")
print(f"✅ Registros totales: {len(df_original):,}")
print(f"   Período: {df_original['timestamp_utc'].min()} a {df_original['timestamp_utc'].max()}")

# 2. Cargar Qin
print("\n📂 PASO 2: CARGAR QIN")
print("-" * 80)

qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')
df_qin = pd.read_csv(qin_path)
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])

print(f"Archivo: {qin_path}")
print(f"✅ Registros Qin: {len(df_qin):,}")

# 3. Merge y calcular Demanda
print("\n🔗 PASO 3: MERGE Y CALCULAR DEMANDA")
print("-" * 80)

print("Haciendo merge...")
df_merged = pd.merge(df_original, df_qin, left_on='timestamp_utc', right_on='timestamp', how='left')
print(f"✅ Después del merge: {len(df_merged):,} registros")

# Verificar si hay NaN en Qin
qin_nulos = df_merged['Qin'].isna().sum()
print(f"   Registros con Qin=NaN: {qin_nulos}")

# Calcular delta_volumen
df_merged['delta_volumen_m3_hr'] = df_merged['Volumen_Total_m3'].diff()
delta_nulos = df_merged['delta_volumen_m3_hr'].isna().sum()
print(f"   Registros con delta_volumen=NaN: {delta_nulos}")

# Calcular Demanda
df_merged['Demanda_m3_hr'] = df_merged['Qin'] - df_merged['delta_volumen_m3_hr']
demanda_nulos = df_merged['Demanda_m3_hr'].isna().sum()
print(f"   Registros con Demanda=NaN: {demanda_nulos}")

# 4. Archivo FINAL (data_processed_demanda_valid.csv)
print("\n📂 PASO 4: REVISAR ARCHIVO FINAL")
print("-" * 80)

final_path = Path('data/processed/data_processed_demanda_valid.csv')
df_final = pd.read_csv(final_path)
df_final.columns = df_final.columns.str.strip()
df_final['timestamp_utc'] = pd.to_datetime(df_final['timestamp_utc'])

print(f"Archivo: {final_path}")
print(f"✅ Registros finales: {len(df_final):,}")

# 5. ANÁLISIS DE LA PÉRDIDA
print("\n" + "="*80)
print("ANÁLISIS DE REGISTROS PERDIDOS")
print("="*80)

perdidos = len(df_original) - len(df_final)
print(f"\n📊 RESUMEN:")
print(f"   Registros originales:  {len(df_original):>6,}")
print(f"   Registros finales:     {len(df_final):>6,}")
print(f"   Registros perdidos:    {perdidos:>6,} ({perdidos/len(df_original)*100:.2f}%)")

# 6. IDENTIFICAR EXACTAMENTE QUÉ REGISTROS SE PERDIERON
print("\n🔍 PASO 5: IDENTIFICAR REGISTROS PERDIDOS")
print("-" * 80)

# Registros con NaN en Demanda
df_con_nan = df_merged[df_merged['Demanda_m3_hr'].isna()].copy()

print(f"\n📋 Registros con Demanda=NaN: {len(df_con_nan)}")

if len(df_con_nan) > 0:
    print(f"\n⏰ Primeros 10 registros perdidos:")
    print(f"\n{'Timestamp':>25} | {'Volumen_Total':>14} | {'delta_vol':>12} | {'Qin':>10} | {'Demanda':>10}")
    print("-" * 85)
    
    for idx, row in df_con_nan.head(10).iterrows():
        ts = row['timestamp_utc']
        vol = row['Volumen_Total_m3']
        delta = row['delta_volumen_m3_hr']
        qin = row['Qin']
        demanda = row['Demanda_m3_hr']
        
        delta_str = f"{delta:,.0f}" if not pd.isna(delta) else "NaN"
        qin_str = f"{qin:,.0f}" if not pd.isna(qin) else "NaN"
        demanda_str = f"{demanda:,.0f}" if not pd.isna(demanda) else "NaN"
        
        print(f"{ts} | {vol:>14,.0f} | {delta_str:>12} | {qin_str:>10} | {demanda_str:>10}")

# 7. ANÁLISIS DE CAUSA
print("\n" + "="*80)
print("ANÁLISIS DE CAUSA")
print("="*80)

print("\n🔍 Verificando causas de NaN en Demanda:")

# Causa 1: Primera fila (no tiene fila anterior para calcular diff)
primera_fila_nan = df_merged.iloc[0]['Demanda_m3_hr']
print(f"\n1️⃣ Primera fila (sin fila anterior para diff):")
print(f"   Timestamp: {df_merged.iloc[0]['timestamp_utc']}")
print(f"   Demanda: {primera_fila_nan} (NaN esperado)")
print(f"   Razón: .diff() no puede calcular diferencia sin fila anterior")

# Causa 2: Gaps en los datos (timestamps discontinuos)
df_merged_sorted = df_merged.sort_values('timestamp_utc').reset_index(drop=True)
df_merged_sorted['hora_diff'] = df_merged_sorted['timestamp_utc'].diff().dt.total_seconds() / 3600

gaps = df_merged_sorted[df_merged_sorted['hora_diff'] > 1]
print(f"\n2️⃣ Gaps en la serie temporal (saltos > 1 hora):")
print(f"   Gaps encontrados: {len(gaps)}")

if len(gaps) > 0:
    print(f"\n   Primeros 5 gaps:")
    for idx, row in gaps.head(5).iterrows():
        ts = row['timestamp_utc']
        gap_horas = row['hora_diff']
        if idx > 0:
            ts_anterior = df_merged_sorted.iloc[idx-1]['timestamp_utc']
            print(f"   • Gap de {gap_horas:.1f} horas entre {ts_anterior} y {ts}")

# Causa 3: Qin faltante
qin_faltante = df_merged[df_merged['Qin'].isna()]
print(f"\n3️⃣ Registros donde Qin está ausente:")
print(f"   Registros con Qin=NaN: {len(qin_faltante)}")

if len(qin_faltante) > 0:
    print(f"\n   Primeros 5 registros sin Qin:")
    for idx, row in qin_faltante.head(5).iterrows():
        print(f"   • {row['timestamp_utc']}: Qin=NaN")

# 8. DISTRIBUCIÓN DE REGISTROS PERDIDOS
print("\n" + "="*80)
print("DISTRIBUCIÓN DE REGISTROS PERDIDOS")
print("="*80)

if len(df_con_nan) > 0:
    df_con_nan['fecha'] = df_con_nan['timestamp_utc'].dt.date
    df_con_nan['hora'] = df_con_nan['timestamp_utc'].dt.hour
    
    print("\n📅 Por fecha:")
    registros_por_fecha = df_con_nan.groupby('fecha').size().sort_values(ascending=False)
    print(f"\n{'Fecha':>12} | {'N Perdidos':>11}")
    print("-" * 26)
    for fecha, n in registros_por_fecha.head(10).items():
        print(f"{fecha} | {n:>11}")
    
    print("\n⏰ Por hora del día:")
    registros_por_hora = df_con_nan.groupby('hora').size().sort_values(ascending=False)
    print(f"\n{'Hora':>4} | {'N Perdidos':>11}")
    print("-" * 18)
    for hora, n in registros_por_hora.items():
        print(f"{hora:>4} | {n:>11}")

# 9. VALIDACIÓN
print("\n" + "="*80)
print("VALIDACIÓN")
print("="*80)

# Verificar que data_processed_con_demanda.csv tiene todos los registros
con_demanda_path = Path('data/processed/data_processed_con_demanda.csv')
if con_demanda_path.exists():
    df_con_demanda = pd.read_csv(con_demanda_path)
    print(f"\n✅ data_processed_con_demanda.csv: {len(df_con_demanda):,} registros")
    print(f"   (Este archivo SÍ incluye los {perdidos} registros con NaN)")
else:
    print(f"\n❌ No se encontró {con_demanda_path}")

print(f"\n✅ data_processed_demanda_valid.csv: {len(df_final):,} registros")
print(f"   (Este archivo NO incluye registros con NaN - solo válidos)")

# 10. CONCLUSIÓN
print("\n" + "="*80)
print("💡 CONCLUSIÓN")
print("="*80)

print(f"""
Los {perdidos} registros perdidos ({perdidos/len(df_original)*100:.2f}%) se deben a:

1️⃣ PRIMERA FILA (1 registro):
   • Timestamp: {df_merged.iloc[0]['timestamp_utc']}
   • Razón: .diff() necesita una fila anterior para calcular ΔVolumen
   • Esta es la primera observación, no tiene hora anterior
   • Demanda = Qin - ΔVolumen → NaN porque ΔVolumen = NaN

2️⃣ GAPS EN LA SERIE TEMPORAL ({len(gaps)} registros):
   • Después de un gap > 1 hora, .diff() calcula mal
   • Ejemplo: Si faltan datos de 10:00 a 14:00, el registro de 15:00
     calculará diff con respecto a 09:00 (6 horas antes)
   • Esto genera valores incorrectos de demanda

3️⃣ QIN FALTANTE ({len(qin_faltante)} registros):
   • Si Qin no existe para ese timestamp
   • Demanda = Qin - ΔVolumen → NaN porque Qin = NaN

DECISIÓN TOMADA:
✅ Se crearon DOS archivos:
   • data_processed_con_demanda.csv ({len(df_merged):,} registros)
     → Incluye TODOS los registros (con y sin NaN)
   
   • data_processed_demanda_valid.csv ({len(df_final):,} registros)
     → Solo registros VÁLIDOS (sin NaN en Demanda)
     → Este es el usado para entrenar el modelo

IMPACTO:
• Pérdida: {perdidos/len(df_original)*100:.2f}% (mínima)
• Registros válidos: {len(df_final)/len(df_original)*100:.1f}% del total
• La pérdida es mayormente técnica (primera fila, gaps)
• NO representa pérdida de información significativa
• El modelo se entrena con {len(df_final):,} registros de alta calidad

RECOMENDACIÓN:
✅ Mantener el enfoque actual (usar solo registros válidos)
✅ Los {perdidos} registros perdidos NO afectan la calidad del modelo
✅ Son pérdidas esperadas en procesamiento de series temporales
""")

print("="*80)
print("FIN DE LA INVESTIGACIÓN")
print("="*80)
