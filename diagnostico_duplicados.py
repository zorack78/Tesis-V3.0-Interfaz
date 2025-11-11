#!/usr/bin/env python3
"""
Script de diagnóstico para investigar duplicados en los datos.
"""

import pandas as pd
from pathlib import Path

def diagnosticar_duplicados():
    print("🔍 DIAGNÓSTICO DE DUPLICADOS EN DATOS")
    print("=" * 60)
    
    # Cargar datos raw
    data_path = Path("data/raw")
    
    # 1. Volumen
    print("\n1️⃣ DATOS DE VOLUMEN:")
    vol_df = pd.read_csv(data_path / "BD_VolTotal_X_Hr_m3_UTC.csv")
    vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'])
    
    print(f"   Total registros: {len(vol_df)}")
    print(f"   Registros únicos por timestamp: {vol_df['timestamp'].nunique()}")
    print(f"   Duplicados: {len(vol_df) - vol_df['timestamp'].nunique()}")
    
    # Ver primeras fechas
    print(f"\n   Rango de fechas:")
    print(f"   - Inicio: {vol_df['timestamp'].min()}")
    print(f"   - Fin: {vol_df['timestamp'].max()}")
    
    # Ver duplicados
    dups = vol_df[vol_df.duplicated(subset=['timestamp'], keep=False)]
    if len(dups) > 0:
        print(f"\n   ⚠️  Ejemplos de timestamps duplicados:")
        print(dups.head(10)[['timestamp', 'Volumen_Total_m3']])
    
    # 2. Calendario
    print("\n\n2️⃣ DATOS DE CALENDARIO:")
    cal_df = pd.read_csv(data_path / "calendar_social_ES_COMPLETO_20240101_20250930.csv")
    cal_df['timestamp_utc'] = pd.to_datetime(cal_df['timestamp_utc'])
    
    print(f"   Total registros: {len(cal_df)}")
    print(f"   Registros únicos por timestamp: {cal_df['timestamp_utc'].nunique()}")
    print(f"   Duplicados: {len(cal_df) - cal_df['timestamp_utc'].nunique()}")
    
    print(f"\n   Rango de fechas:")
    print(f"   - Inicio: {cal_df['timestamp_utc'].min()}")
    print(f"   - Fin: {cal_df['timestamp_utc'].max()}")
    
    # Ver duplicados
    dups_cal = cal_df[cal_df.duplicated(subset=['timestamp_utc'], keep=False)]
    if len(dups_cal) > 0:
        print(f"\n   ⚠️  Ejemplos de timestamps duplicados:")
        print(dups_cal.head(10)[['timestamp_utc', 'fecha_hora_local']])
    
    # 3. Después del merge
    print("\n\n3️⃣ DESPUÉS DEL MERGE:")
    vol_df_renamed = vol_df.rename(columns={'timestamp': 'timestamp_utc'})
    merged = pd.merge(vol_df_renamed, cal_df, on='timestamp_utc', how='left')
    
    print(f"   Total registros después del merge: {len(merged)}")
    print(f"   Registros únicos por timestamp: {merged['timestamp_utc'].nunique()}")
    print(f"   Duplicados generados: {len(merged) - merged['timestamp_utc'].nunique()}")
    
    # Contar NaN en columnas clave
    print(f"\n   NaN en columnas clave:")
    print(f"   - Volumen: {merged[' Volumen_Total_m3'].isna().sum()}")
    print(f"   - fecha_hora_local: {merged['fecha_hora_local'].isna().sum()}")
    
    # 4. Frecuencia de datos
    print("\n\n4️⃣ FRECUENCIA Y GAPS:")
    vol_df_sorted = vol_df.sort_values('timestamp')
    diff = vol_df_sorted['timestamp'].diff()
    
    print(f"   Diferencia entre registros consecutivos:")
    print(f"   - Más común: {diff.mode().values[0] if len(diff.mode()) > 0 else 'N/A'}")
    print(f"   - Mínima: {diff.min()}")
    print(f"   - Máxima: {diff.max()}")
    print(f"   - Mediana: {diff.median()}")
    
    # Contar frecuencias
    freq_counts = diff.value_counts().head(5)
    print(f"\n   Top 5 frecuencias:")
    for freq, count in freq_counts.items():
        print(f"   - {freq}: {count} veces")
    
    print("\n" + "=" * 60)
    print("✅ Diagnóstico completado")

if __name__ == "__main__":
    diagnosticar_duplicados()
