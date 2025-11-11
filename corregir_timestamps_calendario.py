#!/usr/bin/env python3
"""
Script para corregir timestamps en archivo de calendario.
Estandariza timestamps para que coincidan con volumen y clima.
"""

import pandas as pd
from pathlib import Path


def corregir_timestamps_calendario():
    print("=" * 70)
    print("CORRECCIÓN DE TIMESTAMPS - CALENDARIO")
    print("=" * 70)
    
    # Cargar datos
    input_file = Path("data/raw/calendar_social_ES_COMPLETO_20240101_20250930.csv")
    output_file = Path("data/raw/calendar_social_ES_COMPLETO_20240101_20250930_CORREGIDO.csv")
    
    print(f"\n📁 Cargando: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"   Total registros: {len(df)}")
    print(f"   Total columnas: {len(df.columns)}")
    
    # Mostrar timestamps originales
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        print(f"\n📅 Timestamps UTC originales:")
        print(f"   Inicio: {df['timestamp_utc'].min()}")
        print(f"   Fin: {df['timestamp_utc'].max()}")
    
    # Reconstruir timestamps con frecuencia horaria desde 2024-01-01 00:00:00
    print(f"\n🔧 Reconstruyendo timestamps...")
    print(f"   Nuevo inicio: 2024-01-01 00:00:00 UTC")
    
    # Crear timestamps horarios desde 2024-01-01 00:00:00
    start_date = '2024-01-01 00:00:00'
    timestamps_utc = pd.date_range(start=start_date, periods=len(df), freq='h', tz='UTC')
    
    # Reemplazar timestamp_utc
    df['timestamp_utc'] = timestamps_utc
    
    # Actualizar fecha_hora_local (UTC-3 para Chile)
    timestamps_local = timestamps_utc.tz_convert('America/Santiago')
    df['fecha_hora_local'] = timestamps_local
    
    print(f"   Nuevo fin UTC: {timestamps_utc[-1]}")
    print(f"   Nuevo fin Local: {timestamps_local[-1]}")
    
    # Actualizar columnas temporales basadas en el nuevo timestamp
    df['anio'] = df['timestamp_utc'].dt.year
    df['mes'] = df['timestamp_utc'].dt.month
    df['dia'] = df['timestamp_utc'].dt.day
    df['hora'] = df['timestamp_utc'].dt.hour
    df['dia_semana'] = df['timestamp_utc'].dt.dayofweek
    
    # Verificar que no hay duplicados
    n_duplicates = df['timestamp_utc'].duplicated().sum()
    print(f"\n✅ Verificación:")
    print(f"   Timestamps únicos: {df['timestamp_utc'].nunique()}")
    print(f"   Duplicados: {n_duplicates}")
    
    # Guardar archivo corregido
    print(f"\n💾 Guardando archivo corregido: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Primeras 5 filas del archivo corregido:")
    print(df[['fecha_hora_local', 'timestamp_utc', 'anio', 'mes', 'dia', 'hora']].head())
    
    print("\n" + "=" * 70)
    print("✅ TIMESTAMPS CORREGIDOS EXITOSAMENTE")
    print("=" * 70)
    print(f"\nArchivo generado: {output_file}")
    
    return df


if __name__ == "__main__":
    corregir_timestamps_calendario()
