#!/usr/bin/env python3
"""
Script para corregir timestamps en archivo de clima.
Estandariza timestamps para que coincidan con el archivo de volumen.
"""

import pandas as pd
from pathlib import Path

def corregir_timestamps_clima():
    print("=" * 70)
    print("CORRECCIÓN DE TIMESTAMPS - CLIMA")
    print("=" * 70)
    
    # Cargar datos
    input_file = Path("data/raw/BD_Clima2024a202509_UTC.csv")
    output_file = Path("data/raw/BD_Clima2024a202509_UTC_CORREGIDO.csv")
    
    print(f"\n📁 Cargando: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"   Total registros: {len(df)}")
    print(f"   Columnas: {df.columns.tolist()}")
    
    # Mostrar timestamps originales
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\n📅 Timestamps originales:")
    print(f"   Inicio: {df['timestamp'].min()}")
    print(f"   Fin: {df['timestamp'].max()}")
    
    # Reconstruir timestamps con frecuencia horaria desde 2024-01-01 00:00:00
    print(f"\n🔧 Reconstruyendo timestamps...")
    print(f"   Nuevo inicio: 2024-01-01 00:00:00 UTC")
    
    # Crear timestamps horarios desde 2024-01-01 00:00:00
    start_date = '2024-01-01 00:00:00'
    timestamps = pd.date_range(start=start_date, periods=len(df), freq='h', tz='UTC')
    
    # Reemplazar timestamps
    df['timestamp'] = timestamps
    
    print(f"   Nuevo fin: {timestamps[-1]}")
    print(f"   Período: {len(df) / 24:.1f} días (~{len(df) / 24 / 30:.1f} meses)")
    
    # Verificar que no hay duplicados
    n_duplicates = df['timestamp'].duplicated().sum()
    print(f"\n✅ Verificación:")
    print(f"   Timestamps únicos: {df['timestamp'].nunique()}")
    print(f"   Duplicados: {n_duplicates}")
    
    # Verificar valores nulos
    print(f"\n⚠️  Valores nulos:")
    nulls = df.isnull().sum()
    for col, count in nulls.items():
        if count > 0:
            print(f"   {col}: {count} ({100*count/len(df):.1f}%)")
    
    # Guardar archivo corregido
    print(f"\n💾 Guardando archivo corregido: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Primeras 10 filas del archivo corregido:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("✅ TIMESTAMPS CORREGIDOS EXITOSAMENTE")
    print("=" * 70)
    print(f"\nArchivo generado: {output_file}")
    
    return df

if __name__ == "__main__":
    corregir_timestamps_clima()
