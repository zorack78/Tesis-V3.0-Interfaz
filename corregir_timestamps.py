#!/usr/bin/env python3
"""
Script para corregir timestamps en archivo de volumen.
Reconstruye timestamps con frecuencia horaria desde 2024-01-01.
"""

import pandas as pd
from pathlib import Path

def corregir_timestamps():
    print("=" * 70)
    print("CORRECCIÓN DE TIMESTAMPS")
    print("=" * 70)
    
    # Cargar datos
    input_file = Path("data/raw/BD_VolTotal_X_Hr_m3_UTC.csv")
    output_file = Path("data/raw/BD_VolTotal_X_Hr_m3_UTC_CORREGIDO.csv")
    
    print(f"\n📁 Cargando: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"   Total registros: {len(df)}")
    
    # Reconstruir timestamps con frecuencia horaria
    print(f"\n🔧 Reconstruyendo timestamps...")
    print(f"   Inicio: 2024-01-01 00:00:00 UTC")
    
    # Crear timestamps horarios desde 2024-01-01
    start_date = '2024-01-01 00:00:00'
    timestamps = pd.date_range(start=start_date, periods=len(df), freq='h', tz='UTC')
    
    # Reemplazar timestamps
    df['timestamp'] = timestamps
    
    print(f"   Fin: {timestamps[-1]}")
    print(f"   Período: {len(df) / 24:.1f} días (~{len(df) / 24 / 30:.1f} meses)")
    
    # Verificar que no hay duplicados
    n_duplicates = df['timestamp'].duplicated().sum()
    print(f"\n✅ Verificación:")
    print(f"   Timestamps únicos: {df['timestamp'].nunique()}")
    print(f"   Duplicados: {n_duplicates}")
    
    # Guardar archivo corregido
    print(f"\n💾 Guardando archivo corregido: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Primeras 10 filas del archivo corregido:")
    print(df.head(10))
    
    print(f"\n📊 Últimas 10 filas:")
    print(df.tail(10))
    
    print("\n" + "=" * 70)
    print("✅ TIMESTAMPS CORREGIDOS EXITOSAMENTE")
    print("=" * 70)
    print(f"\nArchivo generado: {output_file}")
    print("\n⚠️  SIGUIENTE PASO:")
    print(f"   Renombrar/reemplazar el archivo original con el corregido")
    
    return df

if __name__ == "__main__":
    corregir_timestamps()
