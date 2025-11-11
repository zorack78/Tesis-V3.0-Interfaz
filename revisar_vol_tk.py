#!/usr/bin/env python3
"""
Script para revisar estructura del archivo de volumen por tanque.
"""

import pandas as pd
from pathlib import Path


def revisar_vol_tk():
    file_path = Path("data/raw/Vol_X_TK_Hr_m3_UTC.csv")
    
    print("=" * 70)
    print(f"REVISIÓN: {file_path.name}")
    print("=" * 70)
    
    # Cargar datos
    df = pd.read_csv(file_path)
    
    print(f"\n📊 Estructura:")
    print(f"   Filas: {len(df)}")
    print(f"   Columnas: {len(df.columns)}")
    print(f"   Columnas (primeras 10): {df.columns.tolist()[:10]}")
    if len(df.columns) > 10:
        print(f"   ... y {len(df.columns) - 10} columnas más")
    
    print(f"\n📋 Primeras 5 filas (primeras 5 columnas):")
    print(df.iloc[:5, :5])
    
    # Verificar timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"\n🕐 Análisis de Timestamps:")
        print(f"   Total registros: {len(df)}")
        print(f"   Timestamps únicos: {df['timestamp'].nunique()}")
        print(f"   Duplicados: {df['timestamp'].duplicated().sum()}")
        print(f"   Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
        
        # Verificar frecuencia
        if len(df) > 1:
            time_diff = df['timestamp'].diff().mode()
            if len(time_diff) > 0:
                print(f"   Frecuencia modal: {time_diff.iloc[0]}")
    
    # Verificar valores nulos
    print(f"\n❓ Valores nulos:")
    nulls = df.isnull().sum()
    total_nulls = nulls.sum()
    if total_nulls > 0:
        print(f"   Total valores nulos: {total_nulls}")
        null_cols = nulls[nulls > 0]
        print(f"   Columnas con nulos: {len(null_cols)}")
        if len(null_cols) <= 10:
            for col, count in null_cols.items():
                print(f"   {col}: {count} ({100*count/len(df):.1f}%)")
        else:
            print(f"   (Mostrando primeras 10 columnas con más nulos)")
            for col, count in null_cols.nlargest(10).items():
                print(f"   {col}: {count} ({100*count/len(df):.1f}%)")
    else:
        print("   ✅ Sin valores nulos")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    revisar_vol_tk()
