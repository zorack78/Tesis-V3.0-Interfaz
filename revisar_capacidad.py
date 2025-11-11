#!/usr/bin/env python3
"""
Script para revisar estructura del archivo de capacidad de tanques.
"""

import pandas as pd
from pathlib import Path


def revisar_capacidad():
    file_path = Path("data/raw/BD_Capacidad_89Tks_m3.csv")
    
    print("=" * 70)
    print(f"REVISIÓN: {file_path.name}")
    print("=" * 70)
    
    # Cargar datos
    df = pd.read_csv(file_path)
    
    print(f"\n📊 Estructura:")
    print(f"   Filas: {len(df)}")
    print(f"   Columnas: {len(df.columns)}")
    print(f"   Columnas: {df.columns.tolist()}")
    
    print(f"\n📋 Primeras 10 filas:")
    print(df.head(10))
    
    print(f"\n📋 Últimas 5 filas:")
    print(df.tail())
    
    # Verificar si hay timestamp
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'fecha' in col.lower()]
    if timestamp_cols:
        print(f"\n🕐 Columnas de tiempo detectadas: {timestamp_cols}")
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col])
            print(f"\n   {col}:")
            print(f"   - Total registros: {len(df)}")
            print(f"   - Únicos: {df[col].nunique()}")
            print(f"   - Duplicados: {df[col].duplicated().sum()}")
            print(f"   - Rango: {df[col].min()} a {df[col].max()}")
    else:
        print(f"\n⚠️  No se detectaron columnas de timestamp")
    
    # Verificar valores nulos
    print(f"\n❓ Valores nulos por columna:")
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        for col, count in nulls.items():
            if count > 0:
                print(f"   {col}: {count} ({100*count/len(df):.1f}%)")
    else:
        print("   ✅ Sin valores nulos")
    
    # Estadísticas básicas de columnas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Estadísticas de columnas numéricas:")
        print(df[numeric_cols].describe())
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    revisar_capacidad()
