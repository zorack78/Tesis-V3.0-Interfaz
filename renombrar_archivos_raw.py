"""
Script para renombrar archivos RAW y actualizar columnas
BD_Q_flujo_* → BD_Q_net_*
"""

import pandas as pd
from pathlib import Path

DATA_RAW = Path('data/raw')

print("\n" + "="*80)
print("RENOMBRADO DE ARCHIVOS RAW Y ACTUALIZACIÓN DE COLUMNAS")
print("="*80)

# 1. Actualizar BD_Q_flujo_x_Hr_m3hr.csv (sin LIMPIO)
print("\n[1] Actualizando BD_Q_flujo_x_Hr_m3hr.csv...")
archivo_original = DATA_RAW / 'BD_Q_flujo_x_Hr_m3hr.csv'

if archivo_original.exists():
    df = pd.read_csv(archivo_original)
    print(f"  - Columnas originales: {df.columns.tolist()}")
    
    # Renombrar columna si existe
    if 'Q_flujo_m3hr' in df.columns:
        df.rename(columns={'Q_flujo_m3hr': 'Q_net_m3h'}, inplace=True)
        print(f"  - Columna renombrada: Q_flujo_m3hr → Q_net_m3h")
    
    # Guardar con nuevo nombre
    nuevo_nombre = DATA_RAW / 'BD_Q_net_x_Hr_m3h.csv'
    df.to_csv(nuevo_nombre, index=False)
    print(f"  ✅ Archivo guardado como: BD_Q_net_x_Hr_m3h.csv")
    print(f"  - Columnas finales: {df.columns.tolist()}")
else:
    print(f"  ⚠️ Archivo no encontrado: {archivo_original}")

# 2. Renombrar BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv
print("\n[2] Renombrando BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv...")
archivo_limpio = DATA_RAW / 'BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv'
nuevo_limpio = DATA_RAW / 'BD_Q_net_x_Hr_m3h_LIMPIO.csv'

if archivo_limpio.exists():
    df_limpio = pd.read_csv(archivo_limpio)
    print(f"  - Columnas actuales: {df_limpio.columns.tolist()}")
    
    # Ya debe tener Q_net_m3h, solo renombrar archivo
    df_limpio.to_csv(nuevo_limpio, index=False)
    print(f"  ✅ Archivo guardado como: BD_Q_net_x_Hr_m3h_LIMPIO.csv")
    print(f"  - Registros: {len(df_limpio):,}")
else:
    print(f"  ⚠️ Archivo no encontrado: {archivo_limpio}")

print("\n" + "="*80)
print("✅ RENOMBRADO COMPLETADO")
print("="*80)
print("\nNuevos archivos:")
print("  • data/raw/BD_Q_net_x_Hr_m3h.csv")
print("  • data/raw/BD_Q_net_x_Hr_m3h_LIMPIO.csv")
print("\nBackups disponibles:")
print("  • data/raw/BD_Q_flujo_x_Hr_m3hr.csv.backup")
print("  • data/raw/BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv.backup2")
print("="*80)
