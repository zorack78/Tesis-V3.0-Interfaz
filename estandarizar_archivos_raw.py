"""
Script para estandarizar nombres de columnas en archivos RAW
============================================================
- BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv: Q_flujo_m3hr → Q_net_m3h
- BD_Qin_m3_Local.csv: Qin_m3_hr → Qin_m3h
"""

import pandas as pd
from pathlib import Path
import shutil

print("="*80)
print("ESTANDARIZACIÓN DE ARCHIVOS RAW")
print("="*80)

DATA_RAW = Path('data/raw')

# 1. Backup (por seguridad)
print("\n[1] Creando backups...")
archivos = [
    'BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv',
    'BD_Qin_m3_Local.csv'
]

for archivo in archivos:
    origen = DATA_RAW / archivo
    backup = DATA_RAW / f"{archivo}.backup"
    if origen.exists():
        shutil.copy2(origen, backup)
        print(f"  ✅ Backup: {archivo}.backup")
    else:
        print(f"  ⚠️ No encontrado: {archivo}")

# 2. Renombrar columnas en BD_Q_flujo
print("\n[2] Estandarizando BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv...")
archivo_qflujo = DATA_RAW / 'BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv'

if archivo_qflujo.exists():
    df = pd.read_csv(archivo_qflujo)
    print(f"  - Columnas originales: {df.columns.tolist()}")
    
    # Renombrar
    if 'Q_flujo_m3hr' in df.columns:
        df.rename(columns={'Q_flujo_m3hr': 'Q_net_m3h'}, inplace=True)
        print(f"  - Columnas nuevas: {df.columns.tolist()}")
        
        # Guardar
        df.to_csv(archivo_qflujo, index=False)
        print(f"  ✅ Archivo actualizado: Q_flujo_m3hr → Q_net_m3h")
    else:
        print(f"  ⚠️ Columna Q_flujo_m3hr no encontrada")
else:
    print(f"  ❌ Archivo no existe")

# 3. Renombrar columnas en BD_Qin
print("\n[3] Estandarizando BD_Qin_m3_Local.csv...")
archivo_qin = DATA_RAW / 'BD_Qin_m3_Local.csv'

if archivo_qin.exists():
    df = pd.read_csv(archivo_qin)
    print(f"  - Columnas originales: {df.columns.tolist()}")
    
    # Renombrar (estandarizar unidad)
    cambios = {}
    if 'Qin_m3_hr' in df.columns:
        cambios['Qin_m3_hr'] = 'Qin_m3h'
    if 'Qin_m3hr' in df.columns:
        cambios['Qin_m3hr'] = 'Qin_m3h'
    
    if cambios:
        df.rename(columns=cambios, inplace=True)
        print(f"  - Columnas nuevas: {df.columns.tolist()}")
        
        # Guardar
        df.to_csv(archivo_qin, index=False)
        print(f"  ✅ Archivo actualizado: {list(cambios.keys())[0]} → Qin_m3h")
    else:
        print(f"  ℹ️ No hay cambios necesarios (ya estandarizado)")
else:
    print(f"  ❌ Archivo no existe")

print("\n" + "="*80)
print("✅ ESTANDARIZACIÓN COMPLETADA")
print("="*80)
print("\nBackups guardados en data/raw/*.backup")
print("Para revertir: renombra .backup a .csv")
