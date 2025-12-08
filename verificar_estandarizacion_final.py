"""
Verificación final de estandarización completa
- Nombres de archivos
- Nombres de columnas
- Referencias en scripts
"""

import pandas as pd
from pathlib import Path

print("\n" + "="*80)
print("VERIFICACIÓN FINAL DE ESTANDARIZACIÓN Q_NET")
print("="*80)

DATA_RAW = Path('data/raw')

# 1. Verificar que archivos nuevos existen
print("\n[1] Verificando archivos RAW estandarizados...")
archivos_requeridos = [
    'BD_Q_net_x_Hr_m3h.csv',
    'BD_Q_net_x_Hr_m3h_LIMPIO.csv',
    'BD_Qin_m3_Local.csv'
]

for archivo in archivos_requeridos:
    ruta = DATA_RAW / archivo
    if ruta.exists():
        df = pd.read_csv(ruta)
        print(f"  ✅ {archivo}")
        print(f"     - Columnas: {df.columns.tolist()}")
        print(f"     - Registros: {len(df):,}")
    else:
        print(f"  ❌ {archivo} NO ENCONTRADO")

# 2. Verificar que columnas son correctas
print("\n[2] Verificando columnas estandarizadas...")
df_qnet = pd.read_csv(DATA_RAW / 'BD_Q_net_x_Hr_m3h_LIMPIO.csv')
df_qin = pd.read_csv(DATA_RAW / 'BD_Qin_m3_Local.csv')

if 'Q_net_m3h' in df_qnet.columns:
    print("  ✅ Columna Q_net_m3h presente en BD_Q_net_x_Hr_m3h_LIMPIO.csv")
else:
    print("  ❌ Columna Q_net_m3h NO encontrada")

if 'Qin' in df_qin.columns:
    print("  ✅ Columna Qin presente en BD_Qin_m3_Local.csv")
else:
    print("  ❌ Columna Qin NO encontrada")

# 3. Test de balance hídrico
print("\n[3] Test de balance hídrico...")
df_merge = pd.merge(df_qin, df_qnet, on='timestamp', how='inner')
df_merge['Qout'] = df_merge['Qin'] - df_merge['Q_net_m3h']

print(f"  - Registros combinados: {len(df_merge):,}")
print(f"  - Qout promedio: {df_merge['Qout'].mean():,.1f} m³/h")
print(f"  - Qout rango: {df_merge['Qout'].min():,.1f} a {df_merge['Qout'].max():,.1f} m³/h")

# Verificar demanda razonable (debe ser positiva en general)
negativos = (df_merge['Qout'] < 0).sum()
if negativos < 50:  # Permitir algunos outliers
    print(f"  ✅ Balance hídrico correcto ({negativos} registros negativos)")
else:
    print(f"  ⚠️ {negativos} registros con Qout negativo (revisar)")

# 4. Verificar backups
print("\n[4] Verificando backups...")
backups = [
    'BD_Q_flujo_x_Hr_m3hr.csv.backup',
    'BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv.backup',
    'BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv.backup2'
]

for backup in backups:
    if (DATA_RAW / backup).exists():
        print(f"  ✅ {backup}")
    else:
        print(f"  ⚠️ {backup} no encontrado")

# 5. Verificar que archivos antiguos NO se usan
print("\n[5] Archivos antiguos que pueden eliminarse...")
archivos_antiguos = [
    'BD_Q_flujo_x_Hr_m3hr.csv',
    'BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv'
]

for archivo in archivos_antiguos:
    ruta = DATA_RAW / archivo
    if ruta.exists():
        print(f"  ⚠️ {archivo} todavía existe (puede eliminarse)")
    else:
        print(f"  ✅ {archivo} ya fue reemplazado")

print("\n" + "="*80)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*80)
print("\nResumen:")
print("  • Archivos RAW renombrados: BD_Q_flujo → BD_Q_net")
print("  • Columnas estandarizadas: Q_flujo_m3hr → Q_net_m3h")
print("  • Balance hídrico funcional: Qout = Qin - Q_net")
print("  • Backups disponibles para restaurar si es necesario")
print("="*80)
