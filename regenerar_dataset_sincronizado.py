"""
Regenera el dataset sincronizando correctamente clima y producción.
Todos los datos deben empezar a las 00:00 del 01/01/2024 hora local Chile.
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("REGENERACIÓN DE DATASET SINCRONIZADO")
print("=" * 80)

# Rutas
DATA_RAW = Path('data/raw')
DATA_PROCESSED = Path('data/processed')

# 1. Cargar datos de producción (empiezan a las 00:00 del 01/01/2024)
print("\n1️⃣ Cargando datos de producción...")
df_qnet = pd.read_csv(DATA_RAW / 'BD_Q_net_x_Hr_m3h_LIMPIO.csv')
df_qnet['timestamp'] = pd.to_datetime(df_qnet['timestamp'], utc=True).dt.tz_localize(None)
print(f"   Q_net: {len(df_qnet)} registros desde {df_qnet['timestamp'].min()}")

df_qin = pd.read_csv(DATA_RAW / 'BD_Qin_m3_Local.csv')
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'], utc=True).dt.tz_localize(None)
print(f"   Qin: {len(df_qin)} registros desde {df_qin['timestamp'].min()}")

df_voltotal = pd.read_csv(DATA_RAW / 'BD_VolTotal_X_Hr_m3_Local.csv')
df_voltotal['timestamp'] = pd.to_datetime(df_voltotal['timestamp'], utc=True).dt.tz_localize(None)
print(f"   VolTotal: {len(df_voltotal)} registros desde {df_voltotal['timestamp'].min()}")

# 2. Cargar datos de clima (empiezan a las 21:00 del 31/12/2023)
print("\n2️⃣ Cargando datos de clima...")
df_clima = pd.read_csv(DATA_RAW / 'BD_Clima2024a202509_Local.csv')
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
print(f"   Clima original: {len(df_clima)} registros desde {df_clima['timestamp'].min()}")

# 3. Filtrar clima para que empiece a las 00:00 del 01/01/2024
fecha_inicio = pd.Timestamp('2024-01-01 00:00:00')
df_clima_sync = df_clima[df_clima['timestamp'] >= fecha_inicio].copy()
print(f"   Clima sincronizado: {len(df_clima_sync)} registros desde {df_clima_sync['timestamp'].min()}")
print(f"   ⚠️  Se eliminaron {len(df_clima) - len(df_clima_sync)} registros del clima (21:00-23:00 del 31/12/2023)")

# 4. Verificar sincronización
print("\n3️⃣ Verificando sincronización temporal...")
min_prod = df_qnet['timestamp'].min()
max_prod = df_qnet['timestamp'].max()
min_clima = df_clima_sync['timestamp'].min()
max_clima = df_clima_sync['timestamp'].max()

print(f"   Producción: {min_prod} → {max_prod}")
print(f"   Clima:      {min_clima} → {max_clima}")

if min_prod == min_clima:
    print("   ✅ Inicio sincronizado correctamente")
else:
    print(f"   ❌ ERROR: Desfase de {min_prod - min_clima}")

# 5. Merge de datos
print("\n4️⃣ Combinando datasets...")
df_base = df_qnet.copy()  # Ya tiene Q_net_m3h

# Merge con Qin
df_base = df_base.merge(
    df_qin[['timestamp', 'Qin']],
    on='timestamp',
    how='left'
)
df_base = df_base.rename(columns={'Qin': 'Qin_m3_hr'})
print("   ✅ Qin añadido")

# Merge con volumen total
df_base = df_base.merge(
    df_voltotal[['timestamp', ' Volumen_Total_m3']],
    on='timestamp',
    how='left'
)
print("   ✅ Volumen total añadido")

# Merge con clima
df_base = df_base.merge(
    df_clima_sync[['timestamp', 'temp', 'HR', 'mmhr']],
    on='timestamp',
    how='left'
)
print("   ✅ Clima añadido")

# 6. Verificar registros completos
print("\n5️⃣ Verificando integridad de datos...")
print(f"   Total registros: {len(df_base)}")
print(f"   Registros sin clima: {df_base['temp'].isna().sum()}")
print(f"   Registros sin volumen: {df_base[' Volumen_Total_m3'].isna().sum()}")
print(f"   Registros sin Qin: {df_base['Qin_m3_hr'].isna().sum()}")

# 7. Añadir features temporales básicos
print("\n6️⃣ Añadiendo features temporales...")
df_base['hora'] = df_base['timestamp'].dt.hour
df_base['dia_semana'] = df_base['timestamp'].dt.dayofweek
df_base['mes'] = df_base['timestamp'].dt.month
df_base['dia'] = df_base['timestamp'].dt.day
df_base['anio'] = df_base['timestamp'].dt.year
print("   ✅ Features temporales añadidos")

# 8. Guardar dataset sincronizado
print("\n7️⃣ Guardando dataset sincronizado...")
output_path = DATA_PROCESSED / 'dataset_completo_sincronizado.csv'
df_base.to_csv(output_path, index=False)
print(f"   ✅ Dataset guardado: {output_path}")
print(f"   📊 Dimensiones: {df_base.shape}")

# 10. Resumen final
print("\n" + "=" * 80)
print("✅ DATASET REGENERADO CORRECTAMENTE")
print("=" * 80)
print(f"📅 Período: {df_base['timestamp'].min()} → {df_base['timestamp'].max()}")
print(f"📊 Total registros: {len(df_base):,}")
print(f"🌡️  Registros con clima: {df_base['temp'].notna().sum():,} ({df_base['temp'].notna().sum()/len(df_base)*100:.1f}%)")
print(f"💧 Registros con volumen: {df_base[' Volumen_Total_m3'].notna().sum():,}")
print("\n🎯 Próximo paso: Ejecutar generador_features_completo.py para añadir features avanzados")
print("=" * 80)
