"""
Corrige el formato de timestamp del archivo de clima para que coincida
con el formato de los otros archivos (con zona horaria +00:00).
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("CORRECCIÓN DE TIMESTAMP - ARCHIVO CLIMA")
print("=" * 80)

# Rutas
archivo_clima = Path('data/raw/BD_Clima2024a202509_Local.csv')
archivo_backup = Path('data/raw/BD_Clima2024a202509_Local_BACKUP.csv')

# 1. Hacer backup del archivo original
print("\n1️⃣ Creando backup del archivo original...")
import shutil
shutil.copy2(archivo_clima, archivo_backup)
print(f"   ✅ Backup guardado: {archivo_backup}")

# 2. Cargar archivo de clima
print("\n2️⃣ Cargando archivo de clima...")
df_clima = pd.read_csv(archivo_clima)
print(f"   📊 Total registros: {len(df_clima)}")
print(f"   📅 Timestamp original ejemplo: {df_clima['timestamp'].iloc[0]}")

# 3. Convertir timestamp a formato con zona horaria
print("\n3️⃣ Convirtiendo timestamps...")
# Parsear como datetime sin zona horaria
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
# Agregar zona horaria UTC (ya que están en hora local pero queremos formato +00:00)
df_clima['timestamp'] = df_clima['timestamp'].dt.tz_localize('UTC')
print(f"   📅 Timestamp corregido ejemplo: {df_clima['timestamp'].iloc[0]}")

# 4. Eliminar las primeras 3 horas (21:00, 22:00, 23:00 del 31/12/2023)
print("\n4️⃣ Eliminando registros previos a 2024-01-01 00:00:00...")
fecha_inicio = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
registros_antes = len(df_clima)
df_clima = df_clima[df_clima['timestamp'] >= fecha_inicio].copy()
registros_eliminados = registros_antes - len(df_clima)
print(f"   ⚠️  Eliminados {registros_eliminados} registros previos")
print(f"   📊 Registros restantes: {len(df_clima)}")
print(f"   📅 Nuevo inicio: {df_clima['timestamp'].min()}")

# 5. Guardar archivo corregido
print("\n5️⃣ Guardando archivo corregido...")
df_clima.to_csv(archivo_clima, index=False)
print(f"   ✅ Archivo actualizado: {archivo_clima}")

# 6. Verificación
print("\n6️⃣ Verificando corrección...")
df_check = pd.read_csv(archivo_clima)
print(f"   📅 Primer timestamp: {df_check['timestamp'].iloc[0]}")
print(f"   📅 Último timestamp: {df_check['timestamp'].iloc[-1]}")
print(f"   📊 Total registros: {len(df_check)}")

# 7. Comparar con archivo de producción
print("\n7️⃣ Comparando con archivo de producción...")
df_qnet = pd.read_csv('data/raw/BD_Q_net_x_Hr_m3h_LIMPIO.csv')
print(f"   Q_net primer timestamp: {df_qnet['timestamp'].iloc[0]}")
print(f"   Clima primer timestamp:   {df_check['timestamp'].iloc[0]}")

if df_qnet['timestamp'].iloc[0] == df_check['timestamp'].iloc[0]:
    print("   ✅ Timestamps coinciden perfectamente")
else:
    print("   ⚠️  Los timestamps aún difieren")

print("\n" + "=" * 80)
print("✅ CORRECCIÓN COMPLETADA")
print("=" * 80)
print(f"📁 Archivo original respaldado en: {archivo_backup}")
print(f"📁 Archivo corregido: {archivo_clima}")
print("=" * 80)
