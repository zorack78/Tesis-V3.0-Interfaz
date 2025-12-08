"""
Conversión de Timestamps UTC a Hora Local de Valparaíso, Chile

Este script convierte los timestamps UTC del archivo de clima a la hora local
de Valparaíso, considerando los cambios de horario de verano/invierno.

Cambios horarios durante el período 01/01/2024 - 30/09/2025:
- 01/01/2024 - 06/04/2024: CLST (Verano) UTC-3
- 07/04/2024 - 07/09/2024: CLT (Estándar) UTC-4  
- 08/09/2024 - 05/04/2025: CLST (Verano) UTC-3
- 06/04/2025 - 06/09/2025: CLT (Estándar) UTC-4
- 07/09/2025 - 30/09/2025: CLST (Verano) UTC-3

IMPORTANTE: Los valores se desplazan según los cambios horarios reales,
manteniendo la zona horaria de Chile/Continental en todo momento.

Autor: Sistema de Análisis de Demanda de Agua
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import pytz

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATA_RAW = Path('data/raw')
INPUT_FILE = DATA_RAW / 'BD_Clima2024a202509_UTC.csv'
OUTPUT_FILE = DATA_RAW / 'BD_Clima2024a202509_Local.csv'

print("="*80)
print("CONVERSIÓN UTC → HORA LOCAL VALPARAÍSO")
print("="*80)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("\n[1/3] Cargando datos UTC...")

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

print(f"   ✓ Registros cargados: {len(df):,}")
print(f"   ✓ Período UTC: {df['timestamp'].min()} a {df['timestamp'].max()}")
print(f"   ✓ Variables: {', '.join(df.columns.tolist())}")

# ============================================================================
# CONVERSIÓN A HORA LOCAL DE CHILE
# ============================================================================

print("\n[2/3] Convirtiendo a hora local de Valparaíso...")

# Definir zona horaria de Chile/Continental
chile_tz = pytz.timezone('Chile/Continental')

# Convertir timestamps UTC a hora local de Chile
# pytz maneja automáticamente los cambios de horario de verano/invierno
df['timestamp_local'] = df['timestamp'].dt.tz_convert(chile_tz)

# Verificar los cambios de horario
print("\n   Verificación de cambios horarios:")
print("   " + "-"*70)

# Puntos clave de cambio
fechas_clave = [
    ('2024-01-01 00:00:00+00:00', 'Inicio del período (verano)'),
    ('2024-04-06 03:00:00+00:00', 'Antes del cambio (fin verano)'),
    ('2024-04-06 04:00:00+00:00', 'Después del cambio (inicio invierno)'),
    ('2024-09-07 03:00:00+00:00', 'Antes del cambio (fin invierno)'),
    ('2024-09-07 04:00:00+00:00', 'Después del cambio (inicio verano)'),
    ('2025-04-05 03:00:00+00:00', 'Antes del cambio (fin verano)'),
    ('2025-04-05 04:00:00+00:00', 'Después del cambio (inicio invierno)'),
    ('2025-09-06 03:00:00+00:00', 'Antes del cambio (fin invierno)'),
    ('2025-09-06 04:00:00+00:00', 'Después del cambio (inicio verano)'),
    ('2025-09-30 23:00:00+00:00', 'Fin del período (verano)')
]

for fecha_utc_str, descripcion in fechas_clave:
    fecha_utc = pd.Timestamp(fecha_utc_str)
    
    # Buscar la fila más cercana
    idx = (df['timestamp'] - fecha_utc).abs().idxmin()
    
    if pd.notna(idx):
        utc_ts = df.loc[idx, 'timestamp']
        local_ts = df.loc[idx, 'timestamp_local']
        offset = local_ts.utcoffset().total_seconds() / 3600
        zona = local_ts.strftime('%Z')
        
        print(f"   {descripcion}")
        print(f"     UTC:   {utc_ts}")
        print(f"     Local: {local_ts} ({zona}, UTC{offset:+.0f})")
        print()

# ============================================================================
# PREPARAR DATOS DE SALIDA
# ============================================================================

print("\n[3/3] Preparando archivo de salida...")

# Crear DataFrame de salida con timestamp local sin zona horaria explícita
# (para que sea más fácil de usar en análisis posteriores)
df_output = df.copy()
df_output['timestamp'] = df_output['timestamp_local'].dt.tz_localize(None)
df_output = df_output.drop(columns=['timestamp_local'])

# Verificar que no hay gaps o duplicados
duplicados = df_output['timestamp'].duplicated().sum()
print(f"   ✓ Timestamps duplicados: {duplicados}")

if duplicados > 0:
    print(f"   ⚠ ADVERTENCIA: Se encontraron {duplicados} timestamps duplicados")
    print("   (Esto es normal en el cambio de horario de invierno a verano)")
    print("   Manteniendo el primer registro de cada timestamp duplicado...")
    df_output = df_output.drop_duplicates(subset=['timestamp'], keep='first')

# Verificar gaps (saltos > 1 hora)
time_diffs = df_output['timestamp'].diff()
gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
print(f"   ✓ Gaps mayores a 1 hora: {len(gaps)}")

if len(gaps) > 0:
    print("   ⚠ Gaps detectados (esperado en cambios de horario):")
    for idx, gap in gaps.items():
        print(f"     - En {df_output.loc[idx, 'timestamp']}: gap de {gap.total_seconds()/3600:.1f} horas")

# Guardar archivo
df_output.to_csv(OUTPUT_FILE, index=False)

print(f"\n   ✓ Archivo guardado: {OUTPUT_FILE}")
print(f"   ✓ Registros finales: {len(df_output):,}")
print(f"   ✓ Período Local: {df_output['timestamp'].min()} a {df_output['timestamp'].max()}")

# ============================================================================
# RESUMEN Y COMPARACIÓN
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE LA CONVERSIÓN")
print("="*80)

# Estadísticas por período horario
periodos = [
    ('2024-01-01', '2024-04-06', 'CLST (Verano)', 'UTC-3'),
    ('2024-04-07', '2024-09-07', 'CLT (Estándar)', 'UTC-4'),
    ('2024-09-08', '2025-04-05', 'CLST (Verano)', 'UTC-3'),
    ('2025-04-06', '2025-09-06', 'CLT (Estándar)', 'UTC-4'),
    ('2025-09-07', '2025-09-30', 'CLST (Verano)', 'UTC-3')
]

print("\nDistribución de registros por período horario:")
print("-" * 80)
print(f"{'Período':<30} {'Zona':<16} {'Offset':<8} {'N° Registros':<15}")
print("-" * 80)

for inicio, fin, zona, offset in periodos:
    mask = (df_output['timestamp'] >= inicio) & (df_output['timestamp'] <= fin + ' 23:59:59')
    n_registros = mask.sum()
    print(f"{inicio} - {fin:<12} {zona:<16} {offset:<8} {n_registros:>10,}")

print("-" * 80)
print(f"{'TOTAL':<30} {'':<16} {'':<8} {len(df_output):>10,}")

# Ejemplo de registros transformados
print("\n" + "="*80)
print("EJEMPLOS DE TRANSFORMACIÓN (primeros 10 registros)")
print("="*80)

# Recargar archivo original para comparación
df_original = pd.read_csv(INPUT_FILE, nrows=10)
df_nuevo = pd.read_csv(OUTPUT_FILE, nrows=10)

print("\nArchivo ORIGINAL (UTC):")
print(df_original[['timestamp', 'temp', 'HR', 'mmhr']].to_string(index=False))

print("\nArchivo NUEVO (Hora Local Valparaíso):")
print(df_nuevo[['timestamp', 'temp', 'HR', 'mmhr']].to_string(index=False))

print("\n" + "="*80)
print("ANÁLISIS DE OFFSET APLICADO")
print("="*80)

# Comparar algunas fechas específicas
print("\nComparación UTC vs Local (muestra):")
print("-" * 80)
print(f"{'Fecha UTC':<30} {'Fecha Local':<30} {'Offset':<10}")
print("-" * 80)

df_original_full = pd.read_csv(INPUT_FILE)
df_original_full['timestamp'] = pd.to_datetime(df_original_full['timestamp'], utc=True)

indices_muestra = [0, 1000, 5000, 10000, 15000, len(df_output)-1]
for idx in indices_muestra:
    if idx < len(df_output):
        utc_ts = df_original_full.loc[idx, 'timestamp']
        local_ts = pd.to_datetime(df_output.loc[idx, 'timestamp'])
        
        # Convertir UTC a naive para comparación simple
        utc_naive = utc_ts.tz_localize(None)
        
        # Calcular diferencia de horas
        offset_hours = (local_ts - utc_naive).total_seconds() / 3600
        
        utc_str = str(utc_ts)[:19]  # Sin timezone info en display
        local_str = str(local_ts)[:19]
        
        print(f"{utc_str:<30} {local_str:<30} {offset_hours:>+5.0f}h")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"\nArchivo generado: {OUTPUT_FILE}")
print("\nNOTA: Los valores climáticos (temp, HR, mmhr) no han cambiado,")
print("solo los timestamps fueron ajustados a la hora local de Valparaíso.")
print("\nEste archivo está listo para ser usado en análisis posteriores")
print("considerando la zona horaria local de Chile.")
print("="*80)
