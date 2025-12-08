"""
Verificación rápida de nomenclatura estandarizada
"""
import pandas as pd

print("="*80)
print("VERIFICACION DE NOMENCLATURA ESTANDARIZADA")
print("="*80)

# 1. Verificar BD_Q_flujo
print("\n[1] BD_Q_net_x_Hr_m3h_LIMPIO.csv:")
df_qnet = pd.read_csv('data/raw/BD_Q_net_x_Hr_m3h_LIMPIO.csv')
print(f"  Columnas: {df_qnet.columns.tolist()}")
print(f"  Registros: {len(df_qnet):,}")
if 'Q_net_m3h' in df_qnet.columns:
    print(f"  Rango Q_net: {df_qnet['Q_net_m3h'].min():.1f} a {df_qnet['Q_net_m3h'].max():.1f} m³/h")
    print("  ✅ Columna Q_net_m3h encontrada")
else:
    print("  ❌ ERROR: Q_net_m3h no encontrada")

# 2. Verificar BD_Qin
print("\n[2] BD_Qin_m3_Local.csv:")
df_qin = pd.read_csv('data/raw/BD_Qin_m3_Local.csv')
print(f"  Columnas: {df_qin.columns.tolist()}")
print(f"  Registros: {len(df_qin):,}")
if 'Qin' in df_qin.columns:
    print(f"  Rango Qin: {df_qin['Qin'].min():.1f} a {df_qin['Qin'].max():.1f} m³/h")
    print("  ✅ Columna Qin encontrada")
else:
    print("  ❌ ERROR: Qin no encontrada")

# 3. Test de cálculo
print("\n[3] Test de balance hídrico:")
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
df_qnet['timestamp'] = pd.to_datetime(df_qnet['timestamp'])
df_merged = df_qin.merge(df_qnet, on='timestamp', how='inner')
df_merged['Qout'] = df_merged['Qin'] - df_merged['Q_net_m3h']

print(f"  Registros combinados: {len(df_merged):,}")
print(f"  Qout promedio: {df_merged['Qout'].mean():.1f} m³/h")
print(f"  Qout rango: {df_merged['Qout'].min():.1f} a {df_merged['Qout'].max():.1f} m³/h")

# Verificar que no hay valores anómalos
qout_negativos = (df_merged['Qout'] < 0).sum()
if qout_negativos == 0:
    print("  ✅ No hay Qout negativos")
else:
    print(f"  ⚠️ {qout_negativos} registros con Qout negativo")

print("\n" + "="*80)
print("✅ VERIFICACION COMPLETADA")
print("="*80)
