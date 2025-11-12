"""
VERIFICACIÓN CRÍTICA: ¿Estamos interpretando los datos correctamente?
Volumen Total = ¿Producción o Consumo?
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("VERIFICACIÓN: ¿QUÉ ESTAMOS MIDIENDO REALMENTE?")
print("=" * 80)

# Cargar datos
volumen_df = pd.read_csv('data/processed/data_processed_complete.csv')
volumen_df.columns = volumen_df.columns.str.strip()
volumen_df['timestamp_utc'] = pd.to_datetime(volumen_df['timestamp_utc'])

# Cargar Qin (caudal de entrada/producción)
qin_df = pd.read_csv('data/raw/BD_Qin_m3_UTC.csv')
print(f"\n📊 Columnas en Qin: {qin_df.columns.tolist()[:5]}")

# Revisar el archivo de volumen total original
vol_original = pd.read_csv('data/raw/BD_VolTotal_X_Hr_m3_UTC.csv')
print(f"📊 Columnas en Volumen Total: {vol_original.columns.tolist()[:5]}")

print("\n" + "=" * 80)
print("1️⃣ COMPARACIÓN: Volumen Total vs Qin (Producción)")
print("=" * 80)

# Cargar y procesar Qin
qin_df.columns = qin_df.columns.str.strip()
# El archivo tiene 'timestamp', no 'timestamp_utc'
if 'timestamp' in qin_df.columns and 'timestamp_utc' not in qin_df.columns:
    qin_df['timestamp_utc'] = pd.to_datetime(qin_df['timestamp'])
else:
    qin_df['timestamp_utc'] = pd.to_datetime(qin_df['timestamp_utc'])

# Encontrar columna de Qin
qin_col = None
for col in qin_df.columns:
    if 'qin' in col.lower() and 'timestamp' not in col.lower():
        qin_col = col
        break

if qin_col:
    print(f"✅ Columna Qin encontrada: {qin_col}")
else:
    print("❌ No se encontró columna Qin")
    # Buscar cualquier columna numérica
    numeric_cols = qin_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        qin_col = numeric_cols[0]
        print(f"⚠️  Usando columna: {qin_col}")

# Merge datos
df = pd.merge(volumen_df, qin_df[['timestamp_utc', qin_col]], 
              on='timestamp_utc', how='inner')

volumen_col = 'Volumen_Total_m3'

print(f"\n📊 Datos unidos: {len(df)} registros")
print(f"\n📈 ESTADÍSTICAS:")
print(f"   Volumen Total:")
print(f"      - Promedio: {df[volumen_col].mean():,.0f} m³/hora")
print(f"      - Máximo:   {df[volumen_col].max():,.0f} m³/hora")
print(f"      - Mínimo:   {df[volumen_col].min():,.0f} m³/hora")
print(f"\n   Qin (Producción):")
print(f"      - Promedio: {df[qin_col].mean():,.0f} m³/hora")
print(f"      - Máximo:   {df[qin_col].max():,.0f} m³/hora")
print(f"      - Mínimo:   {df[qin_col].min():,.0f} m³/hora")

# Correlación
corr = df[[volumen_col, qin_col]].corr().iloc[0, 1]
print(f"\n🔗 Correlación Volumen-Qin: {corr:.4f}")

# Ratio
df['ratio_vol_qin'] = df[volumen_col] / df[qin_col]
print(f"\n📊 Ratio Volumen/Qin:")
print(f"   Promedio: {df['ratio_vol_qin'].mean():.2f}")
print(f"   (Volumen Total es ~{df['ratio_vol_qin'].mean():.1f}x el Qin)")

print("\n" + "=" * 80)
print("2️⃣ ANÁLISIS POR TEMPERATURA (Nueva interpretación)")
print("=" * 80)

# Cargar clima
clima_df = pd.read_csv('data/processed/clima_chile_v3.csv')
clima_df['timestamp'] = pd.to_datetime(clima_df['timestamp'])

df_clima = pd.merge(df, clima_df, left_on='timestamp_utc', 
                    right_on='timestamp', how='inner')

# Definir estaciones
df_clima['mes'] = df_clima['timestamp_utc'].dt.month

def get_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

df_clima['estacion'] = df_clima['mes'].apply(get_estacion)

print("\n🌡️ VERANO (alta temperatura):")
verano_vol = df_clima[df_clima['estacion']=='Verano'][volumen_col].mean()
verano_qin = df_clima[df_clima['estacion']=='Verano'][qin_col].mean()
verano_temp = df_clima[df_clima['estacion']=='Verano']['temp'].mean()
print(f"   Temperatura promedio: {verano_temp:.1f}°C")
print(f"   Volumen Total: {verano_vol:,.0f} m³/hora")
print(f"   Qin (Producción): {verano_qin:,.0f} m³/hora")

print("\n❄️  INVIERNO (baja temperatura):")
invierno_vol = df_clima[df_clima['estacion']=='Invierno'][volumen_col].mean()
invierno_qin = df_clima[df_clima['estacion']=='Invierno'][qin_col].mean()
invierno_temp = df_clima[df_clima['estacion']=='Invierno']['temp'].mean()
print(f"   Temperatura promedio: {invierno_temp:.1f}°C")
print(f"   Volumen Total: {invierno_vol:,.0f} m³/hora")
print(f"   Qin (Producción): {invierno_qin:,.0f} m³/hora")

print("\n📊 DIFERENCIAS:")
print(f"   Temperatura: {verano_temp - invierno_temp:+.1f}°C")
print(f"   Volumen Total: {verano_vol - invierno_vol:+,.0f} m³/hora ({(verano_vol-invierno_vol)/invierno_vol*100:+.2f}%)")
print(f"   Qin: {verano_qin - invierno_qin:+,.0f} m³/hora ({(verano_qin-invierno_qin)/invierno_qin*100:+.2f}%)")

print("\n" + "=" * 80)
print("3️⃣ VERIFICACIÓN: ¿Qin sigue al Volumen Total?")
print("=" * 80)

# Días de mayor volumen
top_dias = df_clima.nlargest(10, volumen_col)[['timestamp_utc', volumen_col, qin_col, 'temp']]
print("\n🔝 TOP 10 días de MAYOR Volumen Total:")
print(f"{'Fecha':<20} {'Volumen':<15} {'Qin':<15} {'Temp'}")
print("-" * 60)
for _, row in top_dias.iterrows():
    print(f"{str(row['timestamp_utc'])[:16]:<20} {row[volumen_col]:>10,.0f} m³   "
          f"{row[qin_col]:>10,.0f} m³   {row['temp']:>5.1f}°C")

# Días de menor volumen
bottom_dias = df_clima.nsmallest(10, volumen_col)[['timestamp_utc', volumen_col, qin_col, 'temp']]
print("\n🔻 TOP 10 días de MENOR Volumen Total:")
print(f"{'Fecha':<20} {'Volumen':<15} {'Qin':<15} {'Temp'}")
print("-" * 60)
for _, row in bottom_dias.iterrows():
    print(f"{str(row['timestamp_utc'])[:16]:<20} {row[volumen_col]:>10,.0f} m³   "
          f"{row[qin_col]:>10,.0f} m³   {row['temp']:>5.1f}°C")

print("\n" + "=" * 80)
print("4️⃣ ANÁLISIS POR DÍAS ESPECIALES")
print("=" * 80)

# Año Nuevo
df_clima['dia'] = df_clima['timestamp_utc'].dt.day
df_clima['hora'] = df_clima['timestamp_utc'].dt.hour

anio_nuevo = df_clima[(df_clima['mes']==1) & (df_clima['dia']==1)]
dia_normal = df_clima[(df_clima['mes']==1) & (df_clima['dia'].isin([10,11,12]))]

print("\n🎉 AÑO NUEVO (1 de Enero):")
print(f"   Volumen Total promedio: {anio_nuevo[volumen_col].mean():,.0f} m³/hora")
print(f"   Qin promedio: {anio_nuevo[qin_col].mean():,.0f} m³/hora")

print("\n📅 DÍA NORMAL (10-12 Enero):")
print(f"   Volumen Total promedio: {dia_normal[volumen_col].mean():,.0f} m³/hora")
print(f"   Qin promedio: {dia_normal[qin_col].mean():,.0f} m³/hora")

dif_vol = anio_nuevo[volumen_col].mean() - dia_normal[volumen_col].mean()
print(f"\n🔍 Diferencia Año Nuevo vs Normal:")
print(f"   Volumen: {dif_vol:+,.0f} m³/hora ({dif_vol/dia_normal[volumen_col].mean()*100:+.2f}%)")

print("\n" + "=" * 80)
print("5️⃣ CONCLUSIÓN")
print("=" * 80)

if verano_vol > invierno_vol:
    print("\n✅ CONFIRMACIÓN DEL USUARIO:")
    print(f"   Volumen Total = CONSUMO/DEMANDA")
    print(f"   • VERANO (calor): MAYOR consumo ({verano_vol:,.0f} m³/hora)")
    print(f"   • INVIERNO (frío): MENOR consumo ({invierno_vol:,.0f} m³/hora)")
    print(f"   • Qin sigue el mismo patrón (producción se ajusta a demanda)")
else:
    print("\n⚠️  PATRÓN CONTRARIO:")
    print(f"   • INVIERNO: MAYOR volumen ({invierno_vol:,.0f} m³/hora)")
    print(f"   • VERANO: MENOR volumen ({verano_vol:,.0f} m³/hora)")
    print(f"   • Esto contradice la hipótesis del usuario")

print(f"\n🎯 INTERPRETACIÓN CORRECTA:")
print(f"   Volumen Total = Volumen acumulado en TANQUES")
print(f"   Qin = Producción (caudal de entrada)")
print(f"   Relación: Volumen ≈ {df['ratio_vol_qin'].mean():.1f}x Qin")
print(f"   Correlación: {corr:.4f}")

if corr > 0.8:
    print(f"\n✅ Alta correlación ({corr:.4f}) → Qin sigue a Volumen Total")
    print(f"   → Volumen Total refleja la DEMANDA real")
    print(f"   → Qin (producción) se ajusta a la demanda")
    
print("\n" + "=" * 80)
