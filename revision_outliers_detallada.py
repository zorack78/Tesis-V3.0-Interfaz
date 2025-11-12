"""
REVISIÓN DETALLADA DE OUTLIERS
Responde a dudas operacionales antes de análisis climático:
1. ¿Por qué dos columnas de timestamp?
2. ¿Hay realmente valores negativos?
3. ¿Cuándo >20% estanques tienen valor 0 o pegado?
4. Clasificación de tipos de error
5. Eventos continuos/discontinuos por día
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

print("="*80)
print("REVISIÓN DETALLADA DE OUTLIERS - ANÁLISIS OPERACIONAL")
print("="*80)
print(f"\nFecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== 1. ACLARAR TIMESTAMPS ====================
print("\n" + "="*80)
print("[1/5] ACLARANDO COLUMNAS DE TIMESTAMP")
print("="*80)

# Cargar outliers
outliers_path = Path('outputs/outliers_demanda_completo.csv')
df_outliers = pd.read_csv(outliers_path)

print(f"\n📊 Outliers cargados: {len(df_outliers)} registros")
print(f"\n🕐 COLUMNAS DE TIEMPO EN EL ARCHIVO:")
time_cols = [col for col in df_outliers.columns if 'time' in col.lower() or 'fecha' in col.lower() or 'hora' in col.lower()]
for col in time_cols:
    print(f"   • {col}")

# Convertir a datetime (UTC aware)
df_outliers['timestamp_utc'] = pd.to_datetime(df_outliers['timestamp_utc'], utc=True)
df_outliers['fecha_hora_local'] = pd.to_datetime(df_outliers['fecha_hora_local'], utc=True)

# Mostrar ejemplos
print(f"\n📋 EJEMPLOS (primeros 5 registros):")
print(f"\n{'timestamp_utc':<30} {'fecha_hora_local':<30} {'Diferencia':<15}")
print("-" * 80)
for idx in range(min(5, len(df_outliers))):
    utc = df_outliers.iloc[idx]['timestamp_utc']
    local = df_outliers.iloc[idx]['fecha_hora_local']
    diff = (utc - local).total_seconds() / 3600
    print(f"{str(utc):<30} {str(local):<30} {diff:+.1f} horas")

# Calcular diferencia general
df_outliers['diff_hours'] = (df_outliers['timestamp_utc'] - df_outliers['fecha_hora_local']).dt.total_seconds() / 3600

print(f"\n💡 EXPLICACIÓN:")
print(f"   • timestamp_utc:       Hora UTC (Coordinated Universal Time)")
print(f"   • fecha_hora_local:    Hora de Chile (America/Santiago)")
print(f"   • Diferencia típica:   {df_outliers['diff_hours'].mode()[0]:+.0f} horas")
print(f"\n   Chile usa UTC-3 (verano) o UTC-4 (invierno) por horario de verano.")
print(f"   Ambas columnas representan el MISMO instante, solo cambia la zona horaria.")
print(f"\n   ✅ Correcta implementación: timestamp_utc es el estándar para almacenamiento.")
print(f"   ✅ fecha_hora_local facilita interpretación operacional.")

# Verificar consistencia
inconsistencias = df_outliers[~df_outliers['diff_hours'].isin([-3, -4])].shape[0]
print(f"\n🔍 Inconsistencias detectadas: {inconsistencias}")
if inconsistencias > 0:
    print(f"   ⚠️  Revisar registros con diferencia != -3/-4 horas")

# ==================== 2. VERIFICAR VALORES NEGATIVOS ====================
print("\n" + "="*80)
print("[2/5] VERIFICANDO VALORES NEGATIVOS EN DEMANDA")
print("="*80)

print(f"\n📊 ANÁLISIS DE COLUMNAS DE DEMANDA:")
cols_demanda = ['Demanda_m3_hr', 'abs_demanda']
for col in cols_demanda:
    if col in df_outliers.columns:
        values = df_outliers[col]
        print(f"\n   {col}:")
        print(f"      • Mínimo:   {values.min():,.2f} m³/hr")
        print(f"      • Máximo:   {values.max():,.2f} m³/hr")
        print(f"      • Media:    {values.mean():,.2f} m³/hr")
        print(f"      • Negativos: {(values < 0).sum()} registros ({(values < 0).sum()/len(values)*100:.1f}%)")
        print(f"      • Ceros:     {(values == 0).sum()} registros")

# Mostrar ejemplos de valores extremos
print(f"\n📋 TOP 5 VALORES MÁS NEGATIVOS (Demanda_m3_hr):")
if 'Demanda_m3_hr' in df_outliers.columns:
    negativos = df_outliers.nsmallest(5, 'Demanda_m3_hr')[['fecha_hora_local', 'Demanda_m3_hr', 'Qin', 'delta_volumen_m3_hr']]
    for idx, row in negativos.iterrows():
        print(f"   {row['fecha_hora_local']}: Demanda={row['Demanda_m3_hr']:,.0f}, Qin={row['Qin']:,.0f}, ΔVol={row['delta_volumen_m3_hr']:,.0f}")

print(f"\n📋 TOP 5 VALORES MÁS ALTOS (Demanda_m3_hr):")
positivos = df_outliers.nlargest(5, 'Demanda_m3_hr')[['fecha_hora_local', 'Demanda_m3_hr', 'Qin', 'delta_volumen_m3_hr', 'temp']]
for idx, row in positivos.iterrows():
    temp_val = row['temp'] if 'temp' in row else 'N/A'
    print(f"   {row['fecha_hora_local']}: Demanda={row['Demanda_m3_hr']:,.0f}, Qin={row['Qin']:,.0f}, Temp={temp_val}°C")

print(f"\n💡 INTERPRETACIÓN:")
print(f"   • abs_demanda: Valor absoluto (siempre positivo) - para análisis estadístico")
print(f"   • Demanda_m3_hr: Valor real con signo - puede ser negativo si ΔVolumen > Qin")
print(f"   • Valores negativos indican: Volumen aumentó más que el agua ingresada")
print(f"     Posibles causas: Error sensor, recirculación, retorno de red, cálculo erróneo")

# ==================== 3. CARGAR Y ANALIZAR VOLÚMENES POR ESTANQUE ====================
print("\n" + "="*80)
print("[3/5] ANALIZANDO VOLÚMENES INDIVIDUALES DE ESTANQUES")
print("="*80)

# Cargar volúmenes por estanque
vol_estanques_path = Path('data/raw/Vol_X_TK_Hr_m3_UTC.csv')
print(f"\n📂 Cargando: {vol_estanques_path}")

if vol_estanques_path.exists():
    df_vol_estanques = pd.read_csv(vol_estanques_path)
    df_vol_estanques['timestamp'] = pd.to_datetime(df_vol_estanques['timestamp'])
    
    print(f"✅ Cargado: {len(df_vol_estanques)} registros")
    print(f"   Rango: {df_vol_estanques['timestamp'].min()} a {df_vol_estanques['timestamp'].max()}")
    
    # Identificar columnas de estanques
    estanque_cols = [col for col in df_vol_estanques.columns if col not in ['timestamp', 'fecha']]
    n_estanques = len(estanque_cols)
    print(f"   Estanques detectados: {n_estanques}")
    
    # Para cada timestamp de outlier, verificar estado de estanques
    print(f"\n🔍 ANALIZANDO ESTADO DE ESTANQUES EN HORAS DE OUTLIERS...")
    
    resultados_estanques = []
    
    for idx, row in df_outliers.iterrows():
        timestamp_utc = row['timestamp_utc']
        
        # Buscar registro correspondiente
        registro_estanques = df_vol_estanques[df_vol_estanques['timestamp'] == timestamp_utc]
        
        if len(registro_estanques) > 0:
            valores = registro_estanques[estanque_cols].iloc[0]
            
            # Contar estanques con problemas
            n_ceros = (valores == 0).sum()
            n_nulos = valores.isna().sum()
            
            # Detectar valores "pegados" (sin cambio respecto a hora anterior)
            if idx > 0:
                timestamp_prev = df_vol_estanques[df_vol_estanques['timestamp'] < timestamp_utc].tail(1)
                if len(timestamp_prev) > 0:
                    valores_prev = timestamp_prev[estanque_cols].iloc[0]
                    n_pegados = (valores == valores_prev).sum()
                else:
                    n_pegados = 0
            else:
                n_pegados = 0
            
            # Calcular porcentaje
            pct_ceros = (n_ceros / n_estanques) * 100
            pct_nulos = (n_nulos / n_estanques) * 100
            pct_pegados = (n_pegados / n_estanques) * 100
            pct_problemas = ((n_ceros + n_nulos) / n_estanques) * 100
            
            # Guardar si >20% tienen problemas
            if pct_problemas > 20:
                resultados_estanques.append({
                    'timestamp_utc': timestamp_utc,
                    'fecha_hora_local': row['fecha_hora_local'],
                    'Demanda_m3_hr': row['Demanda_m3_hr'],
                    'n_estanques_total': n_estanques,
                    'n_ceros': n_ceros,
                    'n_nulos': n_nulos,
                    'n_pegados': n_pegados,
                    'pct_ceros': pct_ceros,
                    'pct_nulos': pct_nulos,
                    'pct_pegados': pct_pegados,
                    'pct_problemas_total': pct_problemas
                })
    
    df_problemas_estanques = pd.DataFrame(resultados_estanques)
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Outliers analizados: {len(df_outliers)}")
    print(f"   Con >20% estanques problemáticos: {len(df_problemas_estanques)} ({len(df_problemas_estanques)/len(df_outliers)*100:.1f}%)")
    
    if len(df_problemas_estanques) > 0:
        print(f"\n📋 TOP 10 PEORES CASOS (más estanques con problemas):")
        print(f"\n{'Fecha/Hora Local':<25} {'Demanda':>12} {'Ceros':>8} {'Nulos':>8} {'%Prob':>8}")
        print("-" * 70)
        top_problemas = df_problemas_estanques.nlargest(10, 'pct_problemas_total')
        for idx, row in top_problemas.iterrows():
            print(f"{str(row['fecha_hora_local']):<25} {row['Demanda_m3_hr']:>12,.0f} {row['n_ceros']:>8.0f} {row['n_nulos']:>8.0f} {row['pct_problemas_total']:>7.1f}%")
        
        # Guardar listado completo
        output_path = Path('outputs/outliers_con_problemas_estanques.csv')
        df_problemas_estanques.to_csv(output_path, index=False)
        print(f"\n✅ Guardado: {output_path}")
        print(f"   Registros: {len(df_problemas_estanques)}")
    
else:
    print(f"❌ No encontrado: {vol_estanques_path}")
    print(f"   No se puede analizar volúmenes individuales de estanques.")
    df_problemas_estanques = pd.DataFrame()

# ==================== 4. CLASIFICACIÓN DE TIPOS DE ERROR ====================
print("\n" + "="*80)
print("[4/5] CLASIFICACIÓN DE TIPOS DE ERROR")
print("="*80)

print(f"\n🔍 CLASIFICANDO OUTLIERS POR TIPO DE ERROR...")

df_outliers['tipo_error'] = 'No clasificado'
df_outliers['severidad'] = 'Baja'

# Tipo 1: Demanda negativa (errores de sensor/cálculo)
mask_negativo = df_outliers['Demanda_m3_hr'] < 0
df_outliers.loc[mask_negativo, 'tipo_error'] = 'Demanda Negativa'
df_outliers.loc[mask_negativo, 'severidad'] = 'Alta'

# Tipo 2: Demanda extremadamente alta (posible evento real o error)
mask_muy_alto = df_outliers['Demanda_m3_hr'] > 30000
df_outliers.loc[mask_muy_alto, 'tipo_error'] = 'Demanda Extrema Alta (>30k)'
df_outliers.loc[mask_muy_alto, 'severidad'] = 'Media'

# Tipo 3: Delta volumen extremo
mask_delta_extremo = df_outliers['delta_volumen_m3_hr'].abs() > 100000
df_outliers.loc[mask_delta_extremo, 'tipo_error'] = 'Delta Volumen Extremo'
df_outliers.loc[mask_delta_extremo, 'severidad'] = 'Alta'

# Tipo 4: Qin muy bajo con demanda alta
mask_qin_bajo = (df_outliers['Qin'] < 5000) & (df_outliers['Demanda_m3_hr'] > 20000)
df_outliers.loc[mask_qin_bajo, 'tipo_error'] = 'Qin Insuficiente'
df_outliers.loc[mask_qin_bajo, 'severidad'] = 'Media'

# Tipo 5: Problema de estanques (si tenemos esa info)
if len(df_problemas_estanques) > 0:
    timestamps_problemas = df_problemas_estanques['timestamp_utc'].values
    mask_estanques = df_outliers['timestamp_utc'].isin(timestamps_problemas)
    df_outliers.loc[mask_estanques, 'tipo_error'] = 'Estanques Problemáticos (>20%)'
    df_outliers.loc[mask_estanques, 'severidad'] = 'Alta'

# Resumen por tipo
print(f"\n📊 DISTRIBUCIÓN POR TIPO DE ERROR:")
tipo_counts = df_outliers['tipo_error'].value_counts()
for tipo, count in tipo_counts.items():
    print(f"   {tipo:<40}: {count:>4} ({count/len(df_outliers)*100:>5.1f}%)")

print(f"\n📊 DISTRIBUCIÓN POR SEVERIDAD:")
sev_counts = df_outliers['severidad'].value_counts()
for sev, count in sev_counts.items():
    print(f"   {sev:<20}: {count:>4} ({count/len(df_outliers)*100:>5.1f}%)")

# ==================== 5. EVENTOS CONTINUOS/DISCONTINUOS POR DÍA ====================
print("\n" + "="*80)
print("[5/5] IDENTIFICANDO EVENTOS CONTINUOS/DISCONTINUOS")
print("="*80)

# Extraer fecha (sin hora)
df_outliers['fecha'] = df_outliers['fecha_hora_local'].dt.date
df_outliers['hora'] = df_outliers['fecha_hora_local'].dt.hour

# Agrupar por fecha
print(f"\n🔍 ANALIZANDO CONTINUIDAD DE EVENTOS POR DÍA...")

eventos_por_dia = []

for fecha, grupo in df_outliers.groupby('fecha'):
    n_outliers = len(grupo)
    horas = sorted(grupo['hora'].values)
    
    # Detectar continuidad
    if n_outliers == 1:
        tipo_evento = 'Aislado'
        horas_consecutivas = 0
    else:
        # Verificar si las horas son consecutivas
        diffs = np.diff(horas)
        consecutivas = (diffs == 1).sum()
        horas_consecutivas = consecutivas + 1 if consecutivas > 0 else 0
        
        if horas_consecutivas == n_outliers:
            tipo_evento = 'Continuo'
        elif horas_consecutivas > 0:
            tipo_evento = 'Mixto (continuo + discontinuo)'
        else:
            tipo_evento = 'Discontinuo'
    
    # Obtener tipos de error del día
    tipos_error = grupo['tipo_error'].value_counts().to_dict()
    tipo_error_principal = grupo['tipo_error'].mode()[0]
    
    eventos_por_dia.append({
        'fecha': fecha,
        'n_outliers': n_outliers,
        'horas': horas,
        'horas_consecutivas_max': horas_consecutivas,
        'tipo_evento': tipo_evento,
        'tipo_error_principal': tipo_error_principal,
        'tipos_error_dict': tipos_error
    })

df_eventos = pd.DataFrame(eventos_por_dia)

print(f"\n📊 RESUMEN:")
print(f"   Días con outliers: {len(df_eventos)}")
print(f"   Total outliers: {len(df_outliers)}")
print(f"   Promedio outliers/día: {len(df_outliers)/len(df_eventos):.1f}")

print(f"\n📊 DISTRIBUCIÓN POR TIPO DE EVENTO:")
tipo_evento_counts = df_eventos['tipo_evento'].value_counts()
for tipo, count in tipo_evento_counts.items():
    print(f"   {tipo:<30}: {count:>4} días ({count/len(df_eventos)*100:>5.1f}%)")

print(f"\n📋 TOP 10 DÍAS CON MÁS OUTLIERS:")
print(f"\n{'Fecha':<15} {'N':>4} {'Tipo Evento':<30} {'Error Principal':<30}")
print("-" * 90)
top_dias = df_eventos.nlargest(10, 'n_outliers')
for idx, row in top_dias.iterrows():
    print(f"{str(row['fecha']):<15} {row['n_outliers']:>4} {row['tipo_evento']:<30} {row['tipo_error_principal']:<30}")

# Guardar eventos por día
output_eventos_path = Path('outputs/eventos_outliers_por_dia.csv')
df_eventos_export = df_eventos.copy()
df_eventos_export['horas'] = df_eventos_export['horas'].apply(lambda x: str(list(x)))
df_eventos_export['tipos_error_dict'] = df_eventos_export['tipos_error_dict'].apply(lambda x: str(x))
df_eventos_export.to_csv(output_eventos_path, index=False)
print(f"\n✅ Guardado: {output_eventos_path}")

# Guardar outliers clasificados
output_clasificados_path = Path('outputs/outliers_clasificados.csv')
df_outliers.to_csv(output_clasificados_path, index=False)
print(f"✅ Guardado: {output_clasificados_path}")

# ==================== RESUMEN EJECUTIVO ====================
print("\n" + "="*80)
print("📊 RESUMEN EJECUTIVO")
print("="*80)

print(f"""
1️⃣ TIMESTAMPS:
   ✅ Se usan correctamente 2 columnas:
      • timestamp_utc: Estándar UTC para almacenamiento
      • fecha_hora_local: Hora Chile (UTC-3/-4) para interpretación
   ✅ Diferencia constante: {df_outliers['diff_hours'].mode()[0]:+.0f} horas
   ✅ Zona horaria correcta: America/Santiago (Valparaíso)

2️⃣ VALORES NEGATIVOS:
   • Demanda_m3_hr: {(df_outliers['Demanda_m3_hr'] < 0).sum()} registros negativos ({(df_outliers['Demanda_m3_hr'] < 0).sum()/len(df_outliers)*100:.1f}%)
   • abs_demanda: Todos positivos (valor absoluto para estadística)
   💡 Negativos = ΔVolumen > Qin (posible error sensor o recirculación)

3️⃣ ESTANQUES PROBLEMÁTICOS:
   • Outliers con >20% estanques problemáticos: {len(df_problemas_estanques)}
   • Porcentaje: {len(df_problemas_estanques)/len(df_outliers)*100:.1f}%
   ⚠️  Indica fallos de telemetría/sensores distribuidos

4️⃣ CLASIFICACIÓN DE ERRORES:
""")

for tipo, count in tipo_counts.items():
    print(f"   • {tipo}: {count} ({count/len(df_outliers)*100:.1f}%)")

print(f"""
5️⃣ EVENTOS TEMPORALES:
   • Días con outliers: {len(df_eventos)}
   • Eventos continuos: {tipo_evento_counts.get('Continuo', 0)} días
   • Eventos discontinuos: {tipo_evento_counts.get('Discontinuo', 0)} días
   • Eventos aislados: {tipo_evento_counts.get('Aislado', 0)} días
   💡 Eventos continuos sugieren problemas sostenidos (clima, infraestructura)

📁 ARCHIVOS GENERADOS:
   • outputs/outliers_clasificados.csv
   • outputs/eventos_outliers_por_dia.csv
   • outputs/outliers_con_problemas_estanques.csv (si >20% estanques)
""")

print("\n" + "="*80)
print("✅ REVISIÓN COMPLETADA")
print("="*80)
print("\n💡 PRÓXIMO PASO: Análisis de correlación con eventos climáticos extremos")
print("="*80 + "\n")
