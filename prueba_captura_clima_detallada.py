"""
PRUEBA DETALLADA: CAPTURA CLIMA ACTUAL + PRONÓSTICO 3 DÍAS
===========================================================

Estación: Rodelillo, Ad.
Coordenadas: -33.06528°, -71.55639°
Altura: 335.0 msnm
Link: https://climatologia.meteochile.gob.cl/application/diariob/visorDeDatosEma/330007

Objetivo:
Capturar datos actuales y pronóstico 3 días para validar valores.
"""

import requests
import pandas as pd
from datetime import datetime
import json

print("=" * 80)
print("PRUEBA: CAPTURA CLIMA ACTUAL + PRONÓSTICO 3 DÍAS")
print("=" * 80)

# Configuración Estación Rodelillo
LAT = -33.06528
LON = -71.55639
ESTACION = "Rodelillo, Ad."
ALTURA = 335.0
LINK_DMC = "https://climatologia.meteochile.gob.cl/application/diariob/visorDeDatosEma/330007"

print(f"\n📍 ESTACIÓN METEOROLÓGICA:")
print(f"   Nombre: {ESTACION}")
print(f"   Coordenadas: {LAT}°, {LON}°")
print(f"   Altura: {ALTURA} msnm")
print(f"   Link DMC: {LINK_DMC}")

fecha_consulta = datetime.now()
print(f"\n📅 FECHA/HORA CONSULTA: {fecha_consulta.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONSULTA A OPEN-METEO
# ============================================================================

print("\n" + "=" * 80)
print("CONSULTANDO OPEN-METEO API")
print("=" * 80)

url = "https://api.open-meteo.com/v1/forecast"

params = {
    'latitude': LAT,
    'longitude': LON,
    'current': [
        'temperature_2m',
        'relative_humidity_2m',
        'precipitation',
        'rain',
        'weather_code',
        'pressure_msl',
        'wind_speed_10m'
    ],
    'hourly': [
        'temperature_2m',
        'relative_humidity_2m',
        'precipitation',
        'rain',
        'weather_code'
    ],
    'daily': [
        'temperature_2m_max',
        'temperature_2m_min',
        'temperature_2m_mean',
        'precipitation_sum',
        'rain_sum',
        'precipitation_hours',
        'precipitation_probability_max',
        'wind_speed_10m_max'
    ],
    'timezone': 'America/Santiago',
    'forecast_days': 3
}

try:
    print(f"\n🔄 Consultando API...")
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Error: Status {response.status_code}")
        print(f"Respuesta: {response.text}")
        exit()
    
    data = response.json()
    print(f"✅ Respuesta exitosa")
    
    # ========================================================================
    # DATOS ACTUALES (MOMENTO DE LA CONSULTA)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("🌡️  DATOS CLIMÁTICOS ACTUALES (AHORA)")
    print("=" * 80)
    
    current = data['current']
    
    print(f"\n📊 TIMESTAMP: {current['time']}")
    print(f"\n   Temperatura:        {current['temperature_2m']:>6.1f} °C")
    print(f"   Humedad Relativa:   {current['relative_humidity_2m']:>6.1f} %")
    print(f"   Precipitación:      {current['precipitation']:>6.2f} mm")
    print(f"   Lluvia:             {current['rain']:>6.2f} mm")
    print(f"   Presión:            {current['pressure_msl']:>6.1f} hPa")
    print(f"   Viento:             {current['wind_speed_10m']:>6.1f} km/h")
    print(f"   Código clima:       {current['weather_code']}")
    
    # Interpretación código WMO
    weather_codes = {
        0: "Despejado",
        1: "Principalmente despejado",
        2: "Parcialmente nublado",
        3: "Nublado",
        45: "Niebla",
        48: "Niebla depositando escarcha",
        51: "Llovizna: Ligera",
        53: "Llovizna: Moderada",
        55: "Llovizna: Densa",
        61: "Lluvia: Ligera",
        63: "Lluvia: Moderada",
        65: "Lluvia: Intensa",
        71: "Nieve: Ligera",
        73: "Nieve: Moderada",
        75: "Nieve: Intensa",
        80: "Chubascos: Ligeros",
        81: "Chubascos: Moderados",
        82: "Chubascos: Violentos",
        95: "Tormenta",
        96: "Tormenta con granizo ligero",
        99: "Tormenta con granizo intenso"
    }
    
    descripcion = weather_codes.get(current['weather_code'], "Desconocido")
    print(f"   Descripción:        {descripcion}")
    
    # ========================================================================
    # PRONÓSTICO HORARIO (PRÓXIMAS 72 HORAS)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📅 PRONÓSTICO HORARIO (PRÓXIMAS 72 HORAS)")
    print("=" * 80)
    
    hourly = data['hourly']
    df_hourly = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        'temperatura': hourly['temperature_2m'],
        'humedad_relativa': hourly['relative_humidity_2m'],
        'precipitacion': hourly['precipitation'],
        'lluvia': hourly['rain'],
        'codigo_clima': hourly['weather_code']
    })
    
    # Agregar descripción del clima
    df_hourly['descripcion'] = df_hourly['codigo_clima'].map(
        lambda x: weather_codes.get(x, "Desconocido")
    )
    
    print(f"\n📊 Total registros horarios: {len(df_hourly)}")
    print(f"   Desde: {df_hourly['timestamp'].min()}")
    print(f"   Hasta: {df_hourly['timestamp'].max()}")
    
    # Mostrar datos cada 3 horas
    print(f"\n{'Fecha/Hora':<20} {'Temp':<8} {'HR':<6} {'Precip':<8} {'Lluvia':<8} {'Clima':<30}")
    print("-" * 90)
    
    for i in range(0, len(df_hourly), 3):
        row = df_hourly.iloc[i]
        print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{row['temperatura']:>6.1f}°C "
              f"{row['humedad_relativa']:>5.0f}% "
              f"{row['precipitacion']:>6.2f}mm "
              f"{row['lluvia']:>6.2f}mm "
              f"{row['descripcion']:<30}")
    
    # ========================================================================
    # PRONÓSTICO DIARIO (PRÓXIMOS 3 DÍAS)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📅 PRONÓSTICO DIARIO (PRÓXIMOS 3 DÍAS)")
    print("=" * 80)
    
    daily = data['daily']
    df_daily = pd.DataFrame({
        'fecha': pd.to_datetime(daily['time']),
        'temp_max': daily['temperature_2m_max'],
        'temp_min': daily['temperature_2m_min'],
        'temp_media': daily['temperature_2m_mean'],
        'precip_total': daily['precipitation_sum'],
        'lluvia_total': daily['rain_sum'],
        'horas_precip': daily['precipitation_hours'],
        'prob_precip': daily['precipitation_probability_max'],
        'viento_max': daily['wind_speed_10m_max']
    })
    
    print(f"\n📊 Total días: {len(df_daily)}")
    
    for idx, row in df_daily.iterrows():
        print(f"\n{'=' * 80}")
        print(f"DÍA {idx + 1}: {row['fecha'].strftime('%A, %d de %B de %Y')}")
        print(f"{'=' * 80}")
        
        print(f"\n   🌡️  TEMPERATURA:")
        print(f"      Mínima:        {row['temp_min']:>6.1f} °C")
        print(f"      Máxima:        {row['temp_max']:>6.1f} °C")
        print(f"      Media:         {row['temp_media']:>6.1f} °C")
        print(f"      Amplitud:      {row['temp_max'] - row['temp_min']:>6.1f} °C")
        
        print(f"\n   💧 PRECIPITACIÓN:")
        print(f"      Total:         {row['precip_total']:>6.1f} mm")
        print(f"      Lluvia:        {row['lluvia_total']:>6.1f} mm")
        print(f"      Horas:         {row['horas_precip']:>6.1f} hrs")
        print(f"      Probabilidad:  {row['prob_precip']:>6.0f} %")
        
        print(f"\n   🌪️  VIENTO:")
        print(f"      Máximo:        {row['viento_max']:>6.1f} km/h")
        
        # Alertas para este día
        alertas_dia = []
        
        if row['temp_min'] < 12.0:
            alertas_dia.append(f"🔵 Temperatura muy baja ({row['temp_min']:.1f}°C)")
        
        if row['temp_max'] > 25.0:
            alertas_dia.append(f"🔴 Temperatura alta ({row['temp_max']:.1f}°C)")
        
        if (row['temp_max'] - row['temp_min']) > 12.0:
            alertas_dia.append(f"🟠 Alta variabilidad térmica ({row['temp_max'] - row['temp_min']:.1f}°C)")
        
        if row['precip_total'] > 10.0:
            alertas_dia.append(f"🔴 Precipitación alta ({row['precip_total']:.1f} mm)")
        
        if row['prob_precip'] > 70:
            alertas_dia.append(f"🟡 Alta probabilidad de lluvia ({row['prob_precip']:.0f}%)")
        
        if row['viento_max'] > 40.0:
            alertas_dia.append(f"🟠 Viento fuerte ({row['viento_max']:.1f} km/h)")
        
        if alertas_dia:
            print(f"\n   ⚠️  ALERTAS ({len(alertas_dia)}):")
            for alerta in alertas_dia:
                print(f"      • {alerta}")
        else:
            print(f"\n   ✅ Sin alertas (condiciones normales)")
    
    # ========================================================================
    # ANÁLISIS ESTADÍSTICO 72 HORAS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS ESTADÍSTICO (72 HORAS)")
    print("=" * 80)
    
    print(f"\n🌡️  TEMPERATURA:")
    print(f"   Mínima absoluta:    {df_hourly['temperatura'].min():>6.1f} °C")
    print(f"   Máxima absoluta:    {df_hourly['temperatura'].max():>6.1f} °C")
    print(f"   Media:              {df_hourly['temperatura'].mean():>6.1f} °C")
    print(f"   Desviación estándar:{df_hourly['temperatura'].std():>6.1f} °C")
    
    print(f"\n💧 HUMEDAD RELATIVA:")
    print(f"   Mínima:             {df_hourly['humedad_relativa'].min():>6.0f} %")
    print(f"   Máxima:             {df_hourly['humedad_relativa'].max():>6.0f} %")
    print(f"   Media:              {df_hourly['humedad_relativa'].mean():>6.1f} %")
    
    print(f"\n💧 PRECIPITACIÓN:")
    print(f"   Total acumulada:    {df_hourly['precipitacion'].sum():>6.1f} mm")
    print(f"   Máxima horaria:     {df_hourly['precipitacion'].max():>6.2f} mm/hr")
    print(f"   Horas con precip:   {(df_hourly['precipitacion'] > 0).sum()} hrs")
    
    # ========================================================================
    # ALERTAS CLIMÁTICAS GENERALES
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("⚠️  SISTEMA DE ALERTAS CLIMÁTICAS")
    print("=" * 80)
    
    alertas_generales = []
    
    # Basado en importancia de features del análisis previo
    
    # CRÍTICAS (Importancia >5%)
    if df_hourly['precipitacion'].max() > 2.0:
        alertas_generales.append({
            'nivel': '🔴 CRÍTICO',
            'tipo': 'LLUVIA INTENSA',
            'valor': f"{df_hourly['precipitacion'].max():.2f} mm/hr",
            'umbral': '>2 mm/hr',
            'importancia': '36.36%',
            'impacto': 'Reducción drástica de demanda, mayor recuperación en tanques'
        })
    
    if df_hourly['precipitacion'].sum() > 20.0:
        alertas_generales.append({
            'nivel': '🔴 CRÍTICO',
            'tipo': 'PRECIPITACIÓN ACUMULADA ALTA',
            'valor': f"{df_hourly['precipitacion'].sum():.1f} mm",
            'umbral': '>20 mm en 72h',
            'importancia': '9.27%',
            'impacto': 'Efecto sostenido en reducción de consumo'
        })
    
    # IMPORTANTES (Importancia 1-5%)
    if df_hourly['temperatura'].min() < 12.0:
        alertas_generales.append({
            'nivel': '🟡 IMPORTANTE',
            'tipo': 'TEMPERATURA MUY BAJA',
            'valor': f"{df_hourly['temperatura'].min():.1f} °C",
            'umbral': '<12°C',
            'importancia': '1.03%',
            'impacto': 'Reduce consumo nocturno, aumenta recuperación en madrugada'
        })
    
    if df_hourly['temperatura'].std() > 5.0:
        alertas_generales.append({
            'nivel': '🟡 IMPORTANTE',
            'tipo': 'ALTA VARIABILIDAD TÉRMICA',
            'valor': f"±{df_hourly['temperatura'].std():.1f} °C",
            'umbral': 'std >5°C',
            'importancia': '6.95%',
            'impacto': 'Cambios bruscos alteran patrones de consumo'
        })
    
    # MODERADAS
    if df_hourly['humedad_relativa'].max() > 90:
        alertas_generales.append({
            'nivel': '🟢 MODERADO',
            'tipo': 'HUMEDAD MUY ALTA',
            'valor': f"{df_hourly['humedad_relativa'].max():.0f} %",
            'umbral': '>90%',
            'importancia': '2.17%',
            'impacto': 'Aumenta sensación térmica, afecta confort'
        })
    
    if alertas_generales:
        print(f"\n⚠️  TOTAL ALERTAS DETECTADAS: {len(alertas_generales)}")
        print(f"\n")
        
        for i, alerta in enumerate(alertas_generales, 1):
            print(f"{alerta['nivel']} ALERTA #{i}: {alerta['tipo']}")
            print(f"   Valor actual:   {alerta['valor']}")
            print(f"   Umbral:         {alerta['umbral']}")
            print(f"   Importancia:    {alerta['importancia']} (modelo ML)")
            print(f"   Impacto:        {alerta['impacto']}")
            print()
    else:
        print(f"\n✅ SIN ALERTAS: Condiciones climáticas normales")
    
    # ========================================================================
    # GUARDAR DATOS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("💾 GUARDANDO DATOS")
    print("=" * 80)
    
    # Datos actuales
    df_current = pd.DataFrame([{
        'timestamp': current['time'],
        'temperatura': current['temperature_2m'],
        'humedad_relativa': current['relative_humidity_2m'],
        'precipitacion': current['precipitation'],
        'lluvia': current['rain'],
        'presion': current['pressure_msl'],
        'viento': current['wind_speed_10m'],
        'codigo_clima': current['weather_code'],
        'descripcion': descripcion
    }])
    
    df_current.to_csv('outputs/clima_actual_rodelillo.csv', index=False)
    print(f"✅ Clima actual guardado: outputs/clima_actual_rodelillo.csv")
    
    # Pronóstico horario
    df_hourly.to_csv('outputs/pronostico_horario_72h_rodelillo.csv', index=False)
    print(f"✅ Pronóstico horario guardado: outputs/pronostico_horario_72h_rodelillo.csv")
    
    # Pronóstico diario
    df_daily.to_csv('outputs/pronostico_diario_3dias_rodelillo.csv', index=False)
    print(f"✅ Pronóstico diario guardado: outputs/pronostico_diario_3dias_rodelillo.csv")
    
    # Alertas
    if alertas_generales:
        df_alertas = pd.DataFrame(alertas_generales)
        df_alertas.to_csv('outputs/alertas_climaticas_rodelillo.csv', index=False)
        print(f"✅ Alertas guardadas: outputs/alertas_climaticas_rodelillo.csv")
    
    # JSON completo para respaldo
    with open('outputs/respuesta_completa_openmeteo.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Respuesta JSON completa: outputs/respuesta_completa_openmeteo.json")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    
    print(f"\n📋 RESUMEN:")
    print(f"   Estación:           {ESTACION}")
    print(f"   Fecha consulta:     {fecha_consulta.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Datos actuales:     ✅ Capturados")
    print(f"   Pronóstico horario: ✅ {len(df_hourly)} horas")
    print(f"   Pronóstico diario:  ✅ {len(df_daily)} días")
    print(f"   Alertas:            ✅ {len(alertas_generales)} detectadas")
    print(f"   Archivos generados: ✅ 5 archivos CSV/JSON")
    
    print(f"\n💡 PRÓXIMO PASO:")
    print(f"   Los datos capturados pueden integrarse directamente al modelo")
    print(f"   para generar predicciones de demanda con clima futuro.")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
