"""
TEST: CAPTURA DE PRONÓSTICO CLIMÁTICO
======================================

Objetivo:
Probar diferentes APIs meteorológicas para capturar pronóstico del tiempo
en Valparaíso, Chile.

Datos requeridos:
- Temperatura máxima/mínima (°C)
- Humedad relativa (%)
- Precipitación pronosticada (mm)
- Pronóstico 24-48h adelante

APIs a probar:
1. OpenWeatherMap (GRATUITA, 1000 llamadas/día)
2. WeatherAPI (GRATUITA, 1M llamadas/mes)
3. Open-Meteo (GRATUITA, sin límite, sin API key)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json

print("=" * 80)
print("TEST: CAPTURA DE PRONÓSTICO CLIMÁTICO PARA VALPARAÍSO")
print("=" * 80)
print(f"\nFecha consulta: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Coordenadas de Estación Rodelillo (misma fuente de datos históricos)
LAT = -33.06528
LON = -71.55639
ESTACION = "Rodelillo, Ad."
ALTURA = 335.0  # metros sobre nivel del mar
CIUDAD = "Valparaíso"

print(f"\n📍 ESTACIÓN METEOROLÓGICA:")
print(f"   Nombre: {ESTACION}")
print(f"   Coordenadas: {LAT}°, {LON}°")
print(f"   Altura: {ALTURA} msnm")
print(f"   ⚠️  IMPORTANTE: Misma estación de datos históricos en BD")

# ============================================================================
# OPCIÓN 1: OPEN-METEO (GRATUITA, SIN API KEY, RECOMENDADA)
# ============================================================================

print("\n" + "=" * 80)
print("[OPCIÓN 1] OPEN-METEO (https://open-meteo.com)")
print("=" * 80)
print("✅ Ventajas: GRATUITA, sin API key, sin límite, datos cada hora")
print("✅ Datos: Temp, HR, Precipitación, Viento, Presión")
print("✅ Pronóstico: Hasta 16 días adelante")

try:
    url_meteo = "https://api.open-meteo.com/v1/forecast"
    
    params_meteo = {
        'latitude': LAT,
        'longitude': LON,
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
            'precipitation_sum',
            'precipitation_probability_max'
        ],
        'timezone': 'America/Santiago',
        'forecast_days': 3  # 3 días de pronóstico
    }
    
    print(f"\n🔄 Consultando Open-Meteo para {ESTACION} ({CIUDAD})...")
    response_meteo = requests.get(url_meteo, params=params_meteo, timeout=10)
    
    if response_meteo.status_code == 200:
        data_meteo = response_meteo.json()
        
        print(f"✅ Respuesta exitosa de Open-Meteo")
        print(f"\n📊 PRONÓSTICO HORARIO (próximas 48 horas):")
        print("-" * 80)
        
        # Procesar datos horarios
        hourly = data_meteo['hourly']
        df_hourly = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'temperatura': hourly['temperature_2m'],
            'humedad_relativa': hourly['relative_humidity_2m'],
            'precipitacion': hourly['precipitation']
        })
        
        # Primeras 48 horas
        df_48h = df_hourly.head(48)
        
        # Mostrar cada 6 horas para no saturar
        for i in range(0, min(48, len(df_48h)), 6):
            row = df_48h.iloc[i]
            print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                  f"Temp: {row['temperatura']:>5.1f}°C | "
                  f"HR: {row['humedad_relativa']:>5.1f}% | "
                  f"Precip: {row['precipitacion']:>5.1f} mm")
        
        print(f"\n📊 PRONÓSTICO DIARIO (próximos 3 días):")
        print("-" * 80)
        
        # Procesar datos diarios
        daily = data_meteo['daily']
        df_daily = pd.DataFrame({
            'fecha': pd.to_datetime(daily['time']),
            'temp_max': daily['temperature_2m_max'],
            'temp_min': daily['temperature_2m_min'],
            'precip_total': daily['precipitation_sum'],
            'prob_precip': daily['precipitation_probability_max']
        })
        
        for _, row in df_daily.iterrows():
            print(f"{row['fecha'].strftime('%Y-%m-%d')} | "
                  f"Temp: {row['temp_min']:>5.1f}°C - {row['temp_max']:>5.1f}°C | "
                  f"Precip: {row['precip_total']:>5.1f} mm ({row['prob_precip']:>3.0f}% prob)")
        
        # ANÁLISIS DE CONDICIONES EXTREMAS
        print(f"\n🎯 ANÁLISIS: CONDICIONES EXTREMAS DETECTADAS")
        print("-" * 80)
        
        # Basado en análisis previo: temp_muy_baja <12°C, lluvia_intensa >2mm/hr
        temp_min_48h = df_48h['temperatura'].min()
        temp_max_48h = df_48h['temperatura'].max()
        precip_max_48h = df_48h['precipitacion'].max()
        precip_total_48h = df_48h['precipitacion'].sum()
        hr_min_48h = df_48h['humedad_relativa'].min()
        hr_max_48h = df_48h['humedad_relativa'].max()
        
        print(f"\nRango temperatura 48h: {temp_min_48h:.1f}°C - {temp_max_48h:.1f}°C")
        print(f"Precipitación total 48h: {precip_total_48h:.1f} mm")
        print(f"Precipitación máxima horaria: {precip_max_48h:.1f} mm/hr")
        print(f"Rango humedad relativa: {hr_min_48h:.0f}% - {hr_max_48h:.0f}%")
        
        # Alertas según importancia de features
        alertas = []
        
        if precip_max_48h > 2.0:
            alertas.append(f"🔴 LLUVIA INTENSA: {precip_max_48h:.1f} mm/hr "
                          f"(Feature #1: 36% importancia)")
        
        if precip_total_48h > 10.0:
            alertas.append(f"🟡 PRECIPITACIÓN ACUMULADA ALTA: {precip_total_48h:.1f} mm "
                          f"(Features #2,#5,#7: 16% importancia)")
        
        if temp_min_48h < 12.0:
            alertas.append(f"🔵 TEMPERATURA MUY BAJA: {temp_min_48h:.1f}°C "
                          f"(Feature #16: 1% importancia + efecto recuperación)")
        
        temp_std_48h = df_48h['temperatura'].std()
        if temp_std_48h > 5.0:
            alertas.append(f"🟠 ALTA VARIABILIDAD TÉRMICA: ±{temp_std_48h:.1f}°C "
                          f"(Features #6,#8: 7% importancia)")
        
        if hr_max_48h >= 90:
            alertas.append(f"🔵 HUMEDAD MUY ALTA: {hr_max_48h:.0f}% "
                          f"(Efecto sensación térmica)")
        
        if alertas:
            print(f"\n⚠️  ALERTAS CLIMÁTICAS ({len(alertas)}):")
            for alerta in alertas:
                print(f"   {alerta}")
        else:
            print(f"\n✅ CONDICIONES NORMALES (sin alertas)")
        
        # Guardar datos
        df_48h.to_csv('outputs/pronostico_48h_open_meteo.csv', index=False)
        df_daily.to_csv('outputs/pronostico_diario_open_meteo.csv', index=False)
        
        print(f"\n✅ Datos guardados:")
        print(f"   - outputs/pronostico_48h_open_meteo.csv")
        print(f"   - outputs/pronostico_diario_open_meteo.csv")
        
        print(f"\n✅ OPEN-METEO: FUNCIONANDO PERFECTAMENTE")
        
    else:
        print(f"❌ Error: Status {response_meteo.status_code}")
        print(f"   Respuesta: {response_meteo.text}")

except Exception as e:
    print(f"❌ Error en Open-Meteo: {str(e)}")

# ============================================================================
# OPCIÓN 2: OPENWEATHERMAP (GRATUITA CON API KEY)
# ============================================================================

print("\n" + "=" * 80)
print("[OPCIÓN 2] OPENWEATHERMAP (https://openweathermap.org/api)")
print("=" * 80)
print("⚠️  Requiere: API Key gratuita (1000 llamadas/día)")
print("✅ Datos: Temp, HR, Precipitación, Viento, Nubes")
print("✅ Pronóstico: Hasta 5 días adelante (cada 3 horas)")

# Instrucciones para obtener API key
print(f"\n📝 INSTRUCCIONES PARA OBTENER API KEY:")
print(f"   1. Ir a: https://openweathermap.org/api")
print(f"   2. Registrarse (gratis)")
print(f"   3. Ir a: https://home.openweathermap.org/api_keys")
print(f"   4. Copiar API Key")
print(f"   5. Pegar en variable OPENWEATHER_API_KEY abajo")

# DESCOMENTAR Y AGREGAR TU API KEY AQUÍ:
OPENWEATHER_API_KEY = None  # "TU_API_KEY_AQUI"

if OPENWEATHER_API_KEY:
    try:
        url_owm = "https://api.openweathermap.org/data/2.5/forecast"
        
        params_owm = {
            'lat': LAT,
            'lon': LON,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'lang': 'es'
        }
        
        print(f"\n🔄 Consultando OpenWeatherMap para {ESTACION}...")
        response_owm = requests.get(url_owm, params=params_owm, timeout=10)
        
        if response_owm.status_code == 200:
            data_owm = response_owm.json()
            
            print(f"✅ Respuesta exitosa de OpenWeatherMap")
            print(f"\n📊 PRONÓSTICO (próximas 48 horas, cada 3h):")
            print("-" * 80)
            
            forecasts = []
            for item in data_owm['list'][:16]:  # Primeras 48 horas (16 x 3h)
                forecasts.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperatura': item['main']['temp'],
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'humedad_relativa': item['main']['humidity'],
                    'precipitacion': item.get('rain', {}).get('3h', 0),
                    'descripcion': item['weather'][0]['description']
                })
            
            df_owm = pd.DataFrame(forecasts)
            
            for _, row in df_owm.iterrows():
                print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                      f"Temp: {row['temperatura']:>5.1f}°C ({row['temp_min']:.1f}-{row['temp_max']:.1f}) | "
                      f"HR: {row['humedad_relativa']:>3.0f}% | "
                      f"Precip: {row['precipitacion']:>5.1f} mm | "
                      f"{row['descripcion']}")
            
            df_owm.to_csv('outputs/pronostico_openweathermap.csv', index=False)
            print(f"\n✅ OPENWEATHERMAP: FUNCIONANDO")
            
        else:
            print(f"❌ Error: Status {response_owm.status_code}")
            
    except Exception as e:
        print(f"❌ Error en OpenWeatherMap: {str(e)}")
else:
    print(f"\n⏭️  OMITIDO (No API Key configurada)")

# ============================================================================
# OPCIÓN 3: WEATHERAPI (GRATUITA CON API KEY)
# ============================================================================

print("\n" + "=" * 80)
print("[OPCIÓN 3] WEATHERAPI (https://www.weatherapi.com)")
print("=" * 80)
print("⚠️  Requiere: API Key gratuita (1M llamadas/mes)")
print("✅ Datos: Temp, HR, Precipitación, Viento, UV")
print("✅ Pronóstico: Hasta 3 días adelante (cada hora)")

# DESCOMENTAR Y AGREGAR TU API KEY AQUÍ:
WEATHERAPI_KEY = None  # "TU_API_KEY_AQUI"

if WEATHERAPI_KEY:
    try:
        url_wapi = "http://api.weatherapi.com/v1/forecast.json"
        
        params_wapi = {
            'key': WEATHERAPI_KEY,
            'q': f'{LAT},{LON}',
            'days': 3,
            'lang': 'es'
        }
        
        print(f"\n🔄 Consultando WeatherAPI para {ESTACION}...")
        response_wapi = requests.get(url_wapi, params=params_wapi, timeout=10)
        
        if response_wapi.status_code == 200:
            data_wapi = response_wapi.json()
            print(f"✅ WEATHERAPI: FUNCIONANDO")
            
        else:
            print(f"❌ Error: Status {response_wapi.status_code}")
            
    except Exception as e:
        print(f"❌ Error en WeatherAPI: {str(e)}")
else:
    print(f"\n⏭️  OMITIDO (No API Key configurada)")

# ============================================================================
# RESUMEN Y RECOMENDACIONES
# ============================================================================

print("\n" + "=" * 80)
print("📋 RESUMEN Y RECOMENDACIONES")
print("=" * 80)

print(f"\n✅ RECOMENDACIÓN: Usar OPEN-METEO")
print(f"\nRazones:")
print(f"   1. ✅ GRATUITA sin límites ni API key")
print(f"   2. ✅ Datos horarios (mejor resolución)")
print(f"   3. ✅ Pronóstico 16 días adelante")
print(f"   4. ✅ Todos los datos necesarios (Temp, HR, Precip)")
print(f"   5. ✅ Servicio estable y confiable")
print(f"   6. ✅ Documentación excelente")

print(f"\n💡 CÓMO INTEGRAR AL SISTEMA:")
print(f"\n   PASO 1: Captura automática diaria")
print(f"   - Ejecutar este script cada día a las 06:00")
print(f"   - Guardar pronóstico 48h en CSV")
print(f"   - Detectar alertas climáticas automáticamente")

print(f"\n   PASO 2: Incorporar a predicción")
print(f"   - Leer pronóstico desde CSV")
print(f"   - Generar features climáticas proyectadas")
print(f"   - Predecir demanda con clima futuro")

print(f"\n   PASO 3: Dashboard de alertas")
print(f"   - Mostrar condiciones extremas detectadas")
print(f"   - Alertas: lluvia intensa, frío extremo, variabilidad térmica")
print(f"   - Impacto esperado en demanda")

print(f"\n🎯 UMBRALES DE ALERTAS (basados en importancia features):")
print(f"   🔴 CRÍTICO:")
print(f"      • Lluvia intensa: >2 mm/hr (36% importancia)")
print(f"      • Precipitación acumulada 3h: >10 mm (9% importancia)")
print(f"\n   🟡 IMPORTANTE:")
print(f"      • Temperatura muy baja: <12°C (1% importancia + recuperación)")
print(f"      • Variabilidad térmica: std 24h >5°C (7% importancia)")
print(f"      • Precipitación acumulada 24h: >20 mm (1% importancia)")
print(f"\n   🟢 MODERADO:")
print(f"      • Humedad muy alta: >90% (sensación térmica)")
print(f"      • Humedad muy baja: <30% (confort)")

print(f"\n" + "=" * 80)
print("✅ TEST COMPLETADO")
print("=" * 80)
print(f"\nPróximo paso: ¿Quieres que cree el script de integración al sistema?")
print(f"              Incluirá captura automática + alertas + predicción con clima.")
