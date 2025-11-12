"""
Captura pronóstico 3 días para Estación Rodelillo (Valparaíso)
Reglas:
 - Si la consulta se realiza después de las 17:00 hora local (America/Santiago),
   comenzar el pronóstico desde el día siguiente.
 - Captura datos actuales y pronóstico para 3 días: temp_max, temp_min, humedad, precip.
 - Fuente principal: Open-Meteo (recomendada, sin API key)
 - Fuente secundaria: OpenWeather (opcional, requiere API key)

Salida:
 - outputs/pronostico_3dias_open_meteo.csv
 - outputs/pronostico_3dias_openweather.csv (si API key)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import os

# Config
LAT = -33.06528
LON = -71.55639
ESTACION = "Rodelillo, Ad."
TZ = ZoneInfo('America/Santiago')

# Optional OpenWeather API key (paste your key here to enable comparison)
OPENWEATHER_API_KEY = None  # "TU_API_KEY_AQUI"

os.makedirs('outputs', exist_ok=True)

# Determine start date according to rule
now = datetime.now(TZ)
cutoff_hour = 17
if now.hour >= cutoff_hour:
    start_date = (now + timedelta(days=1)).date()
else:
    start_date = now.date()

target_dates = [start_date + timedelta(days=i) for i in range(3)]
start_iso = target_dates[0].isoformat()
end_iso = (target_dates[-1] + timedelta(days=1)).isoformat()  # inclusive range for hourly aggregation

print("="*80)
print(f"Consulta: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Estación: {ESTACION} | Coordenadas: {LAT}, {LON}")
print(f"Inicio pronóstico usado: {start_iso}")
print("Fechas objetivo:")
for d in target_dates:
    print(f" - {d.isoformat()}")
print("="*80)

# ------------------
# Open-Meteo (daily + hourly)
# ------------------
print('\n[Fuente] Open-Meteo (sin API key)')
url = 'https://api.open-meteo.com/v1/forecast'
params = {
    'latitude': LAT,
    'longitude': LON,
    'hourly': 'temperature_2m,relative_humidity_2m,precipitation',
    'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max',
    'timezone': 'America/Santiago',
    'start_date': start_iso,
    'end_date': (target_dates[-1]).isoformat(),
}

resp = requests.get(url, params=params, timeout=15)
if resp.status_code != 200:
    print(f"ERROR Open-Meteo: status {resp.status_code}")
    print(resp.text)
    open_meteo_df_daily = None
else:
    j = resp.json()
    # daily
    daily = j.get('daily', {})
    df_daily = pd.DataFrame({
        'date': pd.to_datetime(daily.get('time', [])),
        'temp_max': daily.get('temperature_2m_max', []),
        'temp_min': daily.get('temperature_2m_min', []),
        'precip_total': daily.get('precipitation_sum', []),
        'precip_prob_max': daily.get('precipitation_probability_max', []),
    })

    # hourly -> aggregate per date for humidity (mean/min/max) and precipitation (sum)
    hourly = j.get('hourly', {})
    df_hourly = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly.get('time', [])),
        'temperature': hourly.get('temperature_2m', []),
        'humidity': hourly.get('relative_humidity_2m', []),
        'precip_hour': hourly.get('precipitation', []),
    })
    if not df_hourly.empty:
        df_hourly['date'] = df_hourly['timestamp'].dt.date
        agg = df_hourly.groupby('date').agg(
            humidity_mean=('humidity', 'mean'),
            humidity_min=('humidity', 'min'),
            humidity_max=('humidity', 'max'),
            precip_sum_h=('precip_hour', 'sum'),
            temp_mean_h=('temperature', 'mean')
        ).reset_index()
        agg['date'] = pd.to_datetime(agg['date']).dt.date
        # merge with daily
        df_daily['date'] = df_daily['date'].dt.date
        open_meteo_df_daily = df_daily.merge(agg, on='date', how='left')
    else:
        open_meteo_df_daily = df_daily.copy()

    # Keep only target_dates
    open_meteo_df_daily = open_meteo_df_daily[open_meteo_df_daily['date'].isin(target_dates)]
    open_meteo_df_daily = open_meteo_df_daily.sort_values('date')
    open_meteo_df_daily.to_csv('outputs/pronostico_3dias_open_meteo.csv', index=False)
    print('Open-Meteo: datos guardados en outputs/pronostico_3dias_open_meteo.csv')

# ------------------
# OpenWeather (optional) - aggregate per calendar date local time
# ------------------
openweather_df_daily = None
if OPENWEATHER_API_KEY:
    print('\n[Fuente] OpenWeatherMap (agregando por día)')
    url_owm = 'https://api.openweathermap.org/data/2.5/forecast'
    params_owm = {
        'lat': LAT,
        'lon': LON,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }
    try:
        r2 = requests.get(url_owm, params=params_owm, timeout=15)
        if r2.status_code == 200:
            j2 = r2.json()
            items = j2.get('list', [])
            rows = []
            for it in items:
                ts = datetime.utcfromtimestamp(it['dt']).replace(tzinfo=ZoneInfo('UTC')).astimezone(TZ)
                rows.append({
                    'timestamp': ts,
                    'temp': it['main']['temp'],
                    'temp_min': it['main'].get('temp_min'),
                    'temp_max': it['main'].get('temp_max'),
                    'humidity': it['main'].get('humidity'),
                    'precip_3h': it.get('rain', {}).get('3h', 0) if 'rain' in it else 0,
                })
            df_owm = pd.DataFrame(rows)
            df_owm['date'] = df_owm['timestamp'].dt.date
            owm_agg = df_owm.groupby('date').agg(
                temp_max=('temp', 'max'),
                temp_min=('temp', 'min'),
                humidity_mean=('humidity', 'mean'),
                precip_sum_owm=('precip_3h', 'sum')
            ).reset_index()
            owm_agg['date'] = pd.to_datetime(owm_agg['date']).dt.date
            openweather_df_daily = owm_agg[owm_agg['date'].isin(target_dates)].sort_values('date')
            openweather_df_daily.to_csv('outputs/pronostico_3dias_openweather.csv', index=False)
            print('OpenWeather: datos guardados en outputs/pronostico_3dias_openweather.csv')
        else:
            print(f'OpenWeather error status {r2.status_code}')
    except Exception as e:
        print('Error OpenWeather:', e)
else:
    print('\nOpenWeather omitido (no API key configurada)')

# ------------------
# Mostrar comparativa (Open-Meteo vs OpenWeather)
# ------------------
print('\n' + '='*80)
print('Comparativa por día (Open-Meteo  |  OpenWeather)')
print('='*80)

om = open_meteo_df_daily[['date', 'temp_min', 'temp_max', 'humidity_mean', 'precip_total']].copy()
om.columns = ['date', 'om_temp_min', 'om_temp_max', 'om_humidity_mean', 'om_precip_total']

if openweather_df_daily is not None:
    ow = openweather_df_daily[['date', 'temp_min', 'temp_max', 'humidity_mean', 'precip_sum_owm']].copy()
    ow.columns = ['date', 'ow_temp_min', 'ow_temp_max', 'ow_humidity_mean', 'ow_precip_total']
    merged = om.merge(ow, on='date', how='left')
else:
    merged = om.copy()

print(merged.to_string(index=False))

# Simple discrepancy report
if openweather_df_daily is not None:
    print('\nDiferencias (Open-Meteo - OpenWeather):')
    merged['d_temp_min'] = merged['om_temp_min'] - merged['ow_temp_min']
    merged['d_temp_max'] = merged['om_temp_max'] - merged['ow_temp_max']
    merged['d_precip'] = merged['om_precip_total'] - merged['ow_precip_total']
    print(merged[['date', 'd_temp_min', 'd_temp_max', 'd_precip']].to_string(index=False))
    merged.to_csv('outputs/pronostico_3dias_comparativa.csv', index=False)
    print('\nComparativa guardada: outputs/pronostico_3dias_comparativa.csv')

print('\nProceso completado.')
print('Archivos generados en outputs/')

print('\nSiguiente paso: revisar los CSV y confirmar si los valores coinciden con tus consultas manuales en otras fuentes.')
