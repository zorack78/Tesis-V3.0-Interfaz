"""
Interfaz simple para captura y edición de pronóstico climático
Estación Rodelillo, Valparaíso
"""

import gradio as gr
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Configuración Estación Rodelillo
LAT = -33.06528
LON = -71.55639
ESTACION = "Rodelillo, Ad."
TZ = ZoneInfo('America/Santiago')


def capturar_pronostico():
    """Captura pronóstico 3 días desde Open-Meteo"""
    try:
        # Determinar fechas según regla (si >= 17:00, empezar mañana)
        now = datetime.now(TZ)
        if now.hour >= 17:
            start_date = (now + timedelta(days=1)).date()
        else:
            start_date = now.date()
        
        target_dates = [start_date + timedelta(days=i) for i in range(3)]
        start_iso = target_dates[0].isoformat()
        end_iso = target_dates[-1].isoformat()
        
        # Consultar Open-Meteo
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': LAT,
            'longitude': LON,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
            'hourly': 'relative_humidity_2m',
            'timezone': 'America/Santiago',
            'start_date': start_iso,
            'end_date': end_iso,
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return pd.DataFrame(), f"❌ Error API: {response.status_code}"
        
        data = response.json()
        daily = data.get('daily', {})
        hourly = data.get('hourly', {})
        
        # Procesar datos diarios
        df_daily = pd.DataFrame({
            'Fecha': pd.to_datetime(daily.get('time', [])),
            'Temp_Max_°C': daily.get('temperature_2m_max', []),
            'Temp_Min_°C': daily.get('temperature_2m_min', []),
            'Precipitacion_mm': daily.get('precipitation_sum', []),
        })
        
        # Calcular humedad promedio por día desde datos horarios
        if hourly and 'time' in hourly:
            df_hourly = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly['time']),
                'humidity': hourly['relative_humidity_2m']
            })
            df_hourly['date'] = df_hourly['timestamp'].dt.date
            humidity_daily = df_hourly.groupby('date')['humidity'].mean().reset_index()
            humidity_daily.columns = ['date', 'Humedad_%']
            
            # Merge con datos diarios
            df_daily['date'] = df_daily['Fecha'].dt.date
            df_daily = df_daily.merge(humidity_daily, on='date', how='left')
            df_daily = df_daily.drop('date', axis=1)
        else:
            df_daily['Humedad_%'] = None
        
        # Formatear fecha
        df_daily['Fecha'] = df_daily['Fecha'].dt.strftime('%Y-%m-%d')
        
        # Redondear valores
        df_daily['Temp_Max_°C'] = df_daily['Temp_Max_°C'].round(1)
        df_daily['Temp_Min_°C'] = df_daily['Temp_Min_°C'].round(1)
        df_daily['Precipitacion_mm'] = df_daily['Precipitacion_mm'].round(1)
        df_daily['Humedad_%'] = df_daily['Humedad_%'].round(0)
        
        # Mensaje de éxito
        mensaje = f"✅ Pronóstico capturado: {now.strftime('%Y-%m-%d %H:%M')}"
        mensaje += f"\n📍 Estación: {ESTACION}"
        mensaje += f"\n📅 Fechas: {df_daily['Fecha'].iloc[0]} a {df_daily['Fecha'].iloc[-1]}"
        
        return df_daily, mensaje
        
    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {str(e)}"


def generar_tabla_vacia():
    """Genera tabla vacía para edición manual"""
    now = datetime.now(TZ)
    if now.hour >= 17:
        start_date = (now + timedelta(days=1)).date()
    else:
        start_date = now.date()
    
    fechas = [start_date + timedelta(days=i) for i in range(3)]
    
    df = pd.DataFrame({
        'Fecha': [f.strftime('%Y-%m-%d') for f in fechas],
        'Temp_Max_°C': [0.0, 0.0, 0.0],
        'Temp_Min_°C': [0.0, 0.0, 0.0],
        'Precipitacion_mm': [0.0, 0.0, 0.0],
        'Humedad_%': [0.0, 0.0, 0.0],
    })
    
    return df


def guardar_pronostico(df):
    """Guarda pronóstico en CSV"""
    if df is None or df.empty:
        return "⚠️ No hay datos para guardar"
    
    try:
        df.to_csv('outputs/pronostico_manual_3dias.csv', index=False)
        return f"✅ Guardado: outputs/pronostico_manual_3dias.csv ({len(df)} días)"
    except Exception as e:
        return f"❌ Error al guardar: {str(e)}"


def analizar_alertas(df):
    """Detecta condiciones climáticas extremas"""
    if df is None or df.empty:
        return "⚠️ No hay datos para analizar"
    
    alertas = []
    
    for _, row in df.iterrows():
        fecha = row['Fecha']
        temp_min = row['Temp_Min_°C']
        temp_max = row['Temp_Max_°C']
        precip = row['Precipitacion_mm']
        humedad = row['Humedad_%']
        
        # Alertas según importancia de features
        if precip > 5.0:
            alertas.append(f"🔴 {fecha}: Precipitación alta ({precip:.1f} mm)")
        
        if temp_min < 12.0:
            alertas.append(f"🔵 {fecha}: Temperatura baja ({temp_min:.1f}°C)")
        
        if humedad > 85:
            alertas.append(f"💧 {fecha}: Humedad muy alta ({humedad:.0f}%)")
        
        variacion = temp_max - temp_min
        if variacion > 12:
            alertas.append(f"🟠 {fecha}: Alta variación térmica ({variacion:.1f}°C)")
    
    if not alertas:
        return "✅ Sin alertas climáticas (condiciones normales)"
    
    return "\n".join(alertas)


# Crear interfaz
with gr.Blocks(title="Pronóstico Climático - Rodelillo") as app:
    
    gr.Markdown("# 🌤️ Pronóstico Climático - Estación Rodelillo")
    gr.Markdown(f"**Coordenadas:** {LAT}, {LON} | **Altura:** 335 msnm")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Botones de acción
            with gr.Row():
                btn_capturar = gr.Button("📥 Capturar Pronóstico (Open-Meteo)", 
                                        variant="primary", size="lg")
                btn_limpiar = gr.Button("🗑️ Tabla Vacía (Edición Manual)", 
                                       variant="secondary")
            
            # Tabla de pronóstico (editable)
            tabla_pronostico = gr.Dataframe(
                headers=["Fecha", "Temp_Max_°C", "Temp_Min_°C", "Precipitacion_mm", "Humedad_%"],
                datatype=["str", "number", "number", "number", "number"],
                row_count=3,
                col_count=(5, "fixed"),
                interactive=True,
                label="Pronóstico 3 Días (editable)"
            )
            
            # Botón guardar
            btn_guardar = gr.Button("💾 Guardar Pronóstico", variant="secondary")
            
        with gr.Column(scale=1):
            # Panel de información
            gr.Markdown("### 📊 Estado")
            txt_mensaje = gr.Textbox(label="Mensajes", lines=5, interactive=False)
            
            gr.Markdown("### ⚠️ Alertas Climáticas")
            txt_alertas = gr.Textbox(label="Análisis", lines=6, interactive=False)
            
            btn_analizar = gr.Button("🔍 Analizar Alertas", variant="secondary")
    
    gr.Markdown("""
    ---
    **Instrucciones:**
    1. **Capturar Pronóstico**: Obtiene datos automáticamente de Open-Meteo
    2. **Tabla Vacía**: Para ingresar datos manualmente (edita directamente en la tabla)
    3. **Guardar**: Guarda el pronóstico en `outputs/pronostico_manual_3dias.csv`
    4. **Analizar Alertas**: Detecta condiciones extremas según umbrales operacionales
    
    **Umbrales de Alerta:**
    - 🔴 Precipitación > 5 mm/día
    - 🔵 Temperatura mínima < 12°C
    - 💧 Humedad > 85%
    - 🟠 Variación térmica > 12°C/día
    """)
    
    # Eventos
    btn_capturar.click(
        fn=capturar_pronostico,
        inputs=[],
        outputs=[tabla_pronostico, txt_mensaje]
    )
    
    btn_limpiar.click(
        fn=generar_tabla_vacia,
        inputs=[],
        outputs=[tabla_pronostico]
    )
    
    btn_guardar.click(
        fn=guardar_pronostico,
        inputs=[tabla_pronostico],
        outputs=[txt_mensaje]
    )
    
    btn_analizar.click(
        fn=analizar_alertas,
        inputs=[tabla_pronostico],
        outputs=[txt_alertas]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7862, share=False)
