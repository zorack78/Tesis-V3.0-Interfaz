"""
Interfaz Gradio V4 - Predicción de Volumen Total (Demanda de Agua)
Sin considerar datos de Qin, enfocado únicamente en Volumen Total
Incluye variables climáticas y patrones temporales
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class SistemaVolumenTotal:
    def __init__(self):
        self.model = None
        self.features = None
        self.data_historico = None
        self.clima_historico = None
        
    def cargar_modelo_y_datos(self):
        """Carga el modelo entrenado y datos históricos"""
        try:
            # Cargar modelo
            model_path = Path('models/gradio/water_demand_model.pkl')
            if model_path.exists():
                self.model = joblib.load(model_path)
                print("✅ Modelo cargado")
            else:
                raise FileNotFoundError("No se encontró el modelo entrenado")
            
            # Cargar features
            features_path = Path('models/gradio/features.txt')
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.features = [line.strip() for line in f.readlines()]
                print(f"✅ Features cargadas: {len(self.features)}")
            
            # Cargar datos históricos de volumen
            data_path = Path('data/processed/data_processed_complete.csv')
            if data_path.exists():
                self.data_historico = pd.read_csv(data_path)
                self.data_historico.columns = self.data_historico.columns.str.strip()
                self.data_historico['timestamp_utc'] = pd.to_datetime(
                    self.data_historico['timestamp_utc']
                )
                
                # Encontrar columna de volumen
                self.volumen_col = self._encontrar_columna_volumen()
                print(f"✅ Datos históricos cargados: {len(self.data_historico)} registros")
                print(f"✅ Columna volumen: {self.volumen_col}")
            
            # Cargar datos de clima históricos
            clima_path = Path('data/processed/clima_chile_v3.csv')
            if clima_path.exists():
                self.clima_historico = pd.read_csv(clima_path)
                self.clima_historico['timestamp'] = pd.to_datetime(
                    self.clima_historico['timestamp']
                )
                print(f"✅ Clima histórico cargado: {len(self.clima_historico)} registros")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo/datos: {e}")
            return False
    
    def _encontrar_columna_volumen(self):
        """Encuentra la columna de volumen total"""
        possible_names = ['Volumen_Total_m3', 'volumen_total_m3', 
                         ' Volumen_Total_m3', 'Volumen_Total']
        for name in possible_names:
            if name in self.data_historico.columns:
                return name
        return None
    
    def obtener_clima_historico(self, fecha, hora):
        """Obtiene promedios climáticos para una fecha/hora específica"""
        if self.clima_historico is None:
            return {'temp': 15.0, 'HR': 70.0, 'mmhr': 0.0}
        
        # Filtrar por mes y hora similar
        mes = fecha.month
        clima_filtrado = self.clima_historico[
            self.clima_historico['timestamp'].dt.month == mes
        ]
        clima_filtrado = clima_filtrado[
            clima_filtrado['timestamp'].dt.hour == hora
        ]
        
        if len(clima_filtrado) > 0:
            return {
                'temp': float(clima_filtrado['temp'].median()),
                'HR': float(clima_filtrado['HR'].median()),
                'mmhr': float(clima_filtrado['mmhr'].median())
            }
        else:
            # Defaults por mes
            temps_mes = {
                1: 17.0, 2: 17.5, 3: 16.5, 4: 14.5, 5: 12.0, 6: 11.0,
                7: 11.0, 8: 11.0, 9: 12.5, 10: 13.5, 11: 15.5, 12: 16.0
            }
            return {
                'temp': temps_mes.get(mes, 15.0),
                'HR': 75.0,
                'mmhr': 0.0
            }
    
    def crear_features(self, fecha, hora, temperatura, humedad, precipitacion):
        """Crea el conjunto de features para predicción"""
        features = {}
        
        # Debug: verificar tipo de fecha
        if not hasattr(fecha, 'weekday'):
            print(f"❌ ERROR: fecha no es datetime, es {type(fecha)}: {fecha}")
            raise TypeError(f"fecha debe ser datetime, recibió {type(fecha)}")
        
        # Features temporales básicas
        features['hour'] = hora
        features['day_of_week'] = fecha.weekday()
        features['month'] = fecha.month
        features['quarter'] = (fecha.month - 1) // 3 + 1
        
        # Features cíclicas
        features['hour_sin'] = np.sin(2 * np.pi * hora / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hora / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * fecha.weekday() / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * fecha.weekday() / 7)
        
        # Features de categorización de tiempo
        features['is_weekend'] = 1 if fecha.weekday() >= 5 else 0
        features['is_morning'] = 1 if 6 <= hora < 12 else 0
        features['is_afternoon'] = 1 if 12 <= hora < 18 else 0
        features['is_evening'] = 1 if 18 <= hora < 24 else 0
        
        # Estacionalidad
        features['is_summer'] = 1 if fecha.month in [12, 1, 2] else 0
        features['is_peak_hour'] = 1 if 6 <= hora <= 9 else 0
        
        # Feriados y eventos (simplificado)
        features['feriado'] = 0  # Se podría integrar con calendario
        features['es_fin_de_semana'] = features['is_weekend']
        features['temporada_turistica_alta'] = 1 if fecha.month in [12, 1, 2, 7] else 0
        features['hour_x_feriado'] = features['hour'] * features['feriado']
        
        # LAG features - estimados desde datos históricos
        lags = self._estimar_lags(fecha, hora)
        features.update(lags)
        
        return features
    
    def _estimar_lags(self, fecha, hora):
        """Estima LAG features usando patrones históricos"""
        if self.data_historico is None or self.volumen_col is None:
            return {
                'lag_1h': 115000, 'lag_24h': 115000, 'lag_168h': 115000,
                'rolling_mean_24h': 115000, 'rolling_mean_168h': 115000,
                'rolling_std_24h': 5000, 'rolling_std_168h': 5000,
                'diff_24h': 0, 'ratio_vs_24h': 1.0
            }
        
        # Filtrar datos históricos por hora y día de semana similar
        dia_semana = fecha.weekday()
        mes = fecha.month
        
        datos_similares = self.data_historico[
            (self.data_historico['timestamp_utc'].dt.hour == hora) &
            (self.data_historico['timestamp_utc'].dt.weekday == dia_semana) &
            (self.data_historico['timestamp_utc'].dt.month == mes)
        ]
        
        if len(datos_similares) > 0:
            vol_mediano = float(datos_similares[self.volumen_col].median())
            vol_std = float(datos_similares[self.volumen_col].std())
        else:
            # Usar datos de la misma hora solamente
            datos_hora = self.data_historico[
                self.data_historico['timestamp_utc'].dt.hour == hora
            ]
            if len(datos_hora) > 0:
                vol_mediano = float(datos_hora[self.volumen_col].median())
                vol_std = float(datos_hora[self.volumen_col].std())
            else:
                vol_mediano = 115000.0
                vol_std = 5000.0
        
        return {
            'lag_1h': vol_mediano,
            'lag_24h': vol_mediano,
            'lag_168h': vol_mediano,
            'rolling_mean_24h': vol_mediano,
            'rolling_mean_168h': vol_mediano,
            'rolling_std_24h': vol_std,
            'rolling_std_168h': vol_std,
            'diff_24h': 0,
            'ratio_vs_24h': 1.0
        }
    
    def predecir_volumen(self, fecha_str, hora, temperatura=None, 
                        humedad=None, precipitacion=None):
        """Realiza predicción para una fecha/hora específica"""
        try:
            fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
            
            # Obtener datos climáticos (usar históricos si no se proveen)
            if temperatura is None or humedad is None or precipitacion is None:
                try:
                    clima = self.obtener_clima_historico(fecha, hora)
                    temperatura = temperatura if temperatura is not None else clima['temp']
                    humedad = humedad if humedad is not None else clima['HR']
                    precipitacion = precipitacion if precipitacion is not None else clima['mmhr']
                except Exception as e:
                    print(f"Error obteniendo clima: {e}")
                    temperatura = 15.0 if temperatura is None else temperatura
                    humedad = 70.0 if humedad is None else humedad
                    precipitacion = 0.0 if precipitacion is None else precipitacion
            
            # Crear features
            try:
                features_dict = self.crear_features(
                    fecha, hora, temperatura, humedad, precipitacion
                )
            except Exception as e:
                print(f"Error creando features: {e}")
                import traceback
                traceback.print_exc()
                return None, f"❌ Error creando features: {str(e)}"
            
            # Preparar datos para predicción
            X = pd.DataFrame([features_dict])
            
            # Asegurar que tenemos todas las features necesarias
            for feat in self.features:
                if feat not in X.columns:
                    X[feat] = 0
            
            # Ordenar columnas según el modelo
            X = X[self.features]
            
            # Predecir
            prediccion = self.model.predict(X)[0]
            
            # Información adicional
            info = f"""
📅 Fecha: {fecha.strftime('%d/%m/%Y')} ({self._dia_semana_nombre(fecha.weekday())})
⏰ Hora: {hora:02d}:00
🌡️  Temperatura: {temperatura:.1f}°C
💧 Humedad: {humedad:.1f}%
🌧️  Precipitación: {precipitacion:.2f} mm/hr

📊 PREDICCIÓN DE VOLUMEN TOTAL:
   {prediccion:,.0f} m³/hora

🔍 Contexto:
   • Estación: {self._get_estacion(fecha.month)}
   • Período día: {self._get_periodo_dia(hora)}
   • Día: {"Fin de semana" if fecha.weekday() >= 5 else "Laboral"}
"""
            
            return prediccion, info
            
        except Exception as e:
            return None, f"❌ Error en predicción: {str(e)}"
    
    def predecir_dia_completo(self, fecha_str, usar_clima_historico=True):
        """Predice 24 horas completas para una fecha"""
        try:
            # Verificar que el modelo esté cargado
            if self.model is None:
                return None, "❌ Error: Modelo no cargado"
            
            fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
            
            predicciones = []
            horas = []
            temperaturas = []
            errores = []
            
            for hora in range(24):
                try:
                    if usar_clima_historico:
                        clima = self.obtener_clima_historico(fecha, hora)
                        temp = clima['temp']
                        hum = clima['HR']
                        prec = clima['mmhr']
                    else:
                        # Usar valores fijos
                        temp = 15.0
                        hum = 70.0
                        prec = 0.0
                    
                    pred, _ = self.predecir_volumen(
                        fecha_str, hora, temp, hum, prec
                    )
                    
                    if pred is not None:
                        predicciones.append(float(pred))
                        horas.append(hora)
                        temperaturas.append(float(temp))
                    else:
                        error_msg = f"Predicción None para hora {hora}"
                        print(f"⚠️  {error_msg}")
                        errores.append(error_msg)
                except Exception as e:
                    error_msg = f"Error en hora {hora}: {str(e)}"
                    print(f"❌ {error_msg}")
                    errores.append(error_msg)
                    continue
            
            if len(predicciones) == 0:
                error_detail = "\n".join(errores[:5]) if errores else "Sin detalles"
                return None, f"❌ No se pudieron generar predicciones\n\nErrores:\n{error_detail}"
            
            # Crear DataFrame
            df_pred = pd.DataFrame({
                'Hora': horas,
                'Volumen_m3': predicciones,
                'Temperatura_C': temperaturas
            })
            
            # Estadísticas del día
            total_dia = sum(predicciones)
            promedio = np.mean(predicciones)
            maximo = max(predicciones)
            minimo = min(predicciones)
            hora_pico = horas[predicciones.index(maximo)]
            hora_valle = horas[predicciones.index(minimo)]
            
            # Crear gráfica
            fig = self._crear_grafica_dia_completo(
                df_pred, fecha_str, total_dia, hora_pico, hora_valle
            )
            
            info = f"""
📅 PREDICCIÓN DÍA COMPLETO: {fecha.strftime('%d/%m/%Y')}
📊 Día: {self._dia_semana_nombre(fecha.weekday())}
🌡️  Clima: {'Histórico' if usar_clima_historico else 'Estándar'}

📈 ESTADÍSTICAS DEL DÍA:
   • Volumen total: {total_dia:,.0f} m³
   • Promedio hora: {promedio:,.0f} m³/hora
   • Hora pico: {hora_pico:02d}:00 → {maximo:,.0f} m³
   • Hora valle: {hora_valle:02d}:00 → {minimo:,.0f} m³
   • Variación: {((maximo-minimo)/minimo*100):.1f}%

⏰ PERÍODOS DEL DÍA:
   • Madrugada (00-06): {sum(predicciones[0:6]):,.0f} m³
   • Mañana (06-12): {sum(predicciones[6:12]):,.0f} m³
   • Tarde (12-18): {sum(predicciones[12:18]):,.0f} m³
   • Noche (18-24): {sum(predicciones[18:24]):,.0f} m³
"""
            
            return fig, info
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def _crear_grafica_dia_completo(self, df, fecha_str, total, hora_pico, hora_valle):
        """Crea gráfica de predicción de 24 horas"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfica 1: Volumen por hora
        ax1.plot(df['Hora'], df['Volumen_m3'], 
                marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax1.axvline(hora_pico, color='red', linestyle='--', 
                   alpha=0.5, label=f'Hora pico: {hora_pico:02d}:00')
        ax1.axvline(hora_valle, color='green', linestyle='--', 
                   alpha=0.5, label=f'Hora valle: {hora_valle:02d}:00')
        ax1.fill_between(df['Hora'], df['Volumen_m3'], alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Hora del día', fontsize=12)
        ax1.set_ylabel('Volumen (m³/hora)', fontsize=12)
        ax1.set_title(f'Predicción Volumen Total - {fecha_str}\nTotal día: {total:,.0f} m³', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(range(0, 24, 2))
        
        # Gráfica 2: Temperatura
        ax2.plot(df['Hora'], df['Temperatura_C'], 
                marker='s', linewidth=2, markersize=4, color='#A23B72')
        ax2.fill_between(df['Hora'], df['Temperatura_C'], alpha=0.2, color='#A23B72')
        ax2.set_xlabel('Hora del día', fontsize=12)
        ax2.set_ylabel('Temperatura (°C)', fontsize=12)
        ax2.set_title('Temperatura estimada', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        return fig
    
    def _dia_semana_nombre(self, dia):
        """Convierte número de día a nombre"""
        dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 
                'Viernes', 'Sábado', 'Domingo']
        return dias[dia]
    
    def _get_estacion(self, mes):
        """Obtiene la estación del año"""
        if mes in [12, 1, 2]:
            return 'Verano'
        elif mes in [3, 4, 5]:
            return 'Otoño'
        elif mes in [6, 7, 8]:
            return 'Invierno'
        else:
            return 'Primavera'
    
    def _get_periodo_dia(self, hora):
        """Obtiene el período del día"""
        if 0 <= hora < 6:
            return 'Madrugada'
        elif 6 <= hora < 12:
            return 'Mañana'
        elif 12 <= hora < 18:
            return 'Tarde'
        else:
            return 'Noche'

# Crear instancia del sistema
sistema = SistemaVolumenTotal()

def inicializar_sistema():
    """Inicializa el sistema al arrancar"""
    if sistema.cargar_modelo_y_datos():
        return "✅ Sistema inicializado correctamente"
    else:
        return "❌ Error al inicializar sistema"

def wrapper_prediccion_simple(fecha, hora, temp, hum, precip):
    """Wrapper para predicción simple"""
    try:
        # Convertir hora a int si viene como float
        hora = int(hora)
        pred, info = sistema.predecir_volumen(fecha, hora, temp, hum, precip)
        return info
    except Exception as e:
        import traceback
        return f"❌ Error en predicción: {str(e)}\n\nDetalle:\n{traceback.format_exc()}"

def wrapper_prediccion_dia(fecha, usar_historico):
    """Wrapper para predicción de día completo"""
    try:
        fig, info = sistema.predecir_dia_completo(fecha, usar_historico)
        return fig, info
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg

# Crear interfaz Gradio
def crear_interfaz():
    with gr.Blocks(title="Predicción Volumen Total V4", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 💧 Sistema de Predicción de Volumen Total (Demanda de Agua)
        ## Gran Valparaíso - Chile
        
        **Versión 4.0** - Predicción basada únicamente en Volumen Total
        """)
        
        estado = gr.Textbox(label="Estado del Sistema", 
                           value=inicializar_sistema(),
                           interactive=False)
        
        with gr.Tabs():
            # TAB 1: Predicción Simple
            with gr.Tab("🎯 Predicción Puntual"):
                gr.Markdown("### Predice el volumen para una hora específica")
                
                with gr.Row():
                    with gr.Column():
                        fecha_input = gr.Textbox(
                            label="Fecha (YYYY-MM-DD)",
                            value="2025-12-25",
                            placeholder="2025-12-25"
                        )
                        hora_input = gr.Slider(
                            0, 23, value=8, step=1, 
                            label="Hora (0-23)"
                        )
                        
                        gr.Markdown("**Condiciones Climáticas** (opcional)")
                        temp_input = gr.Slider(
                            0, 40, value=15, step=0.5, 
                            label="Temperatura (°C)"
                        )
                        hum_input = gr.Slider(
                            0, 100, value=70, step=1, 
                            label="Humedad Relativa (%)"
                        )
                        precip_input = gr.Slider(
                            0, 10, value=0, step=0.1, 
                            label="Precipitación (mm/hr)"
                        )
                        
                        btn_predecir = gr.Button("🔮 Predecir", variant="primary")
                    
                    with gr.Column():
                        resultado_simple = gr.Textbox(
                            label="Resultado de Predicción",
                            lines=15,
                            interactive=False
                        )
                
                btn_predecir.click(
                    fn=wrapper_prediccion_simple,
                    inputs=[fecha_input, hora_input, temp_input, 
                           hum_input, precip_input],
                    outputs=resultado_simple
                )
            
            # TAB 2: Predicción Día Completo
            with gr.Tab("📅 Predicción Día Completo"):
                gr.Markdown("### Predice las 24 horas de un día completo")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        fecha_dia_input = gr.Textbox(
                            label="Fecha (YYYY-MM-DD)",
                            value="2025-12-25",
                            placeholder="2025-12-25"
                        )
                        usar_historico = gr.Checkbox(
                            label="Usar clima histórico",
                            value=True,
                            info="Usa promedios históricos de temperatura"
                        )
                        btn_predecir_dia = gr.Button(
                            "📊 Predecir Día Completo", 
                            variant="primary"
                        )
                    
                    with gr.Column(scale=2):
                        grafica_dia = gr.Plot(label="Gráfica de Predicción")
                
                resultado_dia = gr.Textbox(
                    label="Estadísticas del Día",
                    lines=12,
                    interactive=False
                )
                
                btn_predecir_dia.click(
                    fn=wrapper_prediccion_dia,
                    inputs=[fecha_dia_input, usar_historico],
                    outputs=[grafica_dia, resultado_dia]
                )
            
            # TAB 3: Información
            with gr.Tab("ℹ️ Información"):
                gr.Markdown("""
                ### Acerca del Sistema
                
                **Modelo:** XGBoost Regressor
                **Variable Objetivo:** Volumen Total (m³/hora)
                **Sin uso de Qin (caudal de entrada)**
                
                #### 📊 Features del Modelo:
                - ⏰ **Temporales:** Hora, día semana, mes, trimestre
                - 🔄 **Cíclicas:** Seno/coseno de hora y día
                - 📅 **Categóricas:** Fin de semana, período del día
                - 📈 **LAG:** Valores históricos (1h, 24h, 168h)
                - 📊 **Rolling:** Medias y desviaciones móviles
                - 🎉 **Eventos:** Feriados, temporada turística
                
                #### 🎯 Características:
                - ✅ Predicción basada solo en Volumen Total
                - ✅ Integración con patrones climáticos históricos
                - ✅ Considera estacionalidad y eventos especiales
                - ✅ Predicciones hora por hora o día completo
                
                #### 📈 Patrones Esperados:
                - **Verano:** Mayor consumo (altas temperaturas)
                - **Invierno:** Menor consumo (bajas temperaturas)
                - **Hora pico:** Típicamente 7:00-9:00 AM
                - **Hora valle:** Típicamente 21:00-00:00
                
                ---
                **Desarrollado para:** Gran Valparaíso, Chile  
                **Versión:** 4.0 (Sin Qin)
                """)
        
    return app

# Lanzar aplicación
if __name__ == "__main__":
    app = crear_interfaz()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,  # Cambiado a 7863
        share=False,
        show_error=True
    )
