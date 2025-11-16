#!/usr/bin/env python3
"""
Interfaz Gradio V3.0 COMPLETA - Sistema Predictivo ESVAL
=========================================================
Integra:
- Modelo Forecasting V3.0 con temperatura
- 4 tabs de predicción con temperatura (1h, 24h, 72h, testing)
- Pronóstico climático en header
- Dashboard operativo
- Análisis histórico
- INFO completa del modelo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import gradio as gr
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para Gradio
plt.style.use('seaborn-v0_8')


class InterfazGradioV3Completa:
    """Interfaz web completa para sistema predictivo ESVAL V3.0"""
    
    def __init__(self):
        self.df_metricas = None
        self.df_volumen = None
        self.df_clima = None
        self.sistema_info = None
        self.modelo_prediccion = None
        self.features_modelo = None
        self.umbrales = None
        self.metricas_modelo = None
        self.data_historico = None
        
    def cargar_datos(self):
        """Carga todos los datos procesados V3.0"""
        try:
            data_path = Path("data/processed")
            
            # Cargar datos principales
            self.df_metricas = pd.read_csv(data_path / "metricas_diarias_v3.csv")
            self.df_metricas['fecha'] = pd.to_datetime(self.df_metricas['fecha'])
            
            self.df_volumen = pd.read_csv(data_path / "volumen_total_chile_v3.csv")
            self.df_volumen['timestamp_chile'] = pd.to_datetime(self.df_volumen['timestamp_chile'])
            
            self.df_clima = pd.read_csv(data_path / "clima_chile_v3.csv")
            self.df_clima['timestamp_chile'] = pd.to_datetime(self.df_clima['timestamp_chile'])
            
            with open(data_path / "sistema_info_v3.json", 'r') as f:
                self.sistema_info = json.load(f)
            
            # Cargar datos históricos completos para predicciones
            dataset_path = data_path / "dataset_features_completo.csv"
            if dataset_path.exists():
                self.data_historico = pd.read_csv(dataset_path)
                self.data_historico['timestamp'] = pd.to_datetime(self.data_historico['timestamp'])
                print(f"✅ Dataset histórico cargado: {len(self.data_historico)} registros")
                
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
            
    def cargar_modelo_forecasting(self):
        """Carga el modelo Forecasting V3.0 con temperatura"""
        print("🤖 Cargando modelo Forecasting V3.0...")
        
        model_path_v3 = Path("models/forecasting")
        model_file = model_path_v3 / "modelo_forecasting_xgboost.pkl"
        features_file = model_path_v3 / "features.txt"
        metricas_file = model_path_v3 / "metricas.json"
        umbrales_file = model_path_v3 / "umbrales.pkl"
        
        if model_file.exists():
            try:
                # Cargar modelo
                self.modelo_prediccion = joblib.load(model_file)
                
                # Cargar features
                with open(features_file, 'r') as f:
                    self.features_modelo = [line.strip() for line in f]
                
                # Cargar métricas
                with open(metricas_file, 'r') as f:
                    self.metricas_modelo = json.load(f)
                
                # Cargar umbrales
                self.umbrales = joblib.load(umbrales_file)
                
                print(f"✅ Modelo Forecasting V3.0 cargado")
                print(f"   📋 Features: {len(self.features_modelo)}")
                print(f"   🎯 RMSE: {self.metricas_modelo['rmse']:.2f} m³/hr")
                print(f"   🎯 R²: {self.metricas_modelo['r2']:.4f}")
                
                return True
                
            except Exception as e:
                print(f"⚠️ Error cargando modelo V3.0: {e}")
                return False
        else:
            print("⚠️ Modelo Forecasting V3.0 no encontrado")
            return False
    
    def obtener_pronostico_climatico(self):
        """Obtiene pronóstico climático de 3 días"""
        try:
            pronostico_path = Path("outputs/pronostico_3dias_open_meteo.csv")
            if pronostico_path.exists():
                df_pron = pd.read_csv(pronostico_path)
                return df_pron
            return None
        except:
            return None
    
    def crear_header_con_pronostico(self):
        """Crea header azul ESVAL con pronóstico climático compacto"""
        pronostico = self.obtener_pronostico_climatico()
        
        header_html = """
        <div style="background: linear-gradient(135deg, #0066cc 0%, #004d99 100%); 
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; font-size: 28px;">🚰 Sistema Predictivo ESVAL V3.0</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">
                        Predicción de Demanda Agua Potable - Gran Valparaíso
                    </p>
                </div>
        """
        
        if pronostico is not None and len(pronostico) > 0:
            # Agregar pronóstico compacto
            header_html += """
                <div style="background: rgba(255,255,255,0.15); padding: 12px; 
                            border-radius: 8px; min-width: 300px;">
                    <div style="font-size: 12px; margin-bottom: 8px; opacity: 0.9;">
                        🌤️ Pronóstico 72h:
                    </div>
                    <div style="display: flex; gap: 15px; justify-content: space-around;">
            """
            
            for i in range(min(3, len(pronostico))):
                row = pronostico.iloc[i]
                temp = row.get('temperature_2m', 20)
                cond = row.get('weather_condition', 'Despejado')
                
                # Emoji según condición
                emoji = "☀️"
                if "nublado" in cond.lower() or "cloud" in cond.lower():
                    emoji = "☁️"
                elif "lluvia" in cond.lower() or "rain" in cond.lower():
                    emoji = "🌧️"
                
                header_html += f"""
                    <div style="text-align: center;">
                        <div style="font-size: 24px;">{emoji}</div>
                        <div style="font-size: 18px; font-weight: bold;">{temp:.0f}°C</div>
                        <div style="font-size: 11px; opacity: 0.8;">Día {i+1}</div>
                    </div>
                """
            
            header_html += """
                    </div>
                </div>
            """
        
        header_html += """
            </div>
        </div>
        """
        
        return header_html
    
    def predecir_con_temperatura(self, fecha_str, hora, temperatura):
        """Predice demanda usando modelo Forecasting V3.0"""
        try:
            if self.modelo_prediccion is None or self.data_historico is None:
                return "⚠️ Modelo no disponible", None
            
            fecha = pd.to_datetime(fecha_str, format="%d/%m/%Y")
            
            # Obtener última fila de datos históricos como base
            ultima_fila = self.data_historico.iloc[-1:].copy()
            
            # Actualizar features temporales
            ultima_fila['hora'] = hora
            ultima_fila['dia_semana'] = fecha.weekday()
            ultima_fila['mes'] = fecha.month
            ultima_fila['es_fin_de_semana'] = 1 if fecha.weekday() >= 5 else 0
            
            # Actualizar temperatura si está en las features
            if 'clima_temp_c' in ultima_fila.columns:
                ultima_fila['clima_temp_c'] = temperatura
            
            # Seleccionar features del modelo
            features_disponibles = [f for f in self.features_modelo if f in ultima_fila.columns]
            
            if len(features_disponibles) < len(self.features_modelo) * 0.8:
                return f"⚠️ Faltan {len(self.features_modelo) - len(features_disponibles)} features", None
            
            X = ultima_fila[features_disponibles]
            
            # Predecir
            pred = self.modelo_prediccion.predict(X)[0]
            
            # Interpretar resultado
            if pred < -3000:
                nivel = "🔴 MUY ALTA"
                alerta = "⚠️ RIESGO DÉFICIT"
            elif pred < -1000:
                nivel = "🟡 ALTA"
                alerta = "Monitorear"
            elif pred < 0:
                nivel = "🟢 NORMAL"
                alerta = "OK"
            elif pred < 1000:
                nivel = "🟢 RECUPERACIÓN"
                alerta = "Sistema cargando"
            else:
                nivel = "🔵 FUERTE RECUPERACIÓN"
                alerta = "Llenado activo"
            
            resultado = f"""
            ### Predicción para {fecha_str} a las {hora:02d}:00
            
            **🌡️ Temperatura:** {temperatura:.1f}°C
            
            **📊 Q_net Predicho:** {pred:.0f} m³/hr
            
            **📈 Nivel de Demanda:** {nivel}
            
            **⚠️ Estado:** {alerta}
            
            ---
            *Modelo: Forecasting V3.0 | R² = {self.metricas_modelo['r2']:.4f}*
            """
            
            return resultado, pred
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def crear_interfaz(self):
        """Crea la interfaz Gradio completa"""
        
        with gr.Blocks(title="ESVAL V3.0 - Sistema Predictivo", theme=gr.themes.Soft()) as demo:
            
            # Header con pronóstico
            gr.HTML(self.crear_header_con_pronostico())
            
            with gr.Tabs():
                
                # ========== TAB 1: Predicción 1 Hora ==========
                with gr.Tab("⏰ Predicción 1 Hora"):
                    gr.Markdown("### Predice la demanda para una hora específica con temperatura")
                    
                    with gr.Row():
                        with gr.Column():
                            input_fecha_1h = gr.Textbox(
                                label="Fecha",
                                placeholder="DD/MM/YYYY",
                                value=datetime.now().strftime("%d/%m/%Y")
                            )
                            input_hora_1h = gr.Slider(
                                label="Hora del Día",
                                minimum=0,
                                maximum=23,
                                step=1,
                                value=12
                            )
                            input_temp_1h = gr.Number(
                                label="🌡️ Temperatura (°C)",
                                value=18.0,
                                minimum=-10,
                                maximum=45
                            )
                            btn_pred_1h = gr.Button("🚀 Predecir", variant="primary")
                        
                        with gr.Column():
                            output_1h = gr.Markdown()
                    
                    def wrapper_pred_1h(fecha, hora, temp):
                        resultado, pred = self.predecir_con_temperatura(fecha, int(hora), temp)
                        return resultado
                    
                    btn_pred_1h.click(
                        fn=wrapper_pred_1h,
                        inputs=[input_fecha_1h, input_hora_1h, input_temp_1h],
                        outputs=[output_1h]
                    )
                
                # ========== TAB 2: Predicción 24 Horas ==========
                with gr.Tab("📅 Predicción 24 Horas"):
                    gr.Markdown("### Predice la demanda para las próximas 24 horas con temperatura constante")
                    
                    with gr.Row():
                        with gr.Column():
                            input_fecha_24h = gr.Textbox(
                                label="Fecha Inicio",
                                placeholder="DD/MM/YYYY",
                                value=datetime.now().strftime("%d/%m/%Y")
                            )
                            input_temp_24h = gr.Number(
                                label="🌡️ Temperatura Promedio del Día (°C)",
                                value=20.0,
                                minimum=-10,
                                maximum=45
                            )
                            btn_pred_24h = gr.Button("🚀 Generar Predicción 24h", variant="primary")
                        
                        with gr.Column():
                            output_24h = gr.Markdown()
                    
                    output_plot_24h = gr.Plot(label="Gráfica 24 Horas")
                    
                    def wrapper_pred_24h(fecha, temp):
                        try:
                            predicciones = []
                            horas = list(range(24))
                            
                            for hora in horas:
                                _, pred = self.predecir_con_temperatura(fecha, hora, temp)
                                if pred is not None:
                                    predicciones.append(pred)
                                else:
                                    predicciones.append(0)
                            
                            # Crear gráfica
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=horas,
                                y=predicciones,
                                mode='lines+markers',
                                name='Predicción',
                                line=dict(color='#0066cc', width=3)
                            ))
                            
                            fig.update_layout(
                                title=f'Predicción 24h - Temp: {temp:.1f}°C',
                                xaxis_title='Hora del Día',
                                yaxis_title='Q_net (m³/hr)',
                                hovermode='x unified',
                                template='plotly_white'
                            )
                            
                            # Resumen
                            demanda_total = abs(sum([p for p in predicciones if p < 0]))
                            recup_total = sum([p for p in predicciones if p > 0])
                            
                            resumen = f"""
                            ### Resumen 24 Horas - {fecha}
                            
                            **🌡️ Temperatura:** {temp:.1f}°C
                            
                            **📊 Demanda Total:** {demanda_total:,.0f} m³
                            **📈 Recuperación Total:** {recup_total:,.0f} m³
                            **⚖️ Balance:** {recup_total - demanda_total:,.0f} m³
                            
                            **📍 Hora Pico Demanda:** {horas[np.argmin(predicciones)]}:00 ({min(predicciones):.0f} m³/hr)
                            **📍 Hora Máx Recuperación:** {horas[np.argmax(predicciones)]}:00 ({max(predicciones):.0f} m³/hr)
                            """
                            
                            return resumen, fig
                            
                        except Exception as e:
                            return f"❌ Error: {str(e)}", None
                    
                    btn_pred_24h.click(
                        fn=wrapper_pred_24h,
                        inputs=[input_fecha_24h, input_temp_24h],
                        outputs=[output_24h, output_plot_24h]
                    )
                
                # ========== TAB 3: Predicción 72 Horas ==========
                with gr.Tab("🔮 Predicción 72 Horas"):
                    gr.Markdown("### Predice la demanda para 3 días con temperatura diferente por día")
                    
                    with gr.Row():
                        with gr.Column():
                            input_fecha_72h = gr.Textbox(
                                label="Fecha Inicio",
                                placeholder="DD/MM/YYYY",
                                value=datetime.now().strftime("%d/%m/%Y")
                            )
                            
                            gr.Markdown("**🌡️ Temperaturas por Día:**")
                            
                            input_temp_dia1 = gr.Number(
                                label="Día 1 - Temperatura (°C)",
                                value=18.0,
                                minimum=-10,
                                maximum=45
                            )
                            input_temp_dia2 = gr.Number(
                                label="Día 2 - Temperatura (°C)",
                                value=20.0,
                                minimum=-10,
                                maximum=45
                            )
                            input_temp_dia3 = gr.Number(
                                label="Día 3 - Temperatura (°C)",
                                value=22.0,
                                minimum=-10,
                                maximum=45
                            )
                            
                            btn_pred_72h = gr.Button("🚀 Generar Predicción 72h", variant="primary")
                        
                        with gr.Column():
                            output_72h = gr.Markdown()
                    
                    output_plot_72h = gr.Plot(label="Gráfica 3 Días")
                    
                    def wrapper_pred_72h(fecha, temp1, temp2, temp3):
                        try:
                            fecha_base = pd.to_datetime(fecha, format="%d/%m/%Y")
                            temperaturas = [temp1, temp2, temp3]
                            
                            predicciones = []
                            horas_totales = []
                            
                            for dia in range(3):
                                fecha_dia = fecha_base + timedelta(days=dia)
                                fecha_str = fecha_dia.strftime("%d/%m/%Y")
                                temp_dia = temperaturas[dia]
                                
                                for hora in range(24):
                                    _, pred = self.predecir_con_temperatura(fecha_str, hora, temp_dia)
                                    if pred is not None:
                                        predicciones.append(pred)
                                    else:
                                        predicciones.append(0)
                                    horas_totales.append(dia * 24 + hora)
                            
                            # Crear gráfica
                            fig = go.Figure()
                            
                            # Línea de predicción
                            fig.add_trace(go.Scatter(
                                x=horas_totales,
                                y=predicciones,
                                mode='lines+markers',
                                name='Predicción',
                                line=dict(color='#0066cc', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Separadores de días
                            for dia in range(1, 3):
                                fig.add_vline(x=dia*24, line_dash="dash", line_color="gray", opacity=0.5)
                            
                            fig.update_layout(
                                title='Predicción 72 Horas (3 Días)',
                                xaxis_title='Hora Total',
                                yaxis_title='Q_net (m³/hr)',
                                hovermode='x unified',
                                template='plotly_white'
                            )
                            
                            # Resumen por día
                            resumen_dias = []
                            for dia in range(3):
                                inicio = dia * 24
                                fin = (dia + 1) * 24
                                preds_dia = predicciones[inicio:fin]
                                
                                demanda = abs(sum([p for p in preds_dia if p < 0]))
                                recup = sum([p for p in preds_dia if p > 0])
                                balance = recup - demanda
                                
                                resumen_dias.append(f"""
                            **Día {dia+1}** ({temperaturas[dia]:.1f}°C):
                            - Demanda: {demanda:,.0f} m³
                            - Recuperación: {recup:,.0f} m³
                            - Balance: {balance:+,.0f} m³
                                """)
                            
                            resumen = f"""
                            ### Resumen 72 Horas - {fecha}
                            
                            {''.join(resumen_dias)}
                            
                            ---
                            **Balance Total 3 Días:** {sum([p for p in predicciones if p > 0]) - abs(sum([p for p in predicciones if p < 0])):+,.0f} m³
                            """
                            
                            return resumen, fig
                            
                        except Exception as e:
                            return f"❌ Error: {str(e)}", None
                    
                    btn_pred_72h.click(
                        fn=wrapper_pred_72h,
                        inputs=[input_fecha_72h, input_temp_dia1, input_temp_dia2, input_temp_dia3],
                        outputs=[output_72h, output_plot_72h]
                    )
                
                # ========== TAB 4: Evaluación Testing ==========
                with gr.Tab("📈 Evaluación Testing"):
                    gr.Markdown("### Evaluación del modelo en periodo de testing")
                    gr.Markdown("*Usa datos reales del periodo de test para validar performance*")
                    
                    btn_testing = gr.Button("🎯 Generar Evaluación", variant="primary")
                    output_testing = gr.Markdown()
                    output_plot_testing = gr.Plot(label="Resultados Testing")
                    
                    def generar_evaluacion_testing():
                        try:
                            if self.modelo_prediccion is None or self.data_historico is None:
                                return "⚠️ Modelo o datos no disponibles", None
                            
                            # Usar últimas 240 horas (10 días) como test
                            df_test = self.data_historico.tail(240).copy()
                            
                            # Obtener features
                            features_disponibles = [f for f in self.features_modelo if f in df_test.columns]
                            X_test = df_test[features_disponibles]
                            y_real = df_test['Q_net_m3h'].values
                            
                            # Predecir
                            y_pred = self.modelo_prediccion.predict(X_test)
                            
                            # Calcular métricas
                            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                            
                            mae = mean_absolute_error(y_real, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_real, y_pred))
                            r2 = r2_score(y_real, y_pred)
                            
                            # Crear gráfica
                            fig = go.Figure()
                            
                            # Real
                            fig.add_trace(go.Scatter(
                                x=list(range(len(y_real))),
                                y=y_real,
                                mode='lines',
                                name='Real',
                                line=dict(color='#333333', width=2)
                            ))
                            
                            # Predicción
                            fig.add_trace(go.Scatter(
                                x=list(range(len(y_pred))),
                                y=y_pred,
                                mode='lines',
                                name='Predicción',
                                line=dict(color='#0066cc', width=2, dash='dot')
                            ))
                            
                            fig.update_layout(
                                title='Evaluación Testing: Real vs Predicción',
                                xaxis_title='Hora',
                                yaxis_title='Q_net (m³/hr)',
                                hovermode='x unified',
                                template='plotly_white',
                                legend=dict(x=0.02, y=0.98)
                            )
                            
                            resumen = f"""
                            ### Evaluación en Periodo de Testing
                            
                            **📊 Métricas de Performance:**
                            - **RMSE:** {rmse:.2f} m³/hr
                            - **MAE:** {mae:.2f} m³/hr
                            - **R²:** {r2:.4f}
                            - **Registros evaluados:** {len(y_real)}
                            
                            **📈 Comparación con Entrenamiento:**
                            - RMSE entrenamiento: {self.metricas_modelo['rmse']:.2f} m³/hr
                            - Diferencia: {abs(rmse - self.metricas_modelo['rmse']):.2f} m³/hr
                            
                            {'✅ **Excelente generalización**' if abs(rmse - self.metricas_modelo['rmse']) < 50 else '⚠️ **Revisar generalización**'}
                            
                            ---
                            *Modelo: Forecasting V3.0 | 47 features con temperatura*
                            """
                            
                            return resumen, fig
                            
                        except Exception as e:
                            return f"❌ Error: {str(e)}", None
                    
                    btn_testing.click(
                        fn=generar_evaluacion_testing,
                        inputs=[],
                        outputs=[output_testing, output_plot_testing]
                    )
                
                # ========== TAB 5: Dashboard Operativo ==========
                with gr.Tab("📊 Dashboard Operativo"):
                    gr.Markdown("### Monitoreo en Tiempo Real del Sistema")
                    
                    btn_dashboard = gr.Button("🔄 Actualizar Dashboard", variant="primary")
                    output_dashboard = gr.Markdown()
                    output_plot_dashboard = gr.Plot()
                    
                    def generar_dashboard():
                        try:
                            # Últimas 48 horas
                            df_reciente = self.df_volumen.tail(48)
                            
                            # Crear dashboard
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('Volumen Últimas 48h', 'Temperatura Últimas 48h',
                                              'Distribución Porcentaje', 'Eventos Críticos')
                            )
                            
                            # Volumen
                            fig.add_trace(
                                go.Scatter(x=df_reciente['timestamp_chile'], 
                                          y=df_reciente['volumen_total_m3'],
                                          mode='lines+markers',
                                          name='Volumen',
                                          line=dict(color='#0066cc')),
                                row=1, col=1
                            )
                            
                            # Temperatura
                            df_clima_reciente = self.df_clima.tail(48)
                            fig.add_trace(
                                go.Scatter(x=df_clima_reciente['timestamp_chile'],
                                          y=df_clima_reciente['temp_c'],
                                          mode='lines+markers',
                                          name='Temperatura',
                                          line=dict(color='#ff6600')),
                                row=1, col=2
                            )
                            
                            fig.update_layout(height=600, showlegend=False, template='plotly_white')
                            
                            # Resumen
                            vol_actual = df_reciente['volumen_total_m3'].iloc[-1]
                            pct_actual = (vol_actual / self.sistema_info['capacidad_total_m3']) * 100
                            temp_actual = df_clima_reciente['temp_c'].iloc[-1]
                            
                            resumen = f"""
                            ### Estado Actual del Sistema
                            
                            **💧 Volumen:** {vol_actual:,.0f} m³ ({pct_actual:.1f}%)
                            **🌡️ Temperatura:** {temp_actual:.1f}°C
                            **⏰ Última actualización:** {df_reciente['timestamp_chile'].iloc[-1]}
                            
                            **🎯 Umbrales:**
                            - Mínimo operativo: 60% ({self.sistema_info['limite_minimo_60pct']:,.0f} m³)
                            - Máximo operativo: 90% ({self.sistema_info['limite_maximo_90pct']:,.0f} m³)
                            
                            {'🟢 **Sistema Normal**' if 60 <= pct_actual <= 90 else '⚠️ **Fuera de rango operativo**'}
                            """
                            
                            return resumen, fig
                            
                        except Exception as e:
                            return f"❌ Error: {str(e)}", None
                    
                    btn_dashboard.click(
                        fn=generar_dashboard,
                        inputs=[],
                        outputs=[output_dashboard, output_plot_dashboard]
                    )
                
                # ========== TAB 6: Información del Sistema ==========
                with gr.Tab("ℹ️ Información del Sistema"):
                    gr.Markdown(f"""
                    # 🚰 Sistema Predictivo ESVAL V3.0
                    
                    ## 📊 Modelo Forecasting con Temperatura
                    
                    ### 🏆 Performance del Modelo
                    - **Tipo:** XGBoost Regressor
                    - **Features:** {len(self.features_modelo)} (incluye temperatura y clima)
                    - **RMSE:** {self.metricas_modelo['rmse']:.2f} m³/hr (9.6% del std)
                    - **MAE:** {self.metricas_modelo['mae']:.2f} m³/hr
                    - **R²:** {self.metricas_modelo['r2']:.4f} **(99.03% varianza explicada)**
                    - **MAPE:** {self.metricas_modelo['mape']:.2f}%
                    
                    ### 🌡️ Umbrales de Temperatura (Estadísticamente Validados)
                    - **Frío:** < {self.umbrales['temp_frio']:.1f}°C (Q25 - comportamiento irregular, MAE +11%)
                    - **Normal:** {self.umbrales['temp_frio']:.1f}°C - {self.umbrales['temp_calor']:.1f}°C
                    - **Calor:** > {self.umbrales['temp_calor']:.1f}°C (Q75 - patrones predecibles, MAE -26%)
                    
                    ### 📈 TOP 5 Features Más Importantes
                    1. **Q_net_m3h__ema_win_6h** → 26.4% (persistencia 6h DOMINA)
                    2. **periodo_dia_Madrugada** → 17.6% (período crítico 0-6h)
                    3. **cal_hour_cos** → 17.3% (ciclo horario)
                    4. **hora** → 11.3% (efecto directo hora del día)
                    5. **Q_net_m3h__diff_168h** → 7.8% (patrón semanal)
                    
                    ### 🔍 Hallazgos Clave
                    
                    #### 1. Persistencia > Todo
                    **EMA 6h domina con 26.4%** - La demanda en las próximas horas depende más de las últimas 6h que de cualquier otra variable.
                    
                    #### 2. Temperatura: Cambios > Valores
                    **Delta temp 6h-48h es 4x más predictivo** que temperatura absoluta. El sistema reacciona a cambios, no a valores estáticos.
                    
                    #### 3. Qin es Redundante (Validado)
                    **Modelo sin Qin es 4.6% mejor** - Confirma que Qin refleja decisiones operacionales basadas en clima/calendario que el modelo ya captura.
                    
                    #### 4. Períodos Críticos
                    - **Madrugada (0-6h):** 17.6% importancia - Estable, baja demanda
                    - **Mañana crítica (7-9h):** Alta variabilidad, MAE +32%
                    - **Día (9-18h):** Más predecible, MAE -11%
                    
                    #### 5. Eventos Especiales: Bajo Impacto
                    **Feriados <0.01% importancia** - Eventos específicos no mejoran predicción. Mejor usar patrones agregados.
                    
                    ### 📊 Performance por Segmento
                    
                    **Por Temperatura:**
                    - **Calor (>16.3°C):** MAE 200 m³/hr ✅ (MEJOR - patrones predecibles)
                    - **Normal (10.5-16.3°C):** MAE 260 m³/hr
                    - **Frío (<10.5°C):** MAE 301 m³/hr ⚠️ (PEOR - irregular)
                    
                    **Por Período:**
                    - **Día (9-18h):** MAE 240 m³/hr ✅ (MEJOR - estable)
                    - **Madrugada (0-6h):** MAE 280 m³/hr
                    - **Mañana crítica (7-9h):** MAE 356 m³/hr ⚠️ (PEOR - transición)
                    - **Noche (18-22h):** MAE 290 m³/hr
                    
                    ### 🗄️ Datos del Sistema
                    
                    **Capacidad Total:** {self.sistema_info['capacidad_total_m3']:,.0f} m³
                    **Período Análisis:** {self.sistema_info['periodo_inicio']} - {self.sistema_info['periodo_fin']}
                    **Días Analizados:** {self.sistema_info['total_dias_analizados']}
                    **Registros Entrenamiento:** 14,928 horas (después de limpieza)
                    
                    **Límites Operativos:**
                    - **Mínimo:** 60% ({self.sistema_info['limite_minimo_60pct']:,.0f} m³)
                    - **Máximo:** 90% ({self.sistema_info['limite_maximo_90pct']:,.0f} m³)
                    
                    ### 🔧 Arquitectura del Modelo
                    
                    **Split Temporal:**
                    - Train: 10,449 registros (70%) - 2024-01-01 → 2025-03-21
                    - Validación: 2,239 registros (15%) - 2025-03-21 → 2025-06-27
                    - Test: 2,240 registros (15%) - 2025-06-27 → 2025-09-30
                    
                    **Hiperparámetros XGBoost:**
                    - n_estimators: 500
                    - max_depth: 8
                    - learning_rate: 0.05
                    - early_stopping_rounds: 50
                    
                    **Features Generadas:**
                    - 31 prometedoras (correlación > 0.2)
                    - 8 categóricas (one-hot encoding)
                    - 4 temporales (hora, día, mes, finde)
                    - 3 interacciones (temp × hora, temp × finde, etc.)
                    - 1 régimen operacional
                    
                    ### 📁 Archivos del Modelo
                    
                    **Ubicación:** `models/forecasting/`
                    - `modelo_forecasting_xgboost.pkl` - Modelo entrenado
                    - `features.txt` - 47 features en orden correcto
                    - `metricas.json` - RMSE, MAE, R², MAPE
                    - `feature_importance.csv` - Importancia individual
                    - `umbrales.pkl` - Thresholds (10.5°C, 16.3°C, etc.)
                    
                    ### 🚀 Uso Recomendado
                    
                    **Para Forecasting 24-72h:**
                    ✅ Usar Modelo Forecasting V3.0 (este modelo)
                    - No requiere Qin futuro
                    - Solo necesita pronóstico meteorológico
                    - Puede predecir con 3+ días de anticipación
                    
                    **Requisitos:**
                    - Temperatura histórica y futura
                    - Calendario (hora, día, mes, feriados)
                    - Q_net histórico (para features LAG y EMA)
                    
                    ### ⚠️ Limitaciones
                    
                    1. **Mañana crítica (7-9h):** Mayor error debido a transiciones variables
                    2. **Frío extremo (<10.5°C):** Comportamiento menos predecible
                    3. **Eventos únicos:** Muestra pequeña en entrenamiento (1-2 ocurrencias)
                    4. **Datos:** Modelo entrenado con 2024-2025 (21 meses)
                    
                    ### 🔄 Reentrenamiento
                    
                    **Recomendado:**
                    - Mensual: Incorporar datos del mes anterior
                    - Urgente si error aumenta >15%
                    - Urgente si R² cae <0.95
                    - Tras eventos excepcionales
                    
                    ### 📞 Soporte Técnico
                    
                    **Documentación Completa:**
                    - `RESUMEN_MODELO_FORECASTING_V3.md` - Reporte ejecutivo
                    - `GUIA_USO_MODELO_V3.md` - Manual de uso con código
                    - `HALLAZGOS_VALORES_BISAGRA.md` - Análisis de umbrales
                    
                    **Scripts:**
                    - `entrenar_modelos_forecasting_explicativo.py` - Entrenamiento
                    - `comparar_modelo_baseline.py` - Comparación
                    - `test_modelo_forecasting.py` - Tests
                    
                    ---
                    
                    **Versión:** 3.0.0  
                    **Fecha:** Noviembre 2025  
                    **Estado:** 🚀 **PRODUCCIÓN**  
                    **R² = 0.9903** ✅
                    """)
        
        return demo


def main():
    """Función principal"""
    print("=" * 80)
    print("🚰 SISTEMA PREDICTIVO ESVAL V3.0 - INTERFAZ COMPLETA")
    print("=" * 80)
    
    # Crear instancia
    app = InterfazGradioV3Completa()
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    if not app.cargar_datos():
        print("❌ Error cargando datos. Verifique que existan los archivos en data/processed/")
        return
    
    # Cargar modelo
    print("\n🤖 Cargando modelo Forecasting V3.0...")
    if not app.cargar_modelo_forecasting():
        print("⚠️ Modelo Forecasting V3.0 no disponible")
        print("   Algunas funcionalidades de predicción estarán limitadas")
    
    # Crear interfaz
    print("\n🎨 Creando interfaz Gradio...")
    demo = app.crear_interfaz()
    
    # Lanzar
    print("\n🚀 Lanzando aplicación...")
    print("=" * 80)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
