"""
Interfaz Gradio V3.0 COMPLETA - Sistema Predictivo ESVAL
Integra modelo Forecasting V3.0 con predicciones basadas en temperatura
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports para comparación de modelos
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM no disponible")


class InterfazESVAL_V3:
    """Interfaz completa para Sistema Predictivo ESVAL V3.0"""
    
    def __init__(self):
        self.modelo = None
        self.features = []
        self.umbrales = None
        self.metricas = {}
        self.df_completo = None
        self.df_pronostico = None
        
    def cargar_todo(self):
        """Carga modelo V3.0, datos y configuraciones"""
        print("🚀 Cargando Sistema ESVAL V3.0...")
        
        try:
            # Cargar modelo Forecasting V3.0
            model_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
            if model_path.exists():
                self.modelo = joblib.load(model_path)
                print("✅ Modelo Forecasting V3.0 cargado")
            else:
                print("⚠️ Modelo V3.0 no encontrado, usando fallback")
                return False
            
            # Cargar features
            with open('models/forecasting/features.txt') as f:
                self.features = [line.strip() for line in f]
            print(f"✅ {len(self.features)} features cargadas")
            
            # Cargar métricas
            with open('models/forecasting/metricas.json') as f:
                self.metricas = json.load(f)
            print(f"✅ Métricas: R²={self.metricas['r2']:.4f}")
            
            # Cargar umbrales
            self.umbrales = joblib.load('models/forecasting/umbrales.pkl')
            print(f"✅ Umbrales: {self.umbrales['temp_frio']:.1f}°C / {self.umbrales['temp_calor']:.1f}°C")
            
            # Cargar dataset completo
            df_path = Path('data/processed/dataset_features_completo.csv')
            if df_path.exists():
                self.df_completo = pd.read_csv(df_path)
                self.df_completo['timestamp'] = pd.to_datetime(self.df_completo['timestamp'])
                print(f"✅ Dataset: {len(self.df_completo):,} registros")
            
            # Cargar pronóstico climático
            pron_path = Path('data/processed/pronostico_3dias_open_meteo.csv')
            if pron_path.exists():
                self.df_pronostico = pd.read_csv(pron_path)
                print(f"✅ Pronóstico climático cargado")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def crear_features_prediccion(self, hora, dia_semana, mes, temperatura, df_base=None):
        """Crea TODAS las features necesarias para predicción"""
        
        if df_base is not None and len(df_base) > 0:
            # Buscar días similares POR TEMPERATURA Y HORA
            df_similar = df_base[
                (df_base['timestamp'].dt.hour == hora) &
                (df_base['timestamp'].dt.month == mes)
            ].copy()
            
            # Filtrar por rango de temperatura similar (±3°C)
            if 'clima_temp_c' in df_similar.columns and len(df_similar) > 0:
                df_similar = df_similar[
                    (df_similar['clima_temp_c'] >= temperatura - 3) &
                    (df_similar['clima_temp_c'] <= temperatura + 3)
                ]
            
            # Si hay datos similares, usar promedio (solo columnas numéricas)
            if len(df_similar) > 5:
                row = df_similar.median(numeric_only=True).copy()
            elif len(df_base) > 0:
                # Fallback: última fila (solo columnas numéricas)
                numeric_cols = df_base.select_dtypes(include=[np.number]).columns
                row = df_base[numeric_cols].iloc[-1].copy()
            else:
                row = pd.Series(dtype=float)
        else:
            # Crear features desde cero con valores por defecto
            row = pd.Series(dtype=float)
        
        # Features de Q_net ajustadas por temperatura y hora
        if self.df_completo is not None:
            # Filtrar por hora Y temperatura similar
            df_hora_temp = self.df_completo[
                (self.df_completo['timestamp'].dt.hour == hora) &
                (self.df_completo['timestamp'].dt.month == mes)
            ].copy()
            
            if 'clima_temp_c' in df_hora_temp.columns and len(df_hora_temp) > 0:
                # Buscar casos con temperatura similar (±5°C)
                df_hora_temp = df_hora_temp[
                    (df_hora_temp['clima_temp_c'] >= temperatura - 5) &
                    (df_hora_temp['clima_temp_c'] <= temperatura + 5)
                ]
            
            if len(df_hora_temp) > 0:
                q_med = df_hora_temp['Q_net_m3h'].median()
                q_std = df_hora_temp['Q_net_m3h'].std()
                if pd.isna(q_std):
                    q_std = 2000.0
                row['Q_net_m3h__ema_win_6h'] = q_med
                row['Q_net_m3h__ema_win_12h'] = q_med
                row['Q_net_m3h__ema_win_24h'] = q_med
                row['Q_net_m3h__lag_168h'] = q_med
                row['Q_net_m3h__diff_168h'] = 0
            else:
                # No hay datos similares - usar promedio general con ajuste temp
                df_hora_gen = self.df_completo[
                    self.df_completo['timestamp'].dt.hour == hora
                ]
                if len(df_hora_gen) > 0:
                    q_base = df_hora_gen['Q_net_m3h'].median()
                    # Ajustar por temperatura: correlación -0.124
                    # Por cada 10°C de aumento, demanda aumenta ~12% (más negativo)
                    if 'clima_temp_c' in df_hora_gen.columns:
                        temp_promedio = df_hora_gen['clima_temp_c'].median()
                        delta_temp = temperatura - temp_promedio
                        ajuste = q_base * (-0.012 * delta_temp)  # Negativo porque más temp = más negativo
                        q_ajustado = q_base + ajuste
                    else:
                        q_ajustado = q_base
                    
                    row['Q_net_m3h__ema_win_6h'] = q_ajustado
                    row['Q_net_m3h__ema_win_12h'] = q_ajustado
                    row['Q_net_m3h__ema_win_24h'] = q_ajustado
                    row['Q_net_m3h__lag_168h'] = q_ajustado
                    row['Q_net_m3h__diff_168h'] = 0
                else:
                    row['Q_net_m3h__ema_win_6h'] = -500
                    row['Q_net_m3h__ema_win_12h'] = -500
                    row['Q_net_m3h__ema_win_24h'] = -500
                    row['Q_net_m3h__lag_168h'] = -500
                    row['Q_net_m3h__diff_168h'] = 0
            
        # Actualizar features de clima con temperatura input
        row['clima_temp_c__max_win_12h'] = temperatura + 2
        row['clima_temp_c__mean_win_12h'] = temperatura
        row['clima_temp_c__std_win_12h'] = 1.5
        row['clima_temp_c__slope_lin_win_6h'] = 0
        row['clima_temp_c__slope_lin_win_12h'] = 0
        row['clima_temp_c__slope_lin_win_24h'] = 0
        row['clima_temp_c__slope_lin_win_48h'] = 0
        row['clima_temp_c__slope_lin_win_72h'] = 0
        row['clima_temp_c__lag_6h'] = temperatura - 0.5
        row['clima_temp_delta_1h'] = 0.2
        row['clima_temp_delta_3h'] = 0.3
        row['clima_temp_delta_6h'] = 0.5
        row['clima_temp_delta_12h'] = 0.8
        
        # Features de HR (asumir 60% por defecto)
        row['clima_HR_pct__min_win_12h'] = 55
        row['clima_HR_pct__mean_win_12h'] = 60
        row['clima_HR_pct__std_win_12h'] = 5
        row['clima_HR_pct__slope_lin_win_6h'] = 0
        row['clima_HR_pct__slope_lin_win_12h'] = 0
        row['clima_HR_pct__slope_lin_win_24h'] = 0
        row['clima_HR_pct__slope_lin_win_48h'] = 0
        row['clima_HR_pct__lag_6h'] = 60
        
        # Features calendario
        row['cal_hour_sin'] = np.sin(2 * np.pi * hora / 24)
        row['cal_hour_cos'] = np.cos(2 * np.pi * hora / 24)
        
        # Actualizar temporales
        row['hora'] = hora
        row['dia_semana'] = dia_semana
        row['mes'] = mes
        row['es_fin_de_semana'] = 1 if dia_semana >= 5 else 0
        row['hora_seno'] = np.sin(2 * np.pi * hora / 24)
        row['hora_coseno'] = np.cos(2 * np.pi * hora / 24)
        
        # Features categóricas: temp_nivel
        if temperatura < self.umbrales['temp_frio']:
            row['temp_nivel_Frio'] = 1
            row['temp_nivel_Normal'] = 0
            row['temp_nivel_Calor'] = 0
        elif temperatura <= self.umbrales['temp_calor']:
            row['temp_nivel_Frio'] = 0
            row['temp_nivel_Normal'] = 1
            row['temp_nivel_Calor'] = 0
        else:
            row['temp_nivel_Frio'] = 0
            row['temp_nivel_Normal'] = 0
            row['temp_nivel_Calor'] = 1
        
        # Features categóricas: periodo_dia
        if hora < 6:
            periodo = 'Madrugada'
        elif hora < 9:
            periodo = 'Manana_critica'
        elif hora < 18:
            periodo = 'Dia'
        elif hora < 22:
            periodo = 'Noche'
        else:
            periodo = 'Noche_tardia'
        
        for p in ['Madrugada', 'Manana_critica', 'Dia', 'Noche', 'Noche_tardia']:
            row[f'periodo_dia_{p}'] = 1 if periodo == p else 0
        
        # es_hora_bisagra
        row['es_hora_bisagra'] = 1 if hora in [7, 8, 9, 22, 23] else 0
        
        # regimen_equilibrio
        row['regimen_equilibrio'] = 0
        
        # Interacciones
        row['temp_x_hora'] = temperatura * hora
        row['temp_x_finde'] = temperatura * row['es_fin_de_semana']
        
        # delta_temp_6h_x_hora
        if 'clima_temp_delta_6h' in row:
            row['delta_temp_6h_x_hora'] = row['clima_temp_delta_6h'] * hora
        else:
            row['delta_temp_6h_x_hora'] = 0
        
        return row
    
    def predecir_24_horas(self, fecha_str, temperatura):
        """Predicción para 24 horas con misma temperatura"""
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
            
            predicciones = []
            horas = []
            
            for hora in range(24):
                row = self.crear_features_prediccion(
                    hora,
                    fecha.weekday(),
                    fecha.month,
                    temperatura,
            
            # Agregar contexto de temperatura
            if pred < 0:
                tipo_flujo = "DEMANDA (consumo)"
                icono_flujo = "📉"
            else:
                tipo_flujo = "RECUPERACIÓN (llenado)"
                icono_flujo = "📈"
            
            resultado = f"""
### Predicción {fecha_str} - {int(hora):02d}:00

**🌡️ Temperatura:** {temperatura:.1f}°C

**{icono_flujo} Q_net Predicho:** {pred:,.0f} m³/hr  
*({tipo_flujo})*

**� Nivel:** {nivel}

**⚠️ Estado:** {alerta}

---

💡 **Nota:** Valores negativos = demanda/consumo | Valores positivos = recuperación/llenado  
🌡️ Mayor temperatura → Mayor demanda (más negativo)

*Modelo Forecasting V3.0 | R²={self.metricas['r2']:.4f}*
            """
            
            # Crear gráfico simple
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Predicción'],
                y=[pred],
                marker_color='red' if pred < 0 else 'blue',
                text=[f"{pred:,.0f} m³/hr"],
                textposition='outside'
            ))
            fig.update_layout(
                title=f"Predicción {fecha_str} - {int(hora):02d}:00",
                yaxis_title="Q_net (m³/hr)",
                height=300,
                showlegend=False
            )
            
            return resultado, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def predecir_24_horas(self, fecha_str, temperatura):
        """Predicción para 24 horas con misma temperatura"""
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
            
            predicciones = []
            horas = []
            
            for hora in range(24):
                row = self.crear_features_prediccion(
                    hora,
                    fecha.weekday(),
                    fecha.month,
                    temperatura,
                    self.df_completo
                )
                
                X = pd.DataFrame([row])[self.features]
                pred = self.modelo.predict(X)[0]
                
                predicciones.append(pred)
                horas.append(hora)
            
            # Calcular resumen
            demanda_total = sum([p for p in predicciones if p < 0])
            recup_total = sum([p for p in predicciones if p >= 0])
            balance = demanda_total + recup_total
            
            # Contexto de temperatura
            if temperatura < self.umbrales['temp_frio']:
                ctx_temp = f"❄️ Día frío (<{self.umbrales['temp_frio']:.1f}°C) - Demanda baja esperada"
            elif temperatura > self.umbrales['temp_calor']:
                ctx_temp = f"🌡️ Día caluroso (>{self.umbrales['temp_calor']:.1f}°C) - Demanda alta esperada"
            else:
                ctx_temp = "🌤️ Temperatura normal - Demanda moderada"
            
            resultado = f"""
### Resumen 24 Horas - {fecha_str}

**🌡️ Temperatura:** {temperatura:.1f}°C constante  
{ctx_temp}

**� Demanda Total:** {abs(demanda_total):,.0f} m³ (consumo)

**📈 Recuperación Total:** {abs(recup_total):,.0f} m³ (llenado)

**⚖️ Balance Neto:** {balance:+,.0f} m³

**⏰ Hora Máxima Demanda:** {horas[np.argmin(predicciones)]:02d}:00 ({predicciones[np.argmin(predicciones)]:,.0f} m³/hr)

**⏰ Hora Máxima Recuperación:** {horas[np.argmax(predicciones)]:02d}:00 ({predicciones[np.argmax(predicciones)]:,.0f} m³/hr)

---
💡 **Nota:** A mayor temperatura → Mayor demanda (barras rojas más grandes)
            """
            
            # Gráfico
            fig = go.Figure()
            
            colores = ['red' if p < 0 else 'blue' for p in predicciones]
            
            fig.add_trace(go.Bar(
                x=[f"{h:02d}:00" for h in horas],
                y=predicciones,
                marker_color=colores,
                name='Predicción',
                hovertemplate='%{x}<br>Q_net: %{y:,.0f} m³/hr<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Predicción 24 Horas - {fecha_str} ({temperatura:.1f}°C)",
                xaxis_title="Hora",
                yaxis_title="Q_net (m³/hr)",
                height=400,
                hovermode='x unified'
            )
            
            return resultado, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def predecir_72_horas(self, fecha_str, temp_dia1, temp_dia2, temp_dia3):
        """Predicción para 72 horas con diferentes temperaturas por día"""
        try:
            fecha_inicio = datetime.strptime(fecha_str, "%d/%m/%Y")
            temperaturas_dias = [temp_dia1, temp_dia2, temp_dia3]
            
            predicciones = []
            timestamps = []
            temperaturas_plot = []
            
            for dia in range(3):
                fecha_dia = fecha_inicio + timedelta(days=dia)
                temp_dia = temperaturas_dias[dia]
                
                for hora in range(24):
                    row = self.crear_features_prediccion(
                        hora,
                        fecha_dia.weekday(),
                        fecha_dia.month,
                        temp_dia,
                        self.df_completo
                    )
                    
                    X = pd.DataFrame([row])[self.features]
                    pred = self.modelo.predict(X)[0]
                    
                    predicciones.append(pred)
                    timestamps.append(fecha_dia + timedelta(hours=hora))
                    temperaturas_plot.append(temp_dia)
            
            # Resumen por día
            resumen_dias = []
            for dia in range(3):
                inicio_dia = dia * 24
                fin_dia = inicio_dia + 24
                preds_dia = predicciones[inicio_dia:fin_dia]
                
                demanda = sum([p for p in preds_dia if p < 0])
                recup = sum([p for p in preds_dia if p >= 0])
                balance = demanda + recup
                
                fecha_dia = fecha_inicio + timedelta(days=dia)
                temp_dia = temperaturas_dias[dia]
                
                resumen_dias.append(f"""
**Día {dia+1}** ({fecha_dia.strftime('%d/%m')} - {temp_dia:.1f}°C):
- Demanda: {abs(demanda):,.0f} m³
- Recuperación: {abs(recup):,.0f} m³
- Balance: {balance:+,.0f} m³
                """)
            
            balance_total = sum(predicciones)
            
            resultado = f"""
### Resumen 72 Horas - {fecha_str}

{''.join(resumen_dias)}

**⚖️ Balance Total 3 Días:** {balance_total:+,.0f} m³
            """
            
            # Gráfico con dos ejes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Q_net (barras)
            colores = ['red' if p < 0 else 'blue' for p in predicciones]
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=predicciones,
                    marker_color=colores,
                    name='Q_net',
                    hovertemplate='%{x|%d/%m %H:%M}<br>Q_net: %{y:,.0f} m³/hr<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Temperatura (línea)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=temperaturas_plot,
                    mode='lines',
                    name='Temperatura',
                    line=dict(color='orange', width=2),
                    yaxis='y2',
                    hovertemplate='%{x|%d/%m %H:%M}<br>Temp: %{y:.1f}°C<extra></extra>'
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Fecha/Hora")
            fig.update_yaxes(title_text="Q_net (m³/hr)", secondary_y=False)
            fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=True)
            
            fig.update_layout(
                title=f"Predicción 72 Horas - {fecha_str}",
                height=500,
                hovermode='x unified',
                legend=dict(x=0, y=1)
            )
            
            return resultado, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def evaluar_testing(self):
        """Evaluación en periodo de test"""
        try:
            # Usar dataset completo y hacer split temporal (últimos 15%)
            if self.df_completo is None or len(self.df_completo) == 0:
                return "❌ Dataset no disponible", None
            
            # Ordenar por timestamp
            df = self.df_completo.sort_values('timestamp').copy()
            
            # Split: últimos 15% como test (igual que en entrenamiento)
            n = len(df)
            test_start = int(n * 0.85)
            df_test = df.iloc[test_start:].copy()
            
            # Verificar que tengamos Q_net
            if 'Q_net_m3h' not in df_test.columns:
                return "❌ Variable objetivo Q_net_m3h no encontrada", None
            
            # Verificar features
            features_disponibles = [f for f in self.features if f in df_test.columns]
            features_faltantes = [f for f in self.features if f not in df_test.columns]
            
            if len(features_faltantes) > 0:
                print(f"⚠️ Features faltantes: {len(features_faltantes)}")
                # Crear con valor 0
                for feat in features_faltantes:
                    df_test[feat] = 0
            
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            
            # Predecir
            y_pred = self.modelo.predict(X_test)
            
            # Métricas
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mask_no_cero = y_test != 0
            mape = np.mean(np.abs((y_test[mask_no_cero] - y_pred[mask_no_cero]) / 
                                  y_test[mask_no_cero])) * 100
            
            resultado = f"""
### Evaluación en Testing

**📊 Métricas:**
- **RMSE:** {rmse:,.2f} m³/hr
- **MAE:** {mae:,.2f} m³/hr
- **R²:** {r2:.4f}
- **MAPE:** {mape:.2f}%

**📅 Período:** {df_test['timestamp'].min().strftime('%d/%m/%Y')} - {df_test['timestamp'].max().strftime('%d/%m/%Y')}

**📈 Registros:** {len(df_test):,}
            """
            
            # Gráfico: Real vs Predicho con subplot de errores
            from plotly.subplots import make_subplots
            
            # Tomar muestra para visualización (últimos 500 puntos)
            n_plot = min(500, len(df_test))
            idx_plot = range(len(df_test) - n_plot, len(df_test))
            
            # Calcular errores
            errores = y_pred[idx_plot] - y_test.iloc[idx_plot].values
            
            # Crear subplot: predicciones arriba, errores abajo
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(
                    f'Real vs Predicho (últimos {n_plot} registros)',
                    'Error de Predicción'
                ),
                vertical_spacing=0.12
            )
            
            # Subplot 1: Real vs Predicho
            fig.add_trace(go.Scatter(
                x=df_test.iloc[idx_plot]['timestamp'],
                y=y_test.iloc[idx_plot],
                mode='lines',
                name='Real',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='%{x|%d/%m %H:%M}<br>Real: %{y:,.0f} m³/hr<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_test.iloc[idx_plot]['timestamp'],
                y=y_pred[idx_plot],
                mode='lines',
                name='Predicho',
                line=dict(color='#A23B72', width=2, dash='dot'),
                hovertemplate='%{x|%d/%m %H:%M}<br>Pred: %{y:,.0f} m³/hr<extra></extra>'
            ), row=1, col=1)
            
            # Subplot 2: Errores
            fig.add_trace(go.Scatter(
                x=df_test.iloc[idx_plot]['timestamp'],
                y=errores,
                mode='lines',
                name='Error',
                line=dict(color='#F18F01', width=1),
                fill='tozeroy',
                fillcolor='rgba(241, 143, 1, 0.2)',
                hovertemplate='%{x|%d/%m %H:%M}<br>Error: %{y:,.0f} m³/hr<extra></extra>'
            ), row=2, col=1)
            
            # Línea de error cero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=2, col=1)
            
            fig.update_xaxes(title_text="Fecha/Hora", row=2, col=1)
            fig.update_yaxes(title_text="Q_net (m³/hr)", row=1, col=1)
            fig.update_yaxes(title_text="Error (m³/hr)", row=2, col=1)
            
            fig.update_layout(
                height=700,
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=0, y=1.05, orientation='h')
            )
            
            return resultado, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def crear_header_pronostico(self):
        """Crea header con pronóstico climático min/max"""
        
        # Pronóstico por defecto
        dias_pron = [
            {"dia": "Día 1", "temp_min": 15, "temp_max": 25, "icono": "☀️"},
            {"dia": "Día 2", "temp_min": 14, "temp_max": 24, "icono": "⛅"},
            {"dia": "Día 3", "temp_min": 13, "temp_max": 23, "icono": "🌤️"}
        ]
        
        # Intentar cargar pronóstico real
        if self.df_pronostico is not None and len(self.df_pronostico) > 0:
            for i in range(min(3, len(self.df_pronostico))):
                row = self.df_pronostico.iloc[i]
                if 'temperature_min' in row and 'temperature_max' in row:
                    dias_pron[i]['temp_min'] = row['temperature_min']
                    dias_pron[i]['temp_max'] = row['temperature_max']
        
        # Generar HTML
        cards_html = ""
        for dia in dias_pron:
            cards_html += f"""
            <div style="background: rgba(255,255,255,0.95); padding: 12px; border-radius: 10px; 
                        text-align: center; min-width: 100px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 5px;">{dia['icono']}</div>
                <div style="font-size: 16px; font-weight: bold; color: #2c5aa0;">
                    {dia['temp_min']:.0f}°C - {dia['temp_max']:.0f}°C
                </div>
                <div style="font-size: 13px; color: #666; margin-top: 3px;">{dia['dia']}</div>
            </div>
            """
        
        header = f"""
        <div style="background: linear-gradient(135deg, #2c5aa0 0%, #1e3a5f 100%); 
                    padding: 25px; border-radius: 15px; margin-bottom: 20px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div style="color: white;">
                    <h1 style="margin: 0; font-size: 32px; font-weight: bold;">
                        🚰 Sistema Predictivo ESVAL V3.0
                    </h1>
                    <p style="margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;">
                        Predicción de Demanda Agua Potable - Gran Valparaíso
                    </p>
                </div>
                <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                    <div style="color: white; text-align: right; margin-right: 10px;">
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">
                            🌤️ Pronóstico 72h:
                        </div>
                    </div>
                    {cards_html}
                </div>
            </div>
        </div>
        """
        
        return header
    
    def crear_tab_info(self):
        """Crea contenido del tab INFO con todos los detalles"""
        
        info_html = f"""
## ℹ️ Información del Sistema

### 🎯 Modelo Forecasting V3.0

**Performance:**
- **R²:** {self.metricas.get('r2', 0):.4f} (99.03% varianza explicada)
- **RMSE:** {self.metricas.get('rmse', 0):.2f} m³/hr
- **MAE:** {self.metricas.get('mae', 0):.2f} m³/hr
- **MAPE:** {self.metricas.get('mape', 0):.2f}%

**Arquitectura:**
- **Algoritmo:** XGBoost
- **Features:** {len(self.features)} variables
- **Sin Qin:** Predicción pura basada en clima/calendario
- **Training:** 70% | Val: 15% | Test: 15%

---

### 🌡️ Umbrales Estadísticos

**Temperatura:**
- **Frío:** < {self.umbrales.get('temp_frio', 10.5):.1f}°C (comportamiento irregular)
- **Normal:** {self.umbrales.get('temp_frio', 10.5):.1f}°C - {self.umbrales.get('temp_calor', 16.3):.1f}°C
- **Calor:** > {self.umbrales.get('temp_calor', 16.3):.1f}°C (patrones predecibles)

**Producción (Qin):**
- **Bajo:** < {self.umbrales.get('qin_bajo', 10998):.0f} m³/hr
- **Normal:** {self.umbrales.get('qin_bajo', 10998):.0f} - {self.umbrales.get('qin_alto', 12614):.0f} m³/hr
- **Alto:** > {self.umbrales.get('qin_alto', 12614):.0f} m³/hr

---

### 📊 Features Principales (Top 10)

1. **Q_net_m3h__ema_win_6h** - 26.4% (Persistencia 6h DOMINA)
2. **periodo_dia_Madrugada** - 17.6% (Período crítico)
3. **cal_hour_cos** - 17.3% (Ciclo horario)
4. **hora** - 11.3% (Efecto directo hora)
5. **Q_net_m3h__diff_168h** - 7.8% (Patrón semanal)
6. **clima_temp_c__slope_lin_win_48h** - Tendencia temperatura 48h
7. **periodo_dia_Dia** - Segmento 9-18h
8. **temp_nivel (categórico)** - Umbral frío/normal/calor
9. **Interacciones** - temp×hora, temp×finde
10. **es_hora_bisagra** - Horas críticas 7-9h, 22-23h

---

### 💡 Hallazgos Científicos

**1. Persistencia > Todo**
- EMA 6h domina con 26.4% importancia
- La demanda depende más de las últimas 6h que de cualquier otra variable

**2. Temperatura: Cambios > Valores**
- Delta temp 6h-48h es 4x más predictivo que temperatura absoluta
- El sistema reacciona a cambios, no a valores estáticos

**3. Qin es Redundante**
- Modelo sin Qin es 4.6% MEJOR
- Qin refleja decisiones operacionales basadas en clima/calendario

**4. Períodos Críticos**
- Madrugada (0-6h): 17.6% importancia
- Mañana crítica (7-9h): Alta variabilidad, MAE +32%
- Día (9-18h): Más predecible, MAE -11%

**5. Eventos Especiales: Bajo Impacto**
- Feriados individuales <0.01% importancia
- Mejor usar patrones agregados (fin de semana, temporada)

---

### 📈 Performance por Segmento

**Por Temperatura:**
- **Calor (>16.3°C):** MAE 200 m³/hr ✅ MEJOR
- **Normal:** MAE 260 m³/hr
- **Frío (<10.5°C):** MAE 301 m³/hr ⚠️ Irregular

**Por Período:**
- **Día (9-18h):** MAE 240 m³/hr ✅ MEJOR
- **Madrugada (0-6h):** MAE 280 m³/hr
- **Mañana crítica (7-9h):** MAE 356 m³/hr ⚠️ Variable
- **Noche (18-22h):** MAE 290 m³/hr

**Por Día de Semana:**
- **Días laborales:** MAE 268 m³/hr
- **Fin de semana:** MAE 273 m³/hr (+1.9%)

---

### 🔧 Uso Recomendado

**Para Forecasting 24-72h:**
- ✅ Usar Modelo A (Forecasting V3.0)
- ✅ Requiere pronóstico de temperatura
- ✅ No requiere conocer Qin futuro
- ✅ Perfecto para planificación operacional

**Para Análisis Histórico:**
- Use Modelo B (Explicativo) si necesita entender relación Qin-demanda
- Disponible en `models/explicativo/`

---

### 📁 Archivos del Sistema

**Modelo Producción:**
```
models/forecasting/
├── modelo_forecasting_xgboost.pkl
├── features.txt (47 features)
├── metricas.json
├── feature_importance.csv
└── umbrales.pkl
```

**Datos:**
```
data/processed/
├── dataset_features_completo.csv (15,034 registros)
├── data_train.csv (70%)
├── data_validation.csv (15%)
└── data_test.csv (15%)
```

---

### 📞 Soporte Técnico

**Modelo:** Forecasting V3.0  
**Estado:** 🚀 PRODUCCIÓN  
**Última Actualización:** Noviembre 2025  
**Documentación:** `RESUMEN_MODELO_FORECASTING_V3.md`

**Contacto:** Sistema Predictivo ESVAL - Gran Valparaíso
        """
        
        return info_html
    
    def ejecutar_validacion_rapida(self):
        """Ejecuta suite de tests de validación automática"""
        resultados = []
        
        # Test 1: Sensibilidad a Temperatura
        resultado_temp = self.test_sensibilidad_temperatura()
        resultados.append(resultado_temp)
        
        # Test 2: Coherencia Temporal
        resultado_temporal = self.test_coherencia_temporal()
        resultados.append(resultado_temporal)
        
        # Test 3: Rangos Históricos
        resultado_rangos = self.test_rangos_historicos()
        resultados.append(resultado_rangos)
        
        # Test 4: Estado del Sistema
        resultado_sistema = self.test_estado_sistema()
        resultados.append(resultado_sistema)
        
        # Generar reporte
        return self.generar_reporte_validacion(resultados)
    
    def _calcular_prediccion_24h(self, fecha_str, temperatura):
        """Versión interna que retorna datos en vez de Markdown"""
        fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
        
        predicciones = []
        for hora in range(24):
            row = self.crear_features_prediccion(
                hora, fecha.weekday(), fecha.month, temperatura, self.df_completo
            )
            X = pd.DataFrame([row])[self.features]
            pred = self.modelo.predict(X)[0]
            predicciones.append(pred)
        
        demanda_total = sum([p for p in predicciones if p < 0])
        recup_total = sum([p for p in predicciones if p >= 0])
        balance = demanda_total + recup_total
        
        return {
            'predicciones': predicciones,
            'demanda_total': demanda_total,
            'recup_total': recup_total,
            'balance_neto': balance,
            'max_demanda': min(predicciones),
            'max_recuperacion': max(predicciones)
        }
    
    def test_sensibilidad_temperatura(self):
        """Test: Temperatura alta → Mayor demanda"""
        try:
            fecha = "13/11/2025"
            
            # Predicción con temperatura baja (3°C)
            datos_frio = self._calcular_prediccion_24h(fecha, 3.0)
            
            # Predicción con temperatura alta (30°C)
            datos_calor = self._calcular_prediccion_24h(fecha, 30.0)
            
            # Extraer balances (valores ya están en formato negativo = consumo)
            balance_frio = datos_frio['balance_neto']
            balance_calor = datos_calor['balance_neto']
            
            # Validar: 30°C debe tener MÁS demanda (más negativo)
            diferencia_pct = abs((balance_calor - balance_frio) / balance_frio * 100)
            es_correcto = balance_calor < balance_frio  # Más negativo = más demanda
            cambio_esperado = diferencia_pct > 10  # Al menos 10% de cambio
            
            return {
                'nombre': 'Sensibilidad a Temperatura',
                'icono': '🌡️',
                'estado': 'PASS' if (es_correcto and cambio_esperado) else 'FAIL',
                'detalles': f"""
**Test:** Predicción 24h con 3°C vs 30°C

- **Balance con 3°C:** {balance_frio:,.0f} m³
- **Balance con 30°C:** {balance_calor:,.0f} m³
- **Diferencia:** {diferencia_pct:.1f}%
- **Comportamiento:** {'✅ Correcto (30°C más negativo)' if es_correcto else '❌ Invertido'}
- **Sensibilidad:** {'✅ Adecuada (>10%)' if cambio_esperado else '⚠️ Baja (<10%)'}

**Esperado:** Mayor temperatura → Mayor demanda (balance más negativo)
                """,
                'ok': es_correcto and cambio_esperado
            }
        except Exception as e:
            return {
                'nombre': 'Sensibilidad a Temperatura',
                'icono': '🌡️',
                'estado': 'ERROR',
                'detalles': f'❌ Error en test: {str(e)}',
                'ok': False
            }
    
    def test_coherencia_temporal(self):
        """Test: Predicciones consecutivas son coherentes"""
        try:
            # Predecir 2 días consecutivos con misma temperatura
            fecha1 = "13/11/2025"
            fecha2 = "14/11/2025"
            temp = 15.0
            
            datos1 = self._calcular_prediccion_24h(fecha1, temp)
            datos2 = self._calcular_prediccion_24h(fecha2, temp)
            
            balance1 = datos1['balance_neto']
            balance2 = datos2['balance_neto']
            
            # Diferencia entre días no debe ser > 30%
            diferencia = abs(balance2 - balance1)
            diferencia_pct = abs((balance2 - balance1) / balance1 * 100) if balance1 != 0 else 0
            es_coherente = diferencia_pct < 30
            
            return {
                'nombre': 'Coherencia Temporal',
                'icono': '📅',
                'estado': 'PASS' if es_coherente else 'WARNING',
                'detalles': f"""
**Test:** Predicciones días consecutivos (misma temperatura)

- **Día 1 balance:** {balance1:,.0f} m³
- **Día 2 balance:** {balance2:,.0f} m³
- **Diferencia absoluta:** {diferencia:,.0f} m³
- **Diferencia relativa:** {diferencia_pct:.1f}%
- **Estado:** {'✅ Coherente (<30%)' if es_coherente else '⚠️ Alta variación (>30%)'}

**Esperado:** Días similares → Predicciones similares
                """,
                'ok': es_coherente
            }
        except Exception as e:
            return {
                'nombre': 'Coherencia Temporal',
                'icono': '📅',
                'estado': 'ERROR',
                'detalles': f'❌ Error en test: {str(e)}',
                'ok': False
            }
    
    def test_rangos_historicos(self):
        """Test: Predicciones dentro de rangos históricos razonables"""
        try:
            fecha = "13/11/2025"
            temp = 18.0
            
            datos = self._calcular_prediccion_24h(fecha, temp)
            
            # Calcular percentiles históricos
            if self.df_completo is not None:
                p1 = self.df_completo['Q_net_m3h'].quantile(0.01)
                p99 = self.df_completo['Q_net_m3h'].quantile(0.99)
                p25 = self.df_completo['Q_net_m3h'].quantile(0.25)
                p75 = self.df_completo['Q_net_m3h'].quantile(0.75)
                
                # Revisar valores máximos/mínimos de predicción
                max_demanda = datos['max_demanda']
                max_recuperacion = datos['max_recuperacion']
                
                dentro_rango = (max_demanda >= p1) and (max_recuperacion <= p99)
                
                return {
                    'nombre': 'Rangos Históricos',
                    'icono': '📊',
                    'estado': 'PASS' if dentro_rango else 'WARNING',
                    'detalles': f"""
**Test:** Predicciones dentro de rangos históricos

- **Demanda máxima predicha:** {max_demanda:,.0f} m³/hr
- **Recuperación máxima predicha:** {max_recuperacion:,.0f} m³/hr

**Rangos históricos:**
- P1: {p1:,.0f} m³/hr
- P25: {p25:,.0f} m³/hr
- P75: {p75:,.0f} m³/hr
- P99: {p99:,.0f} m³/hr

**Estado:** {'✅ Dentro de rango' if dentro_rango else '⚠️ Fuera de rango histórico'}
                    """,
                    'ok': dentro_rango
                }
            else:
                return {
                    'nombre': 'Rangos Históricos',
                    'icono': '📊',
                    'estado': 'WARNING',
                    'detalles': '⚠️ Dataset completo no disponible',
                    'ok': False
                }
        except Exception as e:
            return {
                'nombre': 'Rangos Históricos',
                'icono': '📊',
                'estado': 'ERROR',
                'detalles': f'❌ Error en test: {str(e)}',
                'ok': False
            }
    
    def test_estado_sistema(self):
        """Test: Verificar que todos los componentes están cargados"""
        try:
            checks = []
            
            # Check modelo
            checks.append(('Modelo XGBoost', self.modelo is not None))
            
            # Check features
            checks.append(('Features (47)', len(self.features) == 47 if self.features else False))
            
            # Check umbrales
            tiene_umbrales = all(k in self.umbrales for k in ['temp_frio', 'temp_calor', 'qin_bajo', 'qin_alto'])
            checks.append(('Umbrales temperatura/Qin', tiene_umbrales))
            
            # Check dataset
            checks.append(('Dataset completo', self.df_completo is not None and len(self.df_completo) > 1000))
            
            # Check métricas
            tiene_metricas = self.metricas is not None and 'r2' in self.metricas
            checks.append(('Métricas modelo', tiene_metricas))
            
            todos_ok = all(check[1] for check in checks)
            
            detalles_checks = '\n'.join([f"{'✅' if ok else '❌'} {nombre}" for nombre, ok in checks])
            
            return {
                'nombre': 'Estado del Sistema',
                'icono': '⚙️',
                'estado': 'PASS' if todos_ok else 'FAIL',
                'detalles': f"""
**Test:** Verificación de componentes del sistema

{detalles_checks}

**Estado general:** {'✅ Todos los componentes OK' if todos_ok else '❌ Faltan componentes'}
                """,
                'ok': todos_ok
            }
        except Exception as e:
            return {
                'nombre': 'Estado del Sistema',
                'icono': '⚙️',
                'estado': 'ERROR',
                'detalles': f'❌ Error en test: {str(e)}',
                'ok': False
            }
    
    def generar_reporte_validacion(self, resultados):
        """Genera reporte Markdown de validación"""
        total = len(resultados)
        passed = sum(1 for r in resultados if r['ok'])
        
        # Determinar estado general
        if passed == total:
            estado_general = "✅ **TODOS LOS TESTS PASARON**"
            color_badge = "🟢"
        elif passed >= total * 0.75:
            estado_general = "⚠️ **ALGUNOS WARNINGS**"
            color_badge = "🟡"
        else:
            estado_general = "❌ **TESTS FALLARON**"
            color_badge = "🔴"
        
        # Header
        reporte = f"""
# 🔍 Validación Rápida del Sistema

{color_badge} **Resultado:** {passed}/{total} tests pasaron

{estado_general}

---

"""
        
        # Agregar cada test
        for r in resultados:
            icono_estado = {
                'PASS': '✅',
                'WARNING': '⚠️',
                'FAIL': '❌',
                'ERROR': '💥'
            }.get(r['estado'], '❓')
            
            reporte += f"""
## {r['icono']} {r['nombre']} {icono_estado}

<details>
<summary><b>Ver detalles</b></summary>

{r['detalles']}

</details>

---

"""
        
        # Footer
        reporte += f"""
**Última ejecución:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

💡 **Nota:** Esta validación verifica el comportamiento correcto del sistema de predicción.
"""
        
        return reporte
    
    def entrenar_randomforest(self, n_estimators=200, max_depth=15):
        """Entrena modelo RandomForest para comparación"""
        try:
            print("\n🌲 Entrenando RandomForest...")
            
            # Usar datos del sistema
            if self.df_completo is None:
                return None, "❌ Dataset no cargado"
            
            # Split temporal
            df = self.df_completo.sort_values('timestamp').copy()
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]
            
            # Preparar X, y
            X_train = df_train[self.features]
            y_train = df_train['Q_net_m3h']
            X_val = df_val[self.features]
            y_val = df_val['Q_net_m3h']
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            
            # Entrenar
            modelo = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train, y_train)
            
            # Evaluar
            y_val_pred = modelo.predict(X_val)
            y_test_pred = modelo.predict(X_test)
            
            metrics_val = self._calculate_metrics(y_val.values, y_val_pred)
            metrics_test = self._calculate_metrics(y_test.values, y_test_pred)
            
            print(f"✅ RandomForest entrenado")
            print(f"   Val - R²: {metrics_val['r2']:.4f}, RMSE: {metrics_val['rmse']:.0f}")
            print(f"   Test - R²: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.0f}")
            
            return modelo, (metrics_val, metrics_test)
            
        except Exception as e:
            import traceback
            return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    def entrenar_lightgbm(self, n_estimators=200, max_depth=10):
        """Entrena modelo LightGBM para comparación"""
        if not LIGHTGBM_AVAILABLE:
            return None, "❌ LightGBM no está instalado"
        
        try:
            print("\n💡 Entrenando LightGBM...")
            
            # Usar datos del sistema
            if self.df_completo is None:
                return None, "❌ Dataset no cargado"
            
            # Split temporal
            df = self.df_completo.sort_values('timestamp').copy()
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]
            
            # Preparar X, y
            X_train = df_train[self.features]
            y_train = df_train['Q_net_m3h']
            X_val = df_val[self.features]
            y_val = df_val['Q_net_m3h']
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            
            # Entrenar
            modelo = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            modelo.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            # Evaluar
            y_val_pred = modelo.predict(X_val)
            y_test_pred = modelo.predict(X_test)
            
            metrics_val = self._calculate_metrics(y_val.values, y_val_pred)
            metrics_test = self._calculate_metrics(y_test.values, y_test_pred)
            
            print(f"✅ LightGBM entrenado")
            print(f"   Val - R²: {metrics_val['r2']:.4f}, RMSE: {metrics_val['rmse']:.0f}")
            print(f"   Test - R²: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.0f}")
            
            return modelo, (metrics_val, metrics_test)
            
        except Exception as e:
            import traceback
            return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE evitando división por cero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def comparar_modelos_ml(self):
        """Compara XGBoost actual vs RandomForest vs LightGBM"""
        try:
            print("\n🔬 Iniciando comparación de modelos...")
            
            resultados = {}
            
            # 1. Modelo actual (XGBoost Forecasting V3.0)
            print("\n📊 Evaluando XGBoost V3.0 (actual)...")
            df = self.df_completo.sort_values('timestamp').copy()
            n = len(df)
            val_end = int(n * 0.85)
            df_test = df.iloc[val_end:]
            
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            y_pred_xgb = self.modelo.predict(X_test)
            
            metrics_xgb = self._calculate_metrics(y_test.values, y_pred_xgb)
            resultados['XGBoost V3.0\n(Actual)'] = metrics_xgb
            
            # 2. RandomForest
            print("\n🌲 Entrenando RandomForest...")
            modelo_rf, resultado_rf = self.entrenar_randomforest()
            if modelo_rf is not None:
                metrics_rf = resultado_rf[1]  # Test metrics
                resultados['RandomForest'] = metrics_rf
            
            # 3. LightGBM
            if LIGHTGBM_AVAILABLE:
                print("\n💡 Entrenando LightGBM...")
                modelo_lgb, resultado_lgb = self.entrenar_lightgbm()
                if modelo_lgb is not None:
                    metrics_lgb = resultado_lgb[1]  # Test metrics
                    resultados['LightGBM'] = metrics_lgb
            
            # Generar reporte
            reporte = self._generar_reporte_comparacion(resultados)
            
            # Generar gráfico
            fig = self._generar_grafico_comparacion(resultados)
            
            return reporte, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def _generar_reporte_comparacion(self, resultados):
        """Genera reporte Markdown de comparación"""
        reporte = f"""
# 🔬 Comparación de Modelos de Machine Learning

**Fecha:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  
**Dataset:** 15,034 registros (Test set: 15% final)

---

## 📊 Resultados en Test Set

"""
        
        # Tabla comparativa
        reporte += "\n| Modelo | R² | RMSE (m³/hr) | MAE (m³/hr) | MAPE (%) |\n"
        reporte += "|--------|----:|-------------:|------------:|---------:|\n"
        
        for nombre, metrics in resultados.items():
            reporte += f"| **{nombre}** | {metrics['r2']:.4f} | {metrics['rmse']:,.0f} | {metrics['mae']:,.0f} | {metrics['mape']:.2f} |\n"
        
        # Determinar mejor modelo por métrica
        reporte += "\n---\n\n## 🏆 Mejor Modelo por Métrica\n\n"
        
        # Mejor R²
        mejor_r2 = max(resultados.items(), key=lambda x: x[1]['r2'])
        reporte += f"- **R² más alto:** {mejor_r2[0]} ({mejor_r2[1]['r2']:.4f})\n"
        
        # Menor RMSE
        mejor_rmse = min(resultados.items(), key=lambda x: x[1]['rmse'])
        reporte += f"- **RMSE más bajo:** {mejor_rmse[0]} ({mejor_rmse[1]['rmse']:,.0f} m³/hr)\n"
        
        # Menor MAE
        mejor_mae = min(resultados.items(), key=lambda x: x[1]['mae'])
        reporte += f"- **MAE más bajo:** {mejor_mae[0]} ({mejor_mae[1]['mae']:,.0f} m³/hr)\n"
        
        # Menor MAPE
        mejor_mape = min(resultados.items(), key=lambda x: x[1]['mape'])
        reporte += f"- **MAPE más bajo:** {mejor_mape[0]} ({mejor_mape[1]['mape']:.2f}%)\n"
        
        reporte += "\n---\n\n## 💡 Interpretación\n\n"
        reporte += "- **R²:** Proporción de varianza explicada (0-1, mayor es mejor)\n"
        reporte += "- **RMSE:** Error cuadrático medio, penaliza errores grandes\n"
        reporte += "- **MAE:** Error absoluto medio, más interpretable\n"
        reporte += "- **MAPE:** Error porcentual, útil para comparar escalas\n"
        
        return reporte
    
    def _generar_grafico_comparacion(self, resultados):
        """Genera gráfico de barras comparativo"""
        from plotly.subplots import make_subplots
        
        modelos = list(resultados.keys())
        
        # Crear subplots: 2x2
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² (mayor es mejor)', 'RMSE (menor es mejor)', 
                          'MAE (menor es mejor)', 'MAPE (menor es mejor)'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Colores
        colores = ['#2E86AB', '#A23B72', '#F18F01'][:len(modelos)]
        
        # R²
        r2_values = [resultados[m]['r2'] for m in modelos]
        fig.add_trace(
            go.Bar(x=modelos, y=r2_values, marker_color=colores, 
                   text=[f"{v:.4f}" for v in r2_values], textposition='outside',
                   showlegend=False),
            row=1, col=1
        )
        
        # RMSE
        rmse_values = [resultados[m]['rmse'] for m in modelos]
        fig.add_trace(
            go.Bar(x=modelos, y=rmse_values, marker_color=colores,
                   text=[f"{v:,.0f}" for v in rmse_values], textposition='outside',
                   showlegend=False),
            row=1, col=2
        )
        
        # MAE
        mae_values = [resultados[m]['mae'] for m in modelos]
        fig.add_trace(
            go.Bar(x=modelos, y=mae_values, marker_color=colores,
                   text=[f"{v:,.0f}" for v in mae_values], textposition='outside',
                   showlegend=False),
            row=2, col=1
        )
        
        # MAPE
        mape_values = [resultados[m]['mape'] for m in modelos]
        fig.add_trace(
            go.Bar(x=modelos, y=mape_values, marker_color=colores,
                   text=[f"{v:.2f}%" for v in mape_values], textposition='outside',
                   showlegend=False),
            row=2, col=2
        )
        
        # Layout
        fig.update_yaxes(title_text="R²", row=1, col=1)
        fig.update_yaxes(title_text="m³/hr", row=1, col=2)
        fig.update_yaxes(title_text="m³/hr", row=2, col=1)
        fig.update_yaxes(title_text="%", row=2, col=2)
        
        fig.update_layout(
            title_text="Comparación de Modelos ML - Test Set",
            height=700,
            showlegend=False
        )
        
        return fig
    
    def lanzar_interfaz(self):
        """Crea y lanza la interfaz completa"""
        
        with gr.Blocks(title="ESVAL V3.0", theme=gr.themes.Soft()) as demo:
            
            # Header
            gr.HTML(self.crear_header_pronostico())
            
            with gr.Tabs():
                
                # TAB: Predicción 24 Horas
                with gr.Tab("📅 Predicción 24 Horas"):
                    gr.Markdown("### Predice la demanda para todo un día con temperatura constante")
                    
                    with gr.Row():
                        with gr.Column():
                            input_fecha_24h = gr.Textbox(
                                label="Fecha",
                                placeholder="DD/MM/YYYY",
                                value=datetime.now().strftime("%d/%m/%Y")
                            )
                            input_temp_24h = gr.Number(
                                label="🌡️ Temperatura Promedio del Día (°C)",
                                value=20.0,
                                minimum=-10,
                                maximum=45
                            )
                            btn_24h = gr.Button("🚀 Generar Predicción 24h", variant="primary")
                        
                        with gr.Column():
                            output_24h = gr.Markdown()
                    
                    plot_24h = gr.Plot(label="Gráfica 24 Horas")
                    
                    btn_24h.click(
                        fn=self.predecir_24_horas,
                        inputs=[input_fecha_24h, input_temp_24h],
                        outputs=[output_24h, plot_24h]
                    )
                
                # TAB: Predicción 72 Horas
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
                            
                            btn_72h = gr.Button("🚀 Generar Predicción 72h", variant="primary")
                        
                        with gr.Column():
                            output_72h = gr.Markdown()
                    
                    plot_72h = gr.Plot(label="Gráfica 3 Días")
                    
                    btn_72h.click(
                        fn=self.predecir_72_horas,
                        inputs=[input_fecha_72h, input_temp_dia1, input_temp_dia2, input_temp_dia3],
                        outputs=[output_72h, plot_72h]
                    )
                
                # TAB: Evaluación Testing
                with gr.Tab("📈 Evaluación Testing"):
                    gr.Markdown("### Evaluación del modelo en periodo de testing")
                    gr.Markdown("*Usa datos reales del periodo de test para validar performance*")
                    
                    btn_testing = gr.Button("🎯 Generar Evaluación", variant="primary")
                    output_testing = gr.Markdown()
                    plot_testing = gr.Plot(label="Resultados Testing")
                    
                    btn_testing.click(
                        fn=self.evaluar_testing,
                        inputs=[],
                        outputs=[output_testing, plot_testing]
                    )
                
                # TAB: Validación Rápida
                with gr.Tab("🔍 Validación Rápida"):
                    gr.Markdown("""
                    ### Suite de Tests Automáticos
                    
                    Esta herramienta ejecuta tests para verificar que el sistema está funcionando correctamente:
                    
                    - **🌡️ Sensibilidad a Temperatura:** Valida que temperatura alta → mayor demanda
                    - **📅 Coherencia Temporal:** Verifica predicciones consistentes entre días
                    - **📊 Rangos Históricos:** Confirma que predicciones están dentro de rangos razonables
                    - **⚙️ Estado del Sistema:** Chequea que todos los componentes están cargados
                    
                    Presiona el botón para ejecutar todos los tests (toma ~15-20 segundos).
                    """)
                    
                    btn_validacion = gr.Button("🚀 Ejecutar Validación", variant="primary", size="lg")
                    output_validacion = gr.Markdown()
                    
                    btn_validacion.click(
                        fn=self.ejecutar_validacion_rapida,
                        inputs=[],
                        outputs=[output_validacion]
                    )
                
                # TAB: Comparación de Modelos ML
                with gr.Tab("🔬 Comparación Modelos ML"):
                    gr.Markdown("""
                    ### Comparación de Múltiples Algoritmos de Machine Learning
                    
                    Este módulo compara el rendimiento de diferentes algoritmos:
                    
                    - **🚀 XGBoost V3.0:** Modelo actual en producción (Forecasting)
                    - **🌲 RandomForest:** Ensamble de árboles de decisión
                    - **💡 LightGBM:** Gradient Boosting optimizado (si está instalado)
                    
                    **Métricas evaluadas:**
                    - **R²:** Coeficiente de determinación (0-1, mayor es mejor)
                    - **RMSE:** Error cuadrático medio (menor es mejor)
                    - **MAE:** Error absoluto medio (menor es mejor)
                    - **MAPE:** Error porcentual (menor es mejor)
                    
                    ⏱️ **Tiempo estimado:** 2-3 minutos (entrena RandomForest y LightGBM desde cero)
                    """)
                    
                    btn_comparar = gr.Button("🔬 Ejecutar Comparación", variant="primary", size="lg")
                    
                    with gr.Row():
                        with gr.Column():
                            output_comparar = gr.Markdown()
                        with gr.Column():
                            plot_comparar = gr.Plot(label="Comparación Visual")
                    
                    btn_comparar.click(
                        fn=self.comparar_modelos_ml,
                        inputs=[],
                        outputs=[output_comparar, plot_comparar]
                    )
                
                # TAB: Información
                with gr.Tab("ℹ️ Información del Sistema"):
                    gr.Markdown(self.crear_tab_info())
            
            gr.Markdown("""
            ---
            **Sistema Predictivo ESVAL V3.0** | Modelo Forecasting con Temperatura  
            *Noviembre 2025*
            """)
        
        return demo


def main():
    """Función principal"""
    print("=" * 80)
    print("INICIANDO INTERFAZ ESVAL V3.0")
    print("=" * 80)
    
    # Crear instancia
    app = InterfazESVAL_V3()
    
    # Cargar todo
    if not app.cargar_todo():
        print("❌ Error cargando sistema")
        return
    
    print("\n✅ Sistema cargado correctamente")
    print("\n🚀 Lanzando interfaz...")
    
    # Crear y lanzar interfaz
    demo = app.lanzar_interfaz()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
