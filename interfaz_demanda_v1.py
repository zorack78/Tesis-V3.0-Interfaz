"""
Interfaz Gradio para Predicción de DEMANDA de Agua Potable

Diferencias con interfaz anterior:
- Predice DEMANDA (m³/hr) no volumen almacenado
- Usa modelo entrenado con Demanda_m3_hr
- Interpretación correcta: horario de inflexión máximo mediodía, valle madrugada
- Features basadas en LAGs de demanda
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SistemaPrediccionDemanda:
    def __init__(self):
        self.model = None
        self.features = []
        self.data_historico = None
        
    def cargar_modelo_y_datos(self):
        """Carga el modelo y datos históricos necesarios"""
        try:
            # Cargar modelo de demanda
            model_path = Path('models/demanda/demanda_xgboost_model.pkl')
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ Modelo de DEMANDA cargado")
            else:
                raise FileNotFoundError(f"No se encontró {model_path}")
            
            # Cargar features
            features_path = Path('models/demanda/features.txt')
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.features = [line.strip() for line in f.readlines()]
                print(f"✅ Features cargadas: {len(self.features)}")
            
            # Cargar datos históricos con demanda
            data_path = Path('data/processed/data_processed_demanda_valid.csv')
            if data_path.exists():
                self.data_historico = pd.read_csv(data_path)
                self.data_historico.columns = self.data_historico.columns.str.strip()
                self.data_historico['timestamp_utc'] = pd.to_datetime(
                    self.data_historico['timestamp_utc']
                )
                print(f"✅ Datos históricos cargados: {len(self.data_historico)} registros")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo/datos: {e}")
            return False
    
    def _estimar_lags_demanda(self, fecha, hora):
        """Estima LAGs de demanda basándose en patrones históricos"""
        
        # Valores por defecto conservadores
        default_lags = {
            'Demanda_lag_1h': 12000, 'Demanda_lag_2h': 12000,
            'Demanda_lag_24h': 12000, 'Demanda_lag_168h': 12000,
            'Demanda_rolling_mean_6h': 12000, 'Demanda_rolling_std_6h': 2000,
            'Demanda_rolling_mean_24h': 12000, 'Demanda_rolling_std_24h': 2000,
            'Demanda_diff_1h': 0, 'Demanda_diff_24h': 0,
            'Demanda_ratio_vs_24h': 1.0
        }
        
        # Filtrar datos similares
        dia_semana = fecha.weekday()
        mes = fecha.month
        
        datos_similares = self.data_historico[
            (self.data_historico['timestamp_utc'].dt.hour == hora) &
            (self.data_historico['timestamp_utc'].dt.weekday == dia_semana) &
            (self.data_historico['timestamp_utc'].dt.month == mes)
        ]
        
        if len(datos_similares) > 0:
            demanda_mediana = float(datos_similares['Demanda_m3_hr'].median())
            demanda_std = float(datos_similares['Demanda_m3_hr'].std())
        else:
            # Usar solo la hora
            datos_hora = self.data_historico[
                self.data_historico['timestamp_utc'].dt.hour == hora
            ]
            if len(datos_hora) > 0:
                demanda_mediana = float(datos_hora['Demanda_m3_hr'].median())
                demanda_std = float(datos_hora['Demanda_m3_hr'].std())
            else:
                demanda_mediana = 12000.0
                demanda_std = 2000.0
        
        return {
            'Demanda_lag_1h': demanda_mediana,
            'Demanda_lag_2h': demanda_mediana,
            'Demanda_lag_24h': demanda_mediana,
            'Demanda_lag_168h': demanda_mediana,
            'Demanda_rolling_mean_6h': demanda_mediana,
            'Demanda_rolling_std_6h': demanda_std if not np.isnan(demanda_std) else 2000.0,
            'Demanda_rolling_mean_24h': demanda_mediana,
            'Demanda_rolling_std_24h': demanda_std if not np.isnan(demanda_std) else 2000.0,
            'Demanda_diff_1h': 0.0,
            'Demanda_diff_24h': 0.0,
            'Demanda_ratio_vs_24h': 1.0
        }
    
    def crear_features(self, fecha, hora):
        """Crea features para predicción"""
        
        # Features temporales básicas
        features_dict = {
            'hour': hora,
            'day_of_week': fecha.weekday(),
            'month': fecha.month,
            'is_weekend': 1 if fecha.weekday() >= 5 else 0,
        }
        
        # Features cíclicas
        features_dict['hour_sin'] = np.sin(2 * np.pi * hora / 24)
        features_dict['hour_cos'] = np.cos(2 * np.pi * hora / 24)
        features_dict['day_of_week_sin'] = np.sin(2 * np.pi * fecha.weekday() / 7)
        features_dict['day_of_week_cos'] = np.cos(2 * np.pi * fecha.weekday() / 7)
        
        # Features de eventos (simplificado - podrías cargar calendario)
        features_dict['feriado'] = 0
        features_dict['temporada_turistica_alta'] = 1 if fecha.month in [1, 2, 7, 12] else 0
        
        # LAGs de demanda (estimados)
        lags = self._estimar_lags_demanda(fecha, hora)
        features_dict.update(lags)
        
        return features_dict
    
    def predecir_demanda(self, fecha_str, hora):
        """Predice demanda para una fecha y hora específica"""
        try:
            fecha = pd.to_datetime(fecha_str)
            
            # Crear features
            features_dict = self.crear_features(fecha, hora)
            
            # Ordenar features según el modelo
            X = pd.DataFrame([features_dict])[self.features]
            
            # Predicción
            demanda_pred = self.model.predict(X)[0]
            
            # Información adicional
            dia_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][fecha.weekday()]
            
            # Determinar período del día
            if 0 <= hora < 6:
                periodo = "Madrugada"
            elif 6 <= hora < 12:
                periodo = "Mañana"
            elif 12 <= hora < 18:
                periodo = "Tarde"
            else:
                periodo = "Noche"
            
            # Determinar estación (hemisferio sur)
            if fecha.month in [12, 1, 2]:
                estacion = "Verano"
            elif fecha.month in [3, 4, 5]:
                estacion = "Otoño"
            elif fecha.month in [6, 7, 8]:
                estacion = "Invierno"
            else:
                estacion = "Primavera"
            
            info = {
                'fecha': fecha.strftime('%d/%m/%Y'),
                'dia_semana': dia_nombre,
                'hora': f"{hora:02d}:00",
                'periodo': periodo,
                'estacion': estacion,
                'fin_semana': '✅ Fin de semana' if fecha.weekday() >= 5 else '📅 Día laboral'
            }
            
            return demanda_pred, info
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def predecir_dia_completo(self, fecha_str):
        """Predice demanda para todas las horas del día"""
        try:
            predicciones = []
            errores = []
            
            for hora in range(24):
                demanda, info = self.predecir_demanda(fecha_str, hora)
                
                if demanda is not None:
                    predicciones.append({
                        'hora': hora,
                        'demanda': demanda,
                        'info': info
                    })
                else:
                    errores.append(f"Predicción None para hora {hora}")
            
            if len(predicciones) == 0:
                return None, f"❌ No se pudieron generar predicciones\n\nErrores:\n" + "\n".join(errores)
            
            return predicciones, None
            
        except Exception as e:
            print(f"Error en predicción día completo: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)

# Inicializar sistema
print("🔄 Inicializando sistema de predicción de DEMANDA...")
sistema = SistemaPrediccionDemanda()

if not sistema.cargar_modelo_y_datos():
    raise Exception("No se pudo cargar el modelo y/o datos")

# Funciones para Gradio
def wrapper_prediccion_simple(fecha_str, hora):
    """Wrapper para predicción puntual"""
    try:
        hora = int(hora)
        if not (0 <= hora <= 23):
            return "❌ Error: La hora debe estar entre 0 y 23"
        
        demanda, info = sistema.predecir_demanda(fecha_str, hora)
        
        if demanda is None:
            return f"❌ Error en predicción: {info}"
        
        # Formatear salida
        output = f"""
📅 **Fecha:** {info['fecha']} ({info['dia_semana']})
⏰ **Hora:** {info['hora']}
🌍 **Contexto:**
   • Estación: {info['estacion']}
   • Período día: {info['periodo']}
   • Día: {info['fin_semana']}

---

## 💧 PREDICCIÓN DE DEMANDA:
### **{demanda:,.0f} m³/hora**

*La demanda es el caudal consumido/demandado por hora.*

---

📊 **Interpretación:**
• Esta es la cantidad de agua que el sistema necesita **entregar** en esa hora
• El sistema debe tener capacidad para satisfacer esta demanda
• **Horarios de inflexión:**
  - Máximo típico: 12:00-14:00 (mediodía)
  - Mínimo típico: 02:00-05:00 (madrugada)
"""
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def wrapper_prediccion_simple_con_grafico(fecha_str, hora):
    """Wrapper para predicción puntual con gráfico de contexto INTERACTIVO"""
    try:
        hora = int(hora)
        if not (0 <= hora <= 23):
            return None, "❌ Error: La hora debe estar entre 0 y 23"
        
        demanda, info = sistema.predecir_demanda(fecha_str, hora)
        
        if demanda is None:
            return None, f"❌ Error en predicción: {info}"
        
        # Predecir el día completo para contexto
        predicciones_dia, _ = sistema.predecir_dia_completo(fecha_str)
        
        # Crear gráfico INTERACTIVO con Plotly
        horas = [p['hora'] for p in predicciones_dia]
        demandas = [p['demanda'] for p in predicciones_dia]
        
        fig = go.Figure()
        
        # Línea de demanda del día
        fig.add_trace(go.Scatter(
            x=horas,
            y=demandas,
            mode='lines+markers',
            name='Demanda del Día',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB'),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.3)',
            hovertemplate='<b>Hora:</b> %{x}:00<br>' +
                          '<b>Demanda:</b> %{y:,.0f} m³/hr<br>' +
                          '<extra></extra>'
        ))
        
        # Punto destacado para hora seleccionada
        fig.add_trace(go.Scatter(
            x=[hora],
            y=[demanda],
            mode='markers',
            name=f'Hora Seleccionada ({hora}:00)',
            marker=dict(size=20, color='red', symbol='star',
                       line=dict(width=2, color='darkred')),
            hovertemplate=f'<b>⭐ HORA SELECCIONADA</b><br>' +
                          f'<b>Hora:</b> {hora}:00<br>' +
                          f'<b>Demanda:</b> {demanda:,.0f} m³/hr<br>' +
                          '<extra></extra>'
        ))
        
        # Línea vertical en hora seleccionada
        fig.add_vline(x=hora, line_dash="dash", line_color="red",
                     opacity=0.5, line_width=2)
        
        # Configuración del layout
        fig.update_layout(
            title=dict(
                text=f'📊 Demanda del Día: {info["fecha"]} ({info["dia_semana"]})<br>' +
                     f'<sub>Hora seleccionada: {hora}:00 = {demanda:,.0f} m³/hr</sub>',
                font=dict(size=16, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Hora del Día',
                titlefont=dict(size=14, color='#333'),
                tickmode='linear',
                tick0=0,
                dtick=2,
                range=[-0.5, 23.5],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title='Demanda (m³/hr)',
                titlefont=dict(size=14, color='#333'),
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=60, r=30, t=100, b=60)
        )
        
        # Formatear salida
        output = f"""
📅 **Fecha:** {info['fecha']} ({info['dia_semana']})
⏰ **Hora:** {info['hora']}
🌍 **Contexto:**
   • Estación: {info['estacion']}
   • Período día: {info['periodo']}
   • Día: {info['fin_semana']}

---

## 💧 PREDICCIÓN DE DEMANDA:
### **{demanda:,.0f} m³/hora**

---

📊 **Interpretación:**
• Esta es la cantidad de agua que el sistema necesita **entregar** en esa hora
• El gráfico muestra el contexto del día completo
• **Herramientas interactivas:** Zoom, Pan, Hover para ver valores
• **Horarios de inflexión:**
  - Máximo típico: 12:00-14:00 (mediodía)
  - Mínimo típico: 02:00-05:00 (madrugada)
"""
        
        return fig, output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"



def wrapper_prediccion_dia_con_grafico(fecha_str):
    """Wrapper para predicción de día completo con gráficos INTERACTIVOS"""
    try:
        predicciones, error = sistema.predecir_dia_completo(fecha_str)
        
        if predicciones is None:
            return None, f"❌ Error: {error}"
        
        # Calcular estadísticas
        demandas = [p['demanda'] for p in predicciones]
        horas = [p['hora'] for p in predicciones]
        total_dia = sum(demandas)
        promedio = np.mean(demandas)
        max_demanda = max(demandas)
        min_demanda = min(demandas)
        hora_pico = predicciones[np.argmax(demandas)]['hora']
        hora_valle = predicciones[np.argmin(demandas)]['hora']
        
        # Info general
        info = predicciones[0]['info']
        
        # Crear figura con 2 subplots usando Plotly
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📊 Patrón Horario del Día', '📊 Distribución por Períodos'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # GRÁFICO 1: Serie temporal
        fig.add_trace(
            go.Scatter(
                x=horas,
                y=demandas,
                mode='lines+markers',
                name='Demanda Predicha',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8, color='#2E86AB'),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.3)',
                hovertemplate='<b>Hora:</b> %{x}:00<br>' +
                              '<b>Demanda:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Punto máximo
        fig.add_trace(
            go.Scatter(
                x=[hora_pico],
                y=[max_demanda],
                mode='markers',
                name=f'Máximo: {hora_pico}:00',
                marker=dict(size=20, color='red', symbol='star',
                           line=dict(width=2, color='darkred')),
                hovertemplate=f'<b>🔴 MÁXIMO</b><br>' +
                              f'<b>Hora:</b> {hora_pico}:00<br>' +
                              f'<b>Demanda:</b> {max_demanda:,.0f} m³/hr<br>' +
                              '<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Punto mínimo
        fig.add_trace(
            go.Scatter(
                x=[hora_valle],
                y=[min_demanda],
                mode='markers',
                name=f'Mínimo: {hora_valle}:00',
                marker=dict(size=20, color='green', symbol='star',
                           line=dict(width=2, color='darkgreen')),
                hovertemplate=f'<b>🟢 MÍNIMO</b><br>' +
                              f'<b>Hora:</b> {hora_valle}:00<br>' +
                              f'<b>Demanda:</b> {min_demanda:,.0f} m³/hr<br>' +
                              '<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Línea de promedio
        fig.add_hline(
            y=promedio,
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"Promedio: {promedio:,.0f} m³/hr",
            annotation_position="right",
            row=1, col=1
        )
        
        # GRÁFICO 2: Demanda por períodos
        periodos = ['Madrugada<br>(00-06)', 'Mañana<br>(06-12)',
                   'Tarde<br>(12-18)', 'Noche<br>(18-24)']
        madrugada = sum([p['demanda'] for p in predicciones if 0 <= p['hora'] < 6])
        manana = sum([p['demanda'] for p in predicciones if 6 <= p['hora'] < 12])
        tarde = sum([p['demanda'] for p in predicciones if 12 <= p['hora'] < 18])
        noche = sum([p['demanda'] for p in predicciones if 18 <= p['hora'] < 24])
        
        valores = [madrugada, manana, tarde, noche]
        colores = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
        
        fig.add_trace(
            go.Bar(
                x=periodos,
                y=valores,
                marker=dict(color=colores, line=dict(color='black', width=2)),
                text=[f'{val:,.0f} m³<br>({val/total_dia*100:.1f}%)' for val in valores],
                textposition='outside',
                hovertemplate='<b>Período:</b> %{x}<br>' +
                              '<b>Demanda Total:</b> %{y:,.0f} m³<br>' +
                              '<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Configuración del layout
        fig.update_xaxes(
            title_text="Hora del Día",
            tickmode='linear',
            tick0=0,
            dtick=2,
            range=[-0.5, 23.5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Demanda (m³/hr)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=1, col=1
        )
        
        fig.update_xaxes(
            showgrid=False,
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="Demanda Total (m³)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text=f'📊 Predicción Día Completo: {info["fecha"]} ({info["dia_semana"]}) - {info["estacion"]}',
                font=dict(size=16, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=900,
            margin=dict(l=60, r=30, t=120, b=60)
        )
        
        # Formatear salida
        output = f"""
📅 **PREDICCIÓN DÍA COMPLETO:** {info['fecha']}
📊 **Día:** {info['dia_semana']}
🌍 **Estación:** {info['estacion']}

---

## 📈 ESTADÍSTICAS DEL DÍA:

• **Demanda total:** {total_dia:,.0f} m³
• **Promedio hora:** {promedio:,.0f} m³/hora
• **Horario máximo:** {hora_pico:02d}:00 → {max_demanda:,.0f} m³/hora
• **Horario mínimo:** {hora_valle:02d}:00 → {min_demanda:,.0f} m³/hora
• **Variación:** {((max_demanda - min_demanda) / promedio * 100):.1f}%

---

💡 **Gráficos Interactivos:** Use zoom, pan y hover para explorar los datos
📊 El gráfico superior muestra el patrón horario completo
📊 El gráfico inferior muestra la distribución por períodos
"""
        
        return fig, output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"



def wrapper_prediccion_72h(fecha_inicio_str):
    """Wrapper para predicción de 72 horas (3 días) con gráficos INTERACTIVOS"""
    try:
        fecha_inicio = datetime.strptime(fecha_inicio_str, '%Y-%m-%d')
        
        # Predecir 3 días
        todas_predicciones = []
        for dia in range(3):
            fecha_actual = fecha_inicio + timedelta(days=dia)
            fecha_str = fecha_actual.strftime('%Y-%m-%d')
            
            predicciones, error = sistema.predecir_dia_completo(fecha_str)
            if predicciones is None:
                return None, f"❌ Error: {error}"
            
            todas_predicciones.extend(predicciones)
        
        # Preparar datos - CORREGIR formato de fecha
        timestamps = []
        for p in todas_predicciones:
            # Convertir fecha que puede venir en formato DD/MM/YYYY
            fecha_str = p['info']['fecha']
            try:
                # Intentar formato YYYY-MM-DD
                fecha_dt = datetime.strptime(fecha_str, '%Y-%m-%d')
            except:
                # Intentar formato DD/MM/YYYY
                fecha_dt = datetime.strptime(fecha_str, '%d/%m/%Y')
            
            timestamp = fecha_dt.replace(hour=p['hora'])
            timestamps.append(timestamp)
        
        demandas = [p['demanda'] for p in todas_predicciones]
        
        # Crear figura con 2 subplots interactivos
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📊 Serie Temporal 72 Horas', '📊 Comparación por Día'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # GRÁFICO 1: Serie temporal completa
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=demandas,
                mode='lines+markers',
                name='Demanda Predicha',
                line=dict(color='#2E86AB', width=2.5),
                marker=dict(size=4, color='#2E86AB'),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.3)',
                hovertemplate='<b>Fecha:</b> %{x|%d-%b %H:%M}<br>' +
                              '<b>Demanda:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Marcar horarios de inflexión máximo y mínimo
        max_idx = np.argmax(demandas)
        min_idx = np.argmin(demandas)
        
        fig.add_trace(
            go.Scatter(
                x=[timestamps[max_idx]],
                y=[demandas[max_idx]],
                mode='markers',
                name=f'Máximo: {demandas[max_idx]:,.0f} m³/hr',
                marker=dict(size=18, color='red', symbol='star',
                           line=dict(width=2, color='darkred')),
                hovertemplate=f'<b>🔴 MÁXIMO</b><br>' +
                              f'<b>Fecha:</b> {timestamps[max_idx].strftime("%d-%b %H:%M")}<br>' +
                              f'<b>Demanda:</b> {demandas[max_idx]:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[timestamps[min_idx]],
                y=[demandas[min_idx]],
                mode='markers',
                name=f'Mínimo: {demandas[min_idx]:,.0f} m³/hr',
                marker=dict(size=18, color='green', symbol='star',
                           line=dict(width=2, color='darkgreen')),
                hovertemplate=f'<b>🟢 MÍNIMO</b><br>' +
                              f'<b>Fecha:</b> {timestamps[min_idx].strftime("%d-%b %H:%M")}<br>' +
                              f'<b>Demanda:</b> {demandas[min_idx]:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Agregar líneas verticales para marcar cada día
        for dia in range(3):
            fecha_dia = fecha_inicio + timedelta(days=dia)
            fig.add_vline(
                x=fecha_dia.timestamp() * 1000,  # Convertir a milisegundos
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                line_width=1,
                row=1, col=1
            )
        
        # GRÁFICO 2: Comparación por día
        dias_labels = []
        dias_promedios = []
        dias_maximos = []
        dias_minimos = []
        
        for dia in range(3):
            inicio_idx = dia * 24
            fin_idx = (dia + 1) * 24
            demandas_dia = demandas[inicio_idx:fin_idx]
            fecha_dia = fecha_inicio + timedelta(days=dia)
            
            dias_labels.append(fecha_dia.strftime('%d-%b'))
            dias_promedios.append(np.mean(demandas_dia))
            dias_maximos.append(max(demandas_dia))
            dias_minimos.append(min(demandas_dia))
        
        # Barras agrupadas
        fig.add_trace(
            go.Bar(
                x=dias_labels,
                y=dias_promedios,
                name='Promedio',
                marker=dict(color='#2E86AB', line=dict(color='black', width=1)),
                text=[f'{val:,.0f}' for val in dias_promedios],
                textposition='outside',
                hovertemplate='<b>Día:</b> %{x}<br>' +
                              '<b>Promedio:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=dias_labels,
                y=dias_maximos,
                name='Máximo',
                marker=dict(color='#A23B72', line=dict(color='black', width=1)),
                text=[f'{val:,.0f}' for val in dias_maximos],
                textposition='outside',
                hovertemplate='<b>Día:</b> %{x}<br>' +
                              '<b>Máximo:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=dias_labels,
                y=dias_minimos,
                name='Mínimo',
                marker=dict(color='#F18F01', line=dict(color='black', width=1)),
                text=[f'{val:,.0f}' for val in dias_minimos],
                textposition='outside',
                hovertemplate='<b>Día:</b> %{x}<br>' +
                              '<b>Mínimo:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Configuración del layout
        fig.update_xaxes(
            title_text="Fecha y Hora",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Demanda (m³/hr)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="Día",
            showgrid=False,
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="Demanda (m³/hr)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text=f'📊 Predicción 72 Horas: {fecha_inicio.strftime("%d-%m-%Y")} al {(fecha_inicio + timedelta(days=2)).strftime("%d-%m-%Y")}',
                font=dict(size=16, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group',
            height=950,
            margin=dict(l=60, r=30, t=120, b=60)
        )
        
        # Estadísticas generales
        total_72h = sum(demandas)
        promedio_72h = np.mean(demandas)
        max_72h = max(demandas)
        min_72h = min(demandas)
        
        output = f"""
📅 **PREDICCIÓN 72 HORAS (3 DÍAS):** {fecha_inicio.strftime('%d-%m-%Y')} al {(fecha_inicio + timedelta(days=2)).strftime('%d-%m-%Y')}

---

## 📊 ESTADÍSTICAS GENERALES (72 horas):

• **Demanda total:** {total_72h:,.0f} m³
• **Promedio hora:** {promedio_72h:,.0f} m³/hr
• **Máximo:** {max_72h:,.0f} m³/hr (hora {timestamps[max_idx].strftime('%d-%b %H:%M')})
• **Mínimo:** {min_72h:,.0f} m³/hr (hora {timestamps[min_idx].strftime('%d-%b %H:%M')})
• **Variación:** {((max_72h - min_72h) / promedio_72h * 100):.1f}%

---

## 📅 RESUMEN POR DÍA:

"""
        for dia in range(3):
            inicio_idx = dia * 24
            fin_idx = (dia + 1) * 24
            demandas_dia = demandas[inicio_idx:fin_idx]
            fecha_dia = fecha_inicio + timedelta(days=dia)
            
            output += f"""
**Día {dia+1}: {fecha_dia.strftime('%d-%m-%Y (%A)')}**
• Total: {sum(demandas_dia):,.0f} m³
• Promedio: {np.mean(demandas_dia):,.0f} m³/hr
• Máximo: {max(demandas_dia):,.0f} m³/hr
• Mínimo: {min(demandas_dia):,.0f} m³/hr

"""
        
        output += """
---

💡 **Gráficos Interactivos:** Use zoom, pan y hover para explorar los 3 días completos
📊 Herramientas: Zoom, Pan, Reset, Box/Lasso Select, mostrar valores al pasar cursor
"""
        
        return fig, output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

def wrapper_prediccion_dia(fecha_str):
    """Wrapper para predicción de día completo"""
    try:
        predicciones, error = sistema.predecir_dia_completo(fecha_str)
        
        if predicciones is None:
            return f"❌ Error: {error}"
        
        # Calcular estadísticas
        demandas = [p['demanda'] for p in predicciones]
        total_dia = sum(demandas)
        promedio = np.mean(demandas)
        max_demanda = max(demandas)
        min_demanda = min(demandas)
        hora_pico = predicciones[np.argmax(demandas)]['hora']
        hora_valle = predicciones[np.argmin(demandas)]['hora']
        
        # Info general
        info = predicciones[0]['info']
        
        # Formatear salida
        output = f"""
📅 **PREDICCIÓN DÍA COMPLETO:** {info['fecha']}
📊 **Día:** {info['dia_semana']}
🌍 **Estación:** {info['estacion']}

---

## 📈 ESTADÍSTICAS DEL DÍA:

• **Demanda total:** {total_dia:,.0f} m³
• **Promedio hora:** {promedio:,.0f} m³/hora
• **Horario máximo:** {hora_pico:02d}:00 → {max_demanda:,.0f} m³/hora
• **Horario mínimo:** {hora_valle:02d}:00 → {min_demanda:,.0f} m³/hora
• **Variación:** {((max_demanda - min_demanda) / promedio * 100):.1f}%

---

## ⏰ DEMANDA POR PERÍODOS:

"""
        # Calcular por períodos
        madrugada = sum([p['demanda'] for p in predicciones if 0 <= p['hora'] < 6])
        manana = sum([p['demanda'] for p in predicciones if 6 <= p['hora'] < 12])
        tarde = sum([p['demanda'] for p in predicciones if 12 <= p['hora'] < 18])
        noche = sum([p['demanda'] for p in predicciones if 18 <= p['hora'] < 24])
        
        output += f"""
• **Madrugada (00-06):** {madrugada:,.0f} m³
• **Mañana (06-12):** {manana:,.0f} m³
• **Tarde (12-18):** {tarde:,.0f} m³
• **Noche (18-24):** {noche:,.0f} m³

---

## 📋 DETALLE HORARIO:

| Hora | Demanda (m³/hr) | Período |
|------|----------------|---------|
"""
        
        for p in predicciones:
            output += f"| {p['hora']:02d}:00 | {p['demanda']:>12,.0f} | {p['info']['periodo']:>10} |\n"
        
        output += """
---

💡 **Nota:** La demanda representa el consumo real de agua por hora.
Horarios con mayor demanda requieren mayor capacidad del sistema.
"""
        
        return output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


def generar_evaluacion_testing():
    """Genera evaluación completa del período de testing"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from PIL import Image
        from io import BytesIO
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Cargar datos completos con demanda
        data_path = Path('data/processed/data_processed_demanda_valid.csv')
        if not data_path.exists():
            return None, "❌ No se encontró data_processed_demanda_valid.csv"
        
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        
        # Verificar columnas necesarias
        if 'Demanda_m3_hr' not in df.columns:
            return None, "❌ No se encontró columna Demanda_m3_hr"
        
        # Crear LAGs y features derivadas de Demanda (igual que en entrenamiento)
        df['Demanda_lag_1h'] = df['Demanda_m3_hr'].shift(1)
        df['Demanda_lag_2h'] = df['Demanda_m3_hr'].shift(2)
        df['Demanda_lag_24h'] = df['Demanda_m3_hr'].shift(24)
        df['Demanda_lag_168h'] = df['Demanda_m3_hr'].shift(168)
        
        df['Demanda_rolling_mean_6h'] = df['Demanda_m3_hr'].rolling(
            window=6, min_periods=1).mean()
        df['Demanda_rolling_std_6h'] = df['Demanda_m3_hr'].rolling(
            window=6, min_periods=1).std()
        df['Demanda_rolling_mean_24h'] = df['Demanda_m3_hr'].rolling(
            window=24, min_periods=1).mean()
        df['Demanda_rolling_std_24h'] = df['Demanda_m3_hr'].rolling(
            window=24, min_periods=1).std()
        
        df['Demanda_diff_1h'] = df['Demanda_m3_hr'].diff()
        df['Demanda_diff_24h'] = df['Demanda_m3_hr'].diff(24)
        df['Demanda_ratio_vs_24h'] = df['Demanda_m3_hr'] / (
            df['Demanda_lag_24h'] + 1)
        
        # Limpiar NaN
        df_clean = df.dropna(subset=[
            'Demanda_m3_hr',
            'Demanda_lag_1h', 'Demanda_lag_24h', 'Demanda_lag_168h',
            'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h'
        ]).reset_index(drop=True)
        
        # FILTRAR SOLO AGOSTO-SEPTIEMBRE 2025 (período de testing real)
        df_test = df_clean[
            (df_clean['timestamp_utc'] >= '2025-08-01') &
            (df_clean['timestamp_utc'] <= '2025-09-30')
        ].copy()
        
        if len(df_test) == 0:
            return None, "❌ No hay datos en el período Agosto-Septiembre 2025"
        
        # FILTRAR OUTLIERS EXTREMOS (demanda negativa o muy alta)
        # Usar límites razonables: entre 0 y 30,000 m³/hr
        df_test = df_test[
            (df_test['Demanda_m3_hr'] >= 0) &
            (df_test['Demanda_m3_hr'] <= 30000)
        ].reset_index(drop=True)
        
        # Preparar features y target
        X_test = df_test[sistema.features]
        y_test = df_test['Demanda_m3_hr']
        timestamps = df_test['timestamp_utc']
        
        # Predecir con modelo actual
        y_pred = sistema.model.predict(X_test)
        
        # Calcular métricas del modelo actual
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Crear figura INTERACTIVA con dos subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📊 Serie Temporal: Demanda Real vs Predicción',
                           '📈 Correlación: Real vs Predicción'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # GRÁFICO 1: Serie temporal completa
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=y_test.values,
                mode='lines',
                name='Demanda Real',
                line=dict(color='black', width=2),
                hovertemplate='<b>Fecha:</b> %{x|%d-%b-%Y %H:%M}<br>' +
                              '<b>Real:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=y_pred,
                mode='lines',
                name='Predicción XGBoost',
                line=dict(color='#2E86AB', width=1.5),
                hovertemplate='<b>Fecha:</b> %{x|%d-%b-%Y %H:%M}<br>' +
                              '<b>Predicción:</b> %{y:,.0f} m³/hr<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # GRÁFICO 2: Scatter plot Real vs Predicción
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predicciones',
                marker=dict(size=6, color='#2E86AB', opacity=0.5),
                hovertemplate='<b>Real:</b> %{x:,.0f} m³/hr<br>' +
                              '<b>Predicción:</b> %{y:,.0f} m³/hr<br>' +
                              '<b>Error:</b> %{customdata:,.0f} m³/hr<br>' +
                              '<extra></extra>',
                customdata=y_pred - y_test.values
            ),
            row=2, col=1
        )
        
        # Línea diagonal perfecta
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Línea perfecta</b><br>' +
                              'Real = Predicción<br>' +
                              '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Configuración de ejes
        fig.update_xaxes(
            title_text="Fecha",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Demanda (m³/hr)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="Demanda Real (m³/hr)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="Demanda Predicha (m³/hr)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            row=2, col=1
        )
        
        # Layout general
        fig.update_layout(
            title=dict(
                text='🧪 Evaluación Testing: Agosto-Septiembre 2025',
                font=dict(size=16, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=1000,
            margin=dict(l=60, r=30, t=120, b=60)
        )
        
        # Añadir anotación con métricas en el segundo gráfico
        metrics_annotation = (f'<b>Métricas:</b><br>' +
                            f'R² = {r2:.4f}<br>' +
                            f'RMSE = {rmse:,.0f} m³/hr<br>' +
                            f'MAE = {mae:,.0f} m³/hr<br>' +
                            f'MAPE = {mape:.2f}%')
        
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain",
            x=0.05, y=0.95,
            xanchor='left', yanchor='top',
            text=metrics_annotation,
            showarrow=False,
            bgcolor='rgba(245, 222, 179, 0.8)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11),
            align='left'
        )
        
        # Fechas del período
        fecha_inicio = timestamps.min().strftime('%d-%m-%Y')
        fecha_fin = timestamps.max().strftime('%d-%m-%Y')
        
        # Crear tabla de métricas
        calidad_r2 = '✅ EXCELENTE' if r2 > 0.95 else '⚠️ BUENO' if r2 > 0.90 else '❌ MEJORABLE'
        calidad_mape = '✅ EXCELENTE' if mape < 5 else '⚠️ BUENO' if mape < 10 else '❌ MEJORABLE'
        
        output = f"""
# 🧪 EVALUACIÓN DEL MODELO - PERÍODO DE TESTING

## � Período Evaluado: Agosto-Septiembre 2025
**Fechas:** {fecha_inicio} al {fecha_fin}  
**Registros:** {len(y_test):,} horas (sin outliers)

---

## �📊 Métricas de Calidad

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R² (Coef. Determinación)** | {r2:.4f} | {calidad_r2} |
| **RMSE (Error Cuadrático)** | {rmse:,.0f} m³/hr | Error típico |
| **MAE (Error Absoluto)** | {mae:,.0f} m³/hr | Desviación promedio |
| **MAPE (Error Porcentual)** | {mape:.2f}% | {calidad_mape} |

---

## 📈 Estadísticas del Período

• **Demanda real promedio:** {y_test.mean():,.0f} m³/hr
• **Demanda predicha promedio:** {y_pred.mean():,.0f} m³/hr
• **Demanda real máxima:** {y_test.max():,.0f} m³/hr
• **Demanda real mínima:** {y_test.min():,.0f} m³/hr
• **Diferencia prom. predicción:** {abs(y_pred.mean() - y_test.mean()):,.0f} m³/hr

---

## 🎯 Interpretación de Resultados

### R² = {r2:.4f}
El modelo explica el **{r2*100:.2f}%** de la variabilidad en la demanda.
{'Esto indica un ajuste EXCELENTE.' if r2 > 0.95 else 'Esto indica un ajuste BUENO.' if r2 > 0.90 else 'Hay margen de mejora.'}

### MAPE = {mape:.2f}%
En promedio, las predicciones se desvían un **{mape:.2f}%** del valor real.
{'Precisión EXCELENTE para aplicaciones prácticas.' if mape < 5 else 'Precisión BUENA, aceptable para planificación.' if mape < 10 else 'Se recomienda mejorar el modelo.'}

### RMSE = {rmse:,.0f} m³/hr
El error típico es de aproximadamente **{rmse:,.0f} m³/hr**.
Esto representa un **{(rmse/y_test.mean())*100:.2f}%** de la demanda promedio.

---

## 📊 Análisis Visual

**Gráfico Superior:** Muestra la serie temporal completa del período de testing.
- Línea negra: Demanda real observada
- Línea azul: Predicción del modelo

**Gráfico Inferior:** Correlación entre valores reales y predichos.
- Puntos cerca de la línea roja diagonal = Buenas predicciones
- Desviaciones de la línea = Errores del modelo

---

💡 **Conclusión:** El modelo XGBoost muestra un rendimiento {'EXCELENTE' if r2 > 0.95 and mape < 5 else 'BUENO' if r2 > 0.90 and mape < 10 else 'ACEPTABLE'} en el período de testing,
con alta capacidad predictiva para la demanda de agua potable.

🔧 **Gráficos Interactivos:** Use herramientas de zoom, pan y hover para explorar los datos en detalle.
"""
        
        return fig, output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# Interfaz Gradio
with gr.Blocks(title="Predicción de Demanda de Agua", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💧 Sistema de Predicción de DEMANDA de Agua Potable
    
    Este sistema predice la **demanda/consumo de agua** (m³/hora) para el Gran Valparaíso.
    
    **📊 Características del modelo:**
    - Variable objetivo: Demanda_m3_hr (consumo real)
    - Modelo: XGBoost optimizado
    - Precisión: R² = 0.97, MAPE = 3%
    - Patrones validados: Pico a mediodía, valle en madrugada
    
    **🎯 Interpretación:**
    - **Demanda ALTA:** Sistema debe entregar más agua (típico 12:00-14:00)
    - **Demanda BAJA:** Menor consumo (típico 02:00-05:00)
    - **Verano:** Mayor demanda (temperaturas altas)
    - **Invierno:** Menor demanda (temperaturas bajas)
    
    **⏰ Horarios de Inflexión:**
    - Horarios donde la demanda cambia significativamente
    - Máximo: Mediodía (12:00-14:00) - Mayor consumo
    - Mínimo: Madrugada (02:00-05:00) - Menor consumo
    """)
    
    with gr.Tab("🎯 Predicción Puntual"):
        gr.Markdown("### Predecir demanda para una fecha y hora específica")
        
        with gr.Row():
            fecha_input = gr.Textbox(
                label="Fecha (YYYY-MM-DD)",
                value="2025-12-25",
                placeholder="2025-12-25"
            )
            hora_input = gr.Slider(
                minimum=0,
                maximum=23,
                step=1,
                value=12,
                label="Hora del día (0-23)"
            )
        
        btn_predecir = gr.Button("🔮 Predecir Demanda", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=1):
                output_simple_texto = gr.Markdown()
            with gr.Column(scale=2):
                output_simple_grafico = gr.Plot(label="Contexto del Día (Interactivo)")
        
        btn_predecir.click(
            fn=wrapper_prediccion_simple_con_grafico,
            inputs=[fecha_input, hora_input],
            outputs=[output_simple_grafico, output_simple_texto]
        )
    
    with gr.Tab("📅 Predicción Día Completo"):
        gr.Markdown("### Predecir demanda para todas las horas de un día")
        
        fecha_dia_input = gr.Textbox(
            label="Fecha (YYYY-MM-DD)",
            value="2025-12-25",
            placeholder="2025-12-25"
        )
        
        btn_predecir_dia = gr.Button("📊 Predecir Día Completo",
                                     variant="primary")
        
        with gr.Row():
            with gr.Column(scale=1):
                output_dia_texto = gr.Markdown()
            with gr.Column(scale=2):
                output_dia_grafico = gr.Plot(label="Gráfico Día Completo (Interactivo)")
        
        btn_predecir_dia.click(
            fn=wrapper_prediccion_dia_con_grafico,
            inputs=fecha_dia_input,
            outputs=[output_dia_grafico, output_dia_texto]
        )
    
    with gr.Tab("📈 Predicción 72 Horas (3 Días)"):
        gr.Markdown("### Predecir demanda para 3 días completos con gráfico")
        
        fecha_72h_input = gr.Textbox(
            label="Fecha de Inicio (YYYY-MM-DD)",
            value="2025-12-25",
            placeholder="2025-12-25"
        )
        
        btn_predecir_72h = gr.Button("📊 Predecir 72 Horas", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                output_72h_texto = gr.Markdown()
            with gr.Column(scale=2):
                output_72h_grafico = gr.Plot(label="Gráfico 72h (Interactivo)")
        
        btn_predecir_72h.click(
            fn=wrapper_prediccion_72h,
            inputs=fecha_72h_input,
            outputs=[output_72h_grafico, output_72h_texto]
        )
    
    with gr.Tab("🧪 Evaluación Testing"):
        gr.Markdown("""
        ### Evaluación del Modelo en Período de Testing
        
        Esta pestaña muestra el rendimiento del modelo XGBoost en el conjunto de testing.
        Incluye métricas de calidad (R², RMSE, MAE, MAPE) y visualizaciones comparativas.
        """)
        
        btn_testing = gr.Button("📊 Generar Evaluación Testing",
                               variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                output_testing_texto = gr.Markdown()
            with gr.Column(scale=2):
                output_testing_grafico = gr.Plot(label="Evaluación Testing (Interactivo)")
        
        btn_testing.click(
            fn=generar_evaluacion_testing,
            inputs=None,
            outputs=[output_testing_grafico, output_testing_texto]
        )
    
    gr.Markdown("""
    ---
    ### ℹ️ Información Adicional
    
    **¿Qué es la Demanda?**
    - Es el caudal de agua consumido/demandado por hora (m³/hr)
    - Se calcula como: Demanda = Qin - ΔVolumen
    - Representa el consumo real del sistema
    
    **Patrones Típicos:**
    - **Madrugada (00-06):** Demanda baja (~5,000-8,000 m³/hr)
    - **Mañana (06-12):** Demanda creciente
    - **Mediodía (12-14):** **Demanda MÁXIMA** (~17,000 m³/hr) - Horario de inflexión
    - **Tarde (14-18):** Demanda alta
    - **Noche (18-24):** Demanda decreciente
    
    **Factores que afectan la demanda:**
    - ☀️ Temperatura (mayor temp → mayor demanda)
    - 📅 Día de la semana (fines de semana varían)
    - 🎉 Feriados y eventos especiales
    - 🏖️ Temporada turística
    """)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 INICIANDO INTERFAZ DE PREDICCIÓN DE DEMANDA")
    print("="*80)
    demo.launch(server_name="0.0.0.0", server_port=7870, share=False)
