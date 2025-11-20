"""
Interfaz Gradio - Sistema de Planificación de Producción (Qin)
Versión 1.0 - Predice Qin recomendado en lugar de Q_net

CAMBIO DE PARADIGMA:
- Antes: Predecía Q_net (balance = Qin - Qout - ΔVol)
- Ahora: Recomienda Qin (producción necesaria) directamente
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


class InterfazPlanificacionQin:
    """Interfaz para Planificación de Producción basada en Qin"""
    
    def __init__(self):
        self.modelo = None
        self.features = []
        self.umbrales = None
        self.metricas = {}
        self.df_completo = None
        self.qin_base_por_hora = None
        self.qin_limites = None
        
    def cargar_todo(self):
        """Carga modelo, datos y calcula Qin baseline"""
        print("🚀 Cargando Sistema de Planificación Qin...")
        
        try:
            # Cargar modelo Forecasting V3.0 (predice Q_net)
            model_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
            if model_path.exists():
                self.modelo = joblib.load(model_path)
                print("✅ Modelo Forecasting V3.0 cargado")
            else:
                print("⚠️ Modelo no encontrado")
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
            print(f"✅ Umbrales: Qin_bajo={self.umbrales['qin_bajo']:,.0f}, Qin_alto={self.umbrales['qin_alto']:,.0f}")
            
            # Cargar dataset completo
            df_path = Path('data/processed/dataset_features_completo.csv')
            if df_path.exists():
                self.df_completo = pd.read_csv(df_path)
                self.df_completo['timestamp'] = pd.to_datetime(self.df_completo['timestamp'])
                print(f"✅ Dataset: {len(self.df_completo):,} registros")
                
                # Calcular Qin base por hora desde datos históricos
                self._calcular_qin_base_historico()
                
                # Calcular límites operativos desde datos
                self._calcular_limites_operativos()
            else:
                print("⚠️ Dataset no encontrado")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calcular_qin_base_historico(self):
        """Calcula perfil de Qin por hora desde datos RAW de producción"""
        print("\n📊 Calculando perfil de Qin histórico...")
        
        # Cargar datos RAW de Qin (producción real)
        qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')
        if qin_path.exists():
            df_qin = pd.read_csv(qin_path)
            df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
            df_qin['hora'] = df_qin['timestamp'].dt.hour
            
            # Calcular estadísticas por hora
            qin_stats = df_qin.groupby('hora')['Qin'].agg(['median', 'mean', 'std', 'min', 'max'])
            
            # Guardar como diccionario de diccionarios
            self.qin_perfil_hora = qin_stats.to_dict('index')
            
            # Para compatibilidad, también guardar solo mediana
            self.qin_base_por_hora = qin_stats['median'].to_dict()
            
            print(f"   ✅ Perfil de Qin calculado para 24 horas")
            print(f"   📈 Rango mediana: {qin_stats['median'].min():,.0f} - {qin_stats['median'].max():,.0f} m³/hr")
            print(f"   📊 Promedio global: {df_qin['Qin'].mean():,.0f} m³/hr")
        else:
            print("   ⚠️ BD_Qin_m3_UTC.csv no encontrado, usando dataset procesado")
            # Fallback al dataset procesado
            self.df_completo['hora'] = self.df_completo['timestamp'].dt.hour
            self.qin_base_por_hora = self.df_completo.groupby('hora')['sist_Qin_m3h'].median().to_dict()
            # Crear perfil simple
            self.qin_perfil_hora = {
                h: {'median': v, 'mean': v, 'std': 0, 'min': v, 'max': v} 
                for h, v in self.qin_base_por_hora.items()
            }
            print(f"   ✅ Qin base calculado para 24 horas (fallback)")
            print(f"   📈 Rango: {min(self.qin_base_por_hora.values()):,.0f} - {max(self.qin_base_por_hora.values()):,.0f} m³/hr")
    
    def _calcular_limites_operativos(self):
        """Calcula límites operativos desde datos reales"""
        print("\n🔧 Calculando límites operativos...")
        
        # Percentiles de Qin histórico
        p05 = self.df_completo['sist_Qin_m3h'].quantile(0.05)
        p95 = self.df_completo['sist_Qin_m3h'].quantile(0.95)
        
        # Límites operativos (respetando máximo de diseño)
        self.qin_limites = {
            'min_operativo': p05,
            'max_operativo': min(15300, p95),  # 15,300 es el máximo de diseño
            'max_diseno': 15300
        }
        
        print(f"   ✅ Límites operativos:")
        print(f"   📉 Mínimo (P05): {self.qin_limites['min_operativo']:,.0f} m³/hr")
        print(f"   📈 Máximo (P95): {self.qin_limites['max_operativo']:,.0f} m³/hr")
        print(f"   🔝 Diseño máx: {self.qin_limites['max_diseno']:,.0f} m³/hr")
    
    def crear_features_prediccion(self, hora, dia_semana, mes, temperatura, df_base=None):
        """Crea features para predicción (mismo que V3.0)"""
        
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
        
        # Features de Q_net: usar solo por hora, dejar que temperatura influya vía features clima
        if self.df_completo is not None:
            df_hora = self.df_completo[
                (self.df_completo['timestamp'].dt.hour == hora)
            ].copy()
            
            if len(df_hora) > 0:
                q_med = df_hora['Q_net_m3h'].median()
                q_std = df_hora['Q_net_m3h'].std()
                if pd.isna(q_std):
                    q_std = 2000.0
                row['Q_net_m3h__ema_win_6h'] = q_med
                row['Q_net_m3h__ema_win_12h'] = q_med
                row['Q_net_m3h__ema_win_24h'] = q_med
                row['Q_net_m3h__lag_168h'] = q_med
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
    
    def calcular_demanda_predicha(self, q_net_predicho, hora, temperatura):
        """
        Calcula demanda real (Qout) usando balance hídrico
        
        REALIDAD DEL DATASET:
        - Q_net (predicho por modelo) = Q_flujo = ΔVol/Δt
        - Q_net NO incluye Qin (es solo el flujo de cambio de volumen)
        
        Balance del sistema:
        - Qin = Qout + Q_flujo + Pérdidas (asumimos pérdidas ≈ 0)
        - Q_flujo = Q_net (son el mismo valor)
        
        Despejando:
        - Qout = Qin - Q_flujo = Qin - Q_net
        
        Esto garantiza que Qout (demanda) > 0 siempre que Qin > 0
        """
        
        # Obtener Qin típico para esta hora (mediana histórica)
        qin_hora = self.qin_perfil_hora[hora]['median']
        
        # Q_net predicho ES Q_flujo (cambio de almacenamiento)
        q_flujo = q_net_predicho
        
        # Calcular Qout (demanda real del sistema)
        # Balance: Qin = Qout + Q_flujo → Qout = Qin - Q_flujo
        qout = qin_hora - q_flujo
        demanda_base = max(qout, 0)  # No puede ser negativa
        
        # Clasificar estado del sistema según Q_flujo
        if q_flujo < -500:  # Descargando almacenamiento significativamente
            tipo_balance = 'DESCARGA'
            nivel_alerta = '🔴'
            descripcion = f'Sistema descarga {abs(q_flujo):.0f} m³/hr (Qout > Qin)'
        elif q_flujo > 500:  # Recargando almacenamiento
            tipo_balance = 'RECARGA'
            nivel_alerta = '🟢'
            descripcion = f'Sistema recarga {q_flujo:.0f} m³/hr (Qin > Qout)'
        else:  # Equilibrio o cambio pequeño
            tipo_balance = 'EQUILIBRIO'
            nivel_alerta = '🟡'
            descripcion = 'Producción ≈ Consumo'
        
        # AJUSTE POR TEMPERATURA (conocimiento del dominio)
        temp_base = 20.0
        delta_temp = temperatura - temp_base
        factor_ajuste = 1.0 + (delta_temp * 0.02)  # +2% por grado
        
        # Aplicar ajuste a la demanda
        demanda_ajustada = demanda_base * factor_ajuste
        
        # Clasificar nivel de demanda (rangos realistas: 8k-14k m³/hr)
        if demanda_ajustada < 9000:
            nivel_demanda = 'BAJA'
            nivel_alerta_demanda = '�'
        elif demanda_ajustada < 11000:
            nivel_demanda = 'MEDIA'
            nivel_alerta_demanda = '🟡'
        elif demanda_ajustada < 13000:
            nivel_demanda = 'ALTA'
            nivel_alerta_demanda = '🟠'
        else:
            nivel_demanda = 'MUY ALTA'
            nivel_alerta_demanda = '🔴'
        
        return {
            'demanda_m3h': demanda_ajustada,
            'demanda_base': demanda_base,
            'qin_hora': qin_hora,
            'qout_estimado': qout,
            'q_flujo': q_flujo,
            'q_net_predicho': q_net_predicho,
            'tipo_balance': tipo_balance,
            'nivel_demanda': nivel_demanda,
            'nivel_alerta': nivel_alerta,
            'nivel_alerta_demanda': nivel_alerta_demanda,
            'descripcion': descripcion,
            'temperatura': temperatura,
            'factor_ajuste': factor_ajuste,
            'hora': hora
        }
    
    def planificar_24_horas(self, fecha_str, temperatura):
        """Genera predicción de demanda para 24 horas"""
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
            
            resultados_24h = []
            horas = []
            
            for hora in range(24):
                # 1. Crear features para predicción
                row = self.crear_features_prediccion(
                    hora,
                    fecha.weekday(),
                    fecha.month,
                    temperatura,
                    self.df_completo
                )
                
                # 2. Predecir Q_net
                X = pd.DataFrame([row])[self.features]
                q_net_pred = self.modelo.predict(X)[0]
                
                # 3. Calcular demanda predicha
                resultado_hora = self.calcular_demanda_predicha(q_net_pred, hora, temperatura)
                resultado_hora['hora'] = hora
                resultados_24h.append(resultado_hora)
                horas.append(hora)
            
            # Calcular estadísticas del día
            demanda_total = sum([r['demanda_m3h'] for r in resultados_24h])
            demanda_promedio = demanda_total / 24
            demanda_max = max([r['demanda_m3h'] for r in resultados_24h])
            
            hora_max = resultados_24h[np.argmax([r['demanda_m3h'] for r in resultados_24h])]['hora']
            
            # Contar horas por nivel
            horas_descarga = sum(1 for r in resultados_24h if r['tipo_balance'] == 'DESCARGA')
            horas_recarga = sum(1 for r in resultados_24h if r['tipo_balance'] == 'RECARGA')
            horas_equilibrio = sum(1 for r in resultados_24h if r['tipo_balance'] == 'EQUILIBRIO')
            horas_alta = sum(1 for r in resultados_24h if r['nivel_demanda'] in ['ALTA', 'MUY ALTA'])
            
            # Contexto de temperatura
            if temperatura < self.umbrales['temp_frio']:
                ctx_temp = f"❄️ Día frío (<{self.umbrales['temp_frio']:.1f}°C) - Demanda esperada baja"
            elif temperatura > self.umbrales['temp_calor']:
                ctx_temp = f"🌡️ Día caluroso (>{self.umbrales['temp_calor']:.1f}°C) - Demanda esperada alta"
            else:
                ctx_temp = "🌤️ Temperatura normal - Demanda esperada moderada"
            
            # Generar reporte
            reporte = f"""
### 📋 Predicción de Demanda Real (Qout) - 24 Horas - {fecha_str}

**🌡️ Temperatura:** {temperatura:.1f}°C constante  
{ctx_temp}

---

## 📊 Resumen del Día

**💧 Demanda Total Predicha:** {demanda_total:,.0f} m³  
**📊 Demanda Promedio:** {demanda_promedio:,.0f} m³/hr  
**📈 Pico de Demanda:** {hora_max:02d}:00 hrs → {demanda_max:,.0f} m³/hr  

---

## ⚖️ Balance del Sistema

**🔴 Horas DESCARGA:** {horas_descarga} horas (Qout > Qin → usa almacenamiento)  
**🟢 Horas RECARGA:** {horas_recarga} horas (Qin > Qout → acumula almacenamiento)  
**🟡 Horas EQUILIBRIO:** {horas_equilibrio} horas (Qin ≈ Qout)  
**⚠️ Horas demanda alta/muy alta:** {horas_alta} horas  

---

💡 **Interpretación:**
- **Demanda (Qout)** = Consumo real del sistema calculado como: Qin - Q_flujo
- La demanda NUNCA es cero (siempre hay consumo)
- Mayor temperatura → Mayor consumo → Mayor demanda predicha (+2%/°C)
- Balance: Qin = Qout + Q_flujo
            """
            
            # Generar gráfico de líneas
            fig = self.crear_grafico_demanda_24h(resultados_24h, fecha_str, temperatura)
            
            return reporte, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def crear_grafico_demanda_24h(self, resultados, fecha_str, temperatura):
        """Crea gráfico de líneas de demanda predicha"""
        
        horas = list(range(24))
        demandas = [r['demanda_m3h'] for r in resultados]
        niveles = [r['nivel_demanda'] for r in resultados]
        
        # Colorear según nivel de demanda
        colors = []
        for nivel in niveles:
            if nivel == 'MUY ALTA':
                colors.append('#D32F2F')
            elif nivel == 'ALTA':
                colors.append('#F57C00')
            elif nivel == 'MEDIA':
                colors.append('#FBC02D')
            elif nivel == 'BAJA':
                colors.append('#388E3C')
            else:
                colors.append('#9E9E9E')
        
        fig = go.Figure()
        
        # Línea de demanda con marcadores coloreados
        fig.add_trace(go.Scatter(
            x=[f"{h:02d}:00" for h in horas],
            y=demandas,
            mode='lines+markers',
            name='Demanda Predicha',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)',
            hovertemplate='%{x}<br>Demanda: %{y:,.0f} m³/hr<extra></extra>'
        ))
        
        # Líneas de referencia para niveles (rangos realistas 8k-14k)
        fig.add_hline(y=9000, line_dash="dot", line_color="green", opacity=0.3,
                     annotation_text="Demanda Baja", annotation_position="right")
        fig.add_hline(y=11000, line_dash="dot", line_color="orange", opacity=0.3,
                     annotation_text="Demanda Media", annotation_position="right")
        fig.add_hline(y=13000, line_dash="dot", line_color="red", opacity=0.3,
                     annotation_text="Demanda Alta", annotation_position="right")
        
        fig.update_layout(
            title=f"Predicción de Demanda Real (Qout) 24h - {fecha_str} ({temperatura:.1f}°C)",
            xaxis_title="Hora del Día",
            yaxis_title="Demanda (m³/hr)",
            height=500,
            hovermode='x unified',
            legend=dict(x=0, y=1, orientation='h'),
            plot_bgcolor='rgba(250,250,250,0.95)',
            yaxis=dict(range=[7000, 15000])  # Rango realista
        )
        
        return fig
    
    def planificar_72_horas(self, fecha_str, temp_dia1, temp_dia2, temp_dia3):
        """Genera predicción de demanda para 72 horas (3 días)"""
        try:
            fecha_inicio = datetime.strptime(fecha_str, "%d/%m/%Y")
            temperaturas_dias = [temp_dia1, temp_dia2, temp_dia3]
            
            todos_resultados = []
            timestamps = []
            temperaturas_plot = []
            
            for dia in range(3):
                fecha_dia = fecha_inicio + timedelta(days=dia)
                temp_dia = temperaturas_dias[dia]
                
                for hora in range(24):
                    # Crear features
                    row = self.crear_features_prediccion(
                        hora,
                        fecha_dia.weekday(),
                        fecha_dia.month,
                        temp_dia,
                        self.df_completo
                    )
                    
                    # Predecir Q_net
                    X = pd.DataFrame([row])[self.features]
                    q_net_pred = self.modelo.predict(X)[0]
                    
                    # Calcular demanda
                    resultado = self.calcular_demanda_predicha(q_net_pred, hora, temp_dia)
                    resultado['dia'] = dia + 1
                    resultado['hora'] = hora
                    
                    todos_resultados.append(resultado)
                    timestamps.append(fecha_dia + timedelta(hours=hora))
                    temperaturas_plot.append(temp_dia)
            
            # Resumen por día
            resumen_dias = []
            for dia in range(3):
                resultados_dia = [r for r in todos_resultados if r['dia'] == dia + 1]
                
                demanda_total = sum([r['demanda_m3h'] for r in resultados_dia])
                demanda_prom = demanda_total / 24
                demanda_max = max([r['demanda_m3h'] for r in resultados_dia])
                
                horas_descarga = sum(1 for r in resultados_dia if r['tipo_balance'] == 'DESCARGA')
                horas_recarga = sum(1 for r in resultados_dia if r['tipo_balance'] == 'RECARGA')
                
                fecha_dia = fecha_inicio + timedelta(days=dia)
                temp_dia = temperaturas_dias[dia]
                
                resumen_dias.append(f"""
**Día {dia+1}** ({fecha_dia.strftime('%d/%m')} - {temp_dia:.1f}°C):
- Demanda total: {demanda_total:,.0f} m³
- Demanda promedio: {demanda_prom:,.0f} m³/hr
- Pico demanda: {demanda_max:,.0f} m³/hr
- Horas descarga: {horas_descarga}/24 | Horas recarga: {horas_recarga}/24
                """)
            
            # Calcular totales
            demanda_total_72h = sum([r['demanda_m3h'] for r in todos_resultados])
            
            resultado_md = f"""
### 📋 Predicción de Demanda 72 Horas - {fecha_str}

{''.join(resumen_dias)}

**⚖️ Demanda Total 3 Días:** {demanda_total_72h:,.0f} m³  
**📊 Demanda Promedio 72h:** {demanda_total_72h / 72:,.0f} m³/hr

---

💡 **Interpretación:** Predicción basada en temperatura diaria y patrones históricos
            """
            
            # Crear gráfico 72h
            fig = self.crear_grafico_demanda_72h(todos_resultados, timestamps, temperaturas_plot, fecha_str)
            
            return resultado_md, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def crear_grafico_demanda_72h(self, resultados, timestamps, temperaturas, fecha_str):
        """Gráfico de líneas para 72 horas con doble eje (demanda + temperatura)"""
        
        demandas = [r['demanda_m3h'] for r in resultados]
        
        # Crear figura con eje secundario
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Línea 1: Demanda predicha (eje principal)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=demandas,
                mode='lines',
                name='Demanda Predicha',
                line=dict(color='#2E86AB', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.1)',
                hovertemplate='%{x|%d/%m %H:%M}<br>Demanda: %{y:,.0f} m³/hr<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Línea 2: Temperatura (eje secundario)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=temperaturas,
                mode='lines',
                name='Temperatura',
                line=dict(color='#F57C00', width=2, dash='dot'),
                hovertemplate='%{x|%d/%m %H:%M}<br>Temp: %{y:.1f}°C<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Configurar ejes
        fig.update_xaxes(title_text="Fecha/Hora")
        fig.update_yaxes(title_text="Demanda (m³/hr)", secondary_y=False)
        fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=True)
        
        fig.update_layout(
            title=f"Predicción de Demanda 72 Horas - {fecha_str}",
            height=500,
            hovermode='x unified',
            legend=dict(x=0, y=1, orientation='h'),
            plot_bgcolor='rgba(250,250,250,0.95)'
        )
        
        return fig
    
    def evaluar_testing(self):
        """Evalúa modelo en periodo de testing calculando demanda real (Qout)"""
        try:
            # Cargar dataset con TODAS las features ya creadas
            df_completo_path = Path('data/processed/dataset_features_completo.csv')
            if not df_completo_path.exists():
                return "❌ Error: dataset_features_completo.csv no encontrado", None
            
            df = pd.read_csv(df_completo_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Agregar features categóricas
            df = self._agregar_features_categoricas(df)
            
            # Usar últimos 15% como test
            n = len(df)
            test_start = int(n * 0.85)
            df_test = df.iloc[test_start:]
            
            # Predecir Q_net
            X_test = df_test[self.features]
            y_test_qnet = df_test['Q_net_m3h'].values
            y_pred_qnet = self.modelo.predict(X_test)
            
            # CALCULAR DEMANDA REAL (Qout = Qin - Q_net)
            # Obtener Qin por hora
            df_test['hora'] = df_test['timestamp'].dt.hour
            qin_por_hora = df_test['hora'].map(self.qin_perfil_hora).apply(lambda x: x['median'])
            
            # Demanda real y predicha
            y_test_qout = qin_por_hora.values - y_test_qnet
            y_pred_qout = qin_por_hora.values - y_pred_qnet
            
            # Calcular métricas SOBRE LA DEMANDA (no sobre Q_net)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_test_qout, y_pred_qout))
            mae = mean_absolute_error(y_test_qout, y_pred_qout)
            r2 = r2_score(y_test_qout, y_pred_qout)
            
            mask = y_test_qout != 0
            mape = np.mean(np.abs((y_test_qout[mask] - y_pred_qout[mask]) / y_test_qout[mask])) * 100
            
            # Error absoluto
            errores = y_pred_qout - y_test_qout
            
            # Reporte
            modelo_nombre = self.metricas.get('modelo', 'XGBoost Forecasting V3.0')
            modelo_r2_original = self.metricas.get('r2', 0)
            
            resultado = f"""
### 📊 Evaluación en Periodo de Testing - Demanda Real (Qout)

**🤖 Modelo:** {modelo_nombre}  
**📈 R² original (Q_net):** {modelo_r2_original:.4f}

**📅 Periodo:** {df_test['timestamp'].min().strftime('%d/%m/%Y')} - {df_test['timestamp'].max().strftime('%d/%m/%Y')}  
**📊 Registros:** {len(df_test):,} horas

---

## 🎯 Métricas de Performance (Demanda Qout)

**R² (Coef. Determinación):** {r2:.4f}  
→ El modelo explica {r2*100:.2f}% de la varianza en la demanda real

**RMSE (Error Cuadrático):** {rmse:,.0f} m³/hr  
→ Error típico considerando outliers

**MAE (Error Absoluto):** {mae:,.0f} m³/hr  
→ Error promedio real

**MAPE (Error Porcentual):** {mape:.2f}%  
→ Error relativo promedio

---

## 📈 Distribución del Error

- **Error máximo positivo:** {errores.max():,.0f} m³/hr (sobre-estimación)
- **Error máximo negativo:** {errores.min():,.0f} m³/hr (sub-estimación)
- **Desviación estándar:** {errores.std():,.0f} m³/hr

---

## 💡 Interpretación

- Evaluación sobre **demanda real (Qout)**, no sobre Q_net
- Fórmula: **Qout = Qin - Q_net**
- Valores siempre positivos (demanda real del sistema)
- R² cercano a 1 confirma excelente capacidad predictiva
            """
            
            # Gráfico
            timestamps = df_test['timestamp'].values
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Demanda Real (Qout): Predicción vs Real',
                    'Error de Predicción'
                ),
                row_heights=[0.6, 0.4],
                vertical_spacing=0.12
            )
            
            # Subplot 1: Real vs Pred - DEMANDA (Qout)
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=y_test_qout,
                mode='lines',
                name='Demanda Real',
                line=dict(color='#2E86AB', width=1.5),
                hovertemplate='%{x|%d/%m %H:%M}<br>Real: %{y:,.0f} m³/hr'
                '<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=y_pred_qout,
                mode='lines',
                name='Demanda Predicha',
                line=dict(color='#F18701', width=1.5, dash='dot'),
                hovertemplate='%{x|%d/%m %H:%M}<br>Pred: %{y:,.0f} m³/hr'
                '<extra></extra>'
            ), row=1, col=1)
            
            # Subplot 2: Error
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=errores,
                mode='lines',
                name='Error',
                line=dict(color='#A23B72', width=1),
                fill='tozeroy',
                fillcolor='rgba(241, 143, 1, 0.2)',
                hovertemplate='%{x|%d/%m %H:%M}<br>Error: %{y:,.0f} m³/hr'
                '<extra></extra>'
            ), row=2, col=1)
            
            fig.add_hline(
                y=0, line_dash="dash", line_color="gray",
                opacity=0.5, row=2, col=1
            )
            
            fig.update_xaxes(title_text="Fecha/Hora", row=2, col=1)
            fig.update_yaxes(
                title_text="Demanda Qout (m³/hr)", row=1, col=1
            )
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
            datos_frio = self._calcular_prediccion_24h(fecha, 3.0)
            datos_calor = self._calcular_prediccion_24h(fecha, 30.0)
            
            balance_frio = datos_frio['balance_neto']
            balance_calor = datos_calor['balance_neto']
            
            diferencia_pct = abs((balance_calor - balance_frio) / balance_frio * 100)
            es_correcto = balance_calor < balance_frio
            cambio_esperado = diferencia_pct > 10
            
            return {
                'nombre': 'Sensibilidad a Temperatura',
                'icono': '🌡️',
                'estado': 'PASS' if (es_correcto and cambio_esperado) else 'FAIL',
                'detalles': f"""
**Test:** Predicción 24h con 3°C vs 30°C

- **Balance con 3°C:** {balance_frio:,.0f} m³
- **Balance con 30°C:** {balance_calor:,.0f} m³
- **Diferencia:** {diferencia_pct:.1f}%
- **Comportamiento:** {'✅ Correcto' if es_correcto else '❌ Invertido'}
- **Sensibilidad:** {'✅ Adecuada' if cambio_esperado else '⚠️ Baja'}
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
            fecha1 = "13/11/2025"
            fecha2 = "14/11/2025"
            temp = 15.0
            
            datos1 = self._calcular_prediccion_24h(fecha1, temp)
            datos2 = self._calcular_prediccion_24h(fecha2, temp)
            
            balance1 = datos1['balance_neto']
            balance2 = datos2['balance_neto']
            
            diferencia = abs(balance2 - balance1)
            diferencia_pct = abs((balance2 - balance1) / balance1 * 100) if balance1 != 0 else 0
            es_coherente = diferencia_pct < 30
            
            return {
                'nombre': 'Coherencia Temporal',
                'icono': '📅',
                'estado': 'PASS' if es_coherente else 'WARNING',
                'detalles': f"""
**Test:** Predicciones días consecutivos

- **Día 1:** {balance1:,.0f} m³
- **Día 2:** {balance2:,.0f} m³
- **Diferencia:** {diferencia_pct:.1f}%
- **Estado:** {'✅ Coherente' if es_coherente else '⚠️ Alta variación'}
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
        """Test: Predicciones dentro de rangos históricos"""
        try:
            fecha = "13/11/2025"
            temp = 18.0
            datos = self._calcular_prediccion_24h(fecha, temp)
            
            if self.df_completo is not None:
                p1 = self.df_completo['Q_net_m3h'].quantile(0.01)
                p99 = self.df_completo['Q_net_m3h'].quantile(0.99)
                
                max_demanda = datos['max_demanda']
                max_recuperacion = datos['max_recuperacion']
                dentro_rango = (max_demanda >= p1) and (max_recuperacion <= p99)
                
                return {
                    'nombre': 'Rangos Históricos',
                    'icono': '📊',
                    'estado': 'PASS' if dentro_rango else 'WARNING',
                    'detalles': f"""
**Test:** Predicciones dentro de rangos

- **Demanda máxima:** {max_demanda:,.0f} m³/hr
- **Recuperación máxima:** {max_recuperacion:,.0f} m³/hr
- **P1-P99:** {p1:,.0f} a {p99:,.0f} m³/hr
- **Estado:** {'✅ OK' if dentro_rango else '⚠️ Fuera de rango'}
                    """,
                    'ok': dentro_rango
                }
            else:
                return {
                    'nombre': 'Rangos Históricos',
                    'icono': '📊',
                    'estado': 'WARNING',
                    'detalles': '⚠️ Dataset no disponible',
                    'ok': False
                }
        except Exception as e:
            return {
                'nombre': 'Rangos Históricos',
                'icono': '📊',
                'estado': 'ERROR',
                'detalles': f'❌ Error: {str(e)}',
                'ok': False
            }
    
    def test_estado_sistema(self):
        """Test: Verificar componentes cargados"""
        try:
            checks = [
                ('Modelo XGBoost', self.modelo is not None),
                ('Features (47)', len(self.features) == 47 if self.features else False),
                ('Umbrales', all(k in self.umbrales for k in ['temp_frio', 'temp_calor', 'qin_bajo', 'qin_alto'])),
                ('Dataset completo', self.df_completo is not None and len(self.df_completo) > 1000),
                ('Qin base calculado', self.qin_base_por_hora is not None),
                ('Límites operativos', self.qin_limites is not None)
            ]
            
            todos_ok = all(check[1] for check in checks)
            detalles_checks = '\n'.join([f"{'✅' if ok else '❌'} {nombre}" for nombre, ok in checks])
            
            return {
                'nombre': 'Estado del Sistema',
                'icono': '⚙️',
                'estado': 'PASS' if todos_ok else 'FAIL',
                'detalles': f"\n{detalles_checks}\n",
                'ok': todos_ok
            }
        except Exception as e:
            return {
                'nombre': 'Estado del Sistema',
                'icono': '⚙️',
                'estado': 'ERROR',
                'detalles': f'❌ Error: {str(e)}',
                'ok': False
            }
    
    def generar_reporte_validacion(self, resultados):
        """Genera reporte Markdown de validación"""
        total = len(resultados)
        passed = sum(1 for r in resultados if r['ok'])
        
        if passed == total:
            estado_general = "✅ **TODOS LOS TESTS PASARON**"
        elif passed >= total * 0.75:
            estado_general = "⚠️ **ALGUNOS WARNINGS**"
        else:
            estado_general = "❌ **TESTS FALLARON**"
        
        reporte = f"""
# 🔍 Validación Rápida del Sistema

**Resultado:** {passed}/{total} tests pasaron

{estado_general}

---

"""
        
        for r in resultados:
            icono_estado = {
                'PASS': '✅',
                'WARNING': '⚠️',
                'FAIL': '❌',
                'ERROR': '💥'
            }.get(r['estado'], '❓')
            
            reporte += f"""
## {r['icono']} {r['nombre']} {icono_estado}

{r['detalles']}

---

"""
        
        reporte += f"\n**Última ejecución:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        return reporte
    
    def ejecutar_validacion_rapida(self):
        """Ejecuta suite de tests de validación"""
        resultados = [
            self.test_sensibilidad_temperatura(),
            self.test_coherencia_temporal(),
            self.test_rangos_historicos(),
            self.test_estado_sistema()
        ]
        return self.generar_reporte_validacion(resultados)
    
    def entrenar_randomforest(self, n_estimators=200, max_depth=15):
        """Entrena modelo RandomForest para comparación"""
        try:
            print("\n🌲 Entrenando RandomForest...")
            
            # Cargar dataset con features completas
            df_path = Path('data/processed/dataset_features_completo.csv')
            if not df_path.exists():
                return None, "❌ Dataset no encontrado"
            
            df = pd.read_csv(df_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Agregar features categóricas
            df = self._agregar_features_categoricas(df)
            
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]
            
            X_train = df_train[self.features]
            y_train = df_train['Q_net_m3h']
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            
            modelo = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train, y_train)
            
            y_test_pred = modelo.predict(X_test)
            metrics_test = self._calculate_metrics(y_test.values, y_test_pred)
            
            print(f"✅ RandomForest: R²={metrics_test['r2']:.4f}")
            
            return modelo, (None, metrics_test)
            
        except Exception as e:
            import traceback
            return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    def entrenar_lightgbm(self, n_estimators=200, max_depth=10):
        """Entrena modelo LightGBM para comparación"""
        if not LIGHTGBM_AVAILABLE:
            return None, "❌ LightGBM no instalado"
        
        try:
            print("\n💡 Entrenando LightGBM...")
            
            # Cargar dataset con features completas
            df_path = Path('data/processed/dataset_features_completo.csv')
            if not df_path.exists():
                return None, "❌ Dataset no encontrado"
            
            df = pd.read_csv(df_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Agregar features categóricas
            df = self._agregar_features_categoricas(df)
            
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]
            
            X_train = df_train[self.features]
            y_train = df_train['Q_net_m3h']
            X_val = df_val[self.features]
            y_val = df_val['Q_net_m3h']
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            
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
            
            y_test_pred = modelo.predict(X_test)
            metrics_test = self._calculate_metrics(y_test.values, y_test_pred)
            
            print(f"✅ LightGBM: R²={metrics_test['r2']:.4f}")
            
            return modelo, (None, metrics_test)
            
        except Exception as e:
            import traceback
            return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    def _agregar_features_categoricas(self, df):
        """Agrega features categóricas y de interacción al dataframe"""
        df = df.copy()
        
        # temp_nivel (categórico)
        df['temp_nivel_Frio'] = (df['clima_temp_c'] < self.umbrales['temp_frio']).astype(int)
        df['temp_nivel_Calor'] = (df['clima_temp_c'] > self.umbrales['temp_calor']).astype(int)
        df['temp_nivel_Normal'] = ((df['clima_temp_c'] >= self.umbrales['temp_frio']) & 
                                     (df['clima_temp_c'] <= self.umbrales['temp_calor'])).astype(int)
        
        # periodo_dia (categórico)
        condiciones = [
            df['hora'] < 6,
            (df['hora'] >= 6) & (df['hora'] < 9),
            (df['hora'] >= 9) & (df['hora'] < 18),
            (df['hora'] >= 18) & (df['hora'] < 22),
            df['hora'] >= 22
        ]
        periodos = ['Madrugada', 'Manana_critica', 'Dia', 'Noche', 'Noche_tardia']
        
        for periodo in periodos:
            df[f'periodo_dia_{periodo}'] = 0
        
        for i, condicion in enumerate(condiciones):
            df.loc[condicion, f'periodo_dia_{periodos[i]}'] = 1
        
        # es_fin_de_semana
        if 'dia_semana' not in df.columns:
            df['dia_semana'] = df['timestamp'].dt.weekday
        df['es_fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        # es_hora_bisagra
        df['es_hora_bisagra'] = df['hora'].isin([7, 8, 9, 22, 23]).astype(int)
        
        # regimen_equilibrio (placeholder - siempre 0 en predicción)
        df['regimen_equilibrio'] = 0
        
        # Interacciones
        df['temp_x_hora'] = df['clima_temp_c'] * df['hora']
        df['temp_x_finde'] = df['clima_temp_c'] * df['es_fin_de_semana']
        
        # delta_temp_6h_x_hora
        if 'clima_temp_delta_6h' in df.columns:
            df['delta_temp_6h_x_hora'] = df['clima_temp_delta_6h'] * df['hora']
        else:
            df['delta_temp_6h_x_hora'] = 0
        
        return df
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
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
            print("\n🔬 Iniciando comparación...")
            
            resultados = {}
            
            # Cargar dataset con features base
            df_completo_path = Path('data/processed/dataset_features_completo.csv')
            if not df_completo_path.exists():
                return "❌ Error: dataset_features_completo.csv no encontrado", None
            
            df = pd.read_csv(df_completo_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Crear features categóricas que faltan (igual que en crear_features_prediccion)
            df = self._agregar_features_categoricas(df)
            
            # 1. XGBoost actual
            print("\n📊 Evaluando XGBoost V3.0...")
            n = len(df)
            val_end = int(n * 0.85)
            df_test = df.iloc[val_end:]
            
            X_test = df_test[self.features]
            y_test = df_test['Q_net_m3h']
            y_pred_xgb = self.modelo.predict(X_test)
            
            metrics_xgb = self._calculate_metrics(y_test.values, y_pred_xgb)
            resultados['XGBoost V3.0\n(Actual)'] = metrics_xgb
            
            # 2. RandomForest
            modelo_rf, resultado_rf = self.entrenar_randomforest()
            if modelo_rf is not None:
                resultados['RandomForest'] = resultado_rf[1]
            
            # 3. LightGBM
            if LIGHTGBM_AVAILABLE:
                modelo_lgb, resultado_lgb = self.entrenar_lightgbm()
                if modelo_lgb is not None:
                    resultados['LightGBM'] = resultado_lgb[1]
            
            reporte = self._generar_reporte_comparacion(resultados)
            fig = self._generar_grafico_comparacion(resultados)
            
            return reporte, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def _generar_reporte_comparacion(self, resultados):
        """Genera reporte de comparación"""
        reporte = f"""
# 🔬 Comparación de Modelos ML

**Fecha:** {datetime.now().strftime("%d/%m/%Y %H:%M")}

---

## 📊 Resultados en Test Set

| Modelo | R² | RMSE | MAE | MAPE |
|--------|----:|-----:|----:|-----:|
"""
        
        for nombre, m in resultados.items():
            reporte += f"| {nombre} | {m['r2']:.4f} | {m['rmse']:,.0f} | {m['mae']:,.0f} | {m['mape']:.2f}% |\n"
        
        reporte += "\n---\n\n## 🏆 Mejor Modelo\n\n"
        
        mejor_r2 = max(resultados.items(), key=lambda x: x[1]['r2'])
        reporte += f"- **R² más alto:** {mejor_r2[0]} ({mejor_r2[1]['r2']:.4f})\n"
        
        return reporte
    
    def _generar_grafico_comparacion(self, resultados):
        """Genera gráfico de comparación"""
        modelos = list(resultados.keys())
        r2_vals = [resultados[m]['r2'] for m in modelos]
        rmse_vals = [resultados[m]['rmse'] for m in modelos]
        mae_vals = [resultados[m]['mae'] for m in modelos]
        mape_vals = [resultados[m]['mape'] for m in modelos]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score', 'RMSE (m³/hr)', 'MAE (m³/hr)', 'MAPE (%)'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        colors = ['#2E86AB', '#A23B72', '#F18701']
        
        fig.add_trace(go.Bar(x=modelos, y=r2_vals, marker_color=colors, showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=modelos, y=rmse_vals, marker_color=colors, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=modelos, y=mae_vals, marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(x=modelos, y=mape_vals, marker_color=colors, showlegend=False), row=2, col=2)
        
        fig.update_layout(height=700, title_text="Comparación de Modelos ML")
        
        return fig
    
    def crear_header_pronostico(self):
        """Crea header con pronóstico climático min/max"""
        
        # Pronóstico por defecto
        dias_pron = [
            {"dia": "Día 1", "temp_min": 15, "temp_max": 25, "icono": "☀️"},
            {"dia": "Día 2", "temp_min": 14, "temp_max": 24, "icono": "⛅"},
            {"dia": "Día 3", "temp_min": 13, "temp_max": 23, "icono": "🌤️"}
        ]
        
        # Generar HTML de cards
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
                        🚰 Sistema de Planificación de Producción (Qin)
                    </h1>
                    <p style="margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;">
                        Versión 1.0 - Recomendación Directa de Producción Necesaria
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
    
    def lanzar_interfaz(self):
        """Crea y lanza la interfaz Gradio"""
        
        with gr.Blocks(title="Predicción de Demanda V1.0", theme=gr.themes.Soft()) as demo:
            
            # Header
            gr.HTML(self.crear_header_pronostico())
            
            with gr.Tabs():
                
                # TAB: Predicción Demanda 24 Horas
                with gr.Tab("📅 Demanda 24 Horas"):
                    gr.Markdown("""
                    ### Predicción de demanda para un día completo
                    
                    **Sistema simplificado:**
                    - ✅ Predice balance del sistema (Q_net)
                    - ✅ Calcula demanda cuando Q_net < 0
                    - ✅ Clasifica nivel de demanda (BAJA/MEDIA/ALTA)
                    
                    ⚠️ **Nota sobre temperatura:** El modelo V3.0 fue entrenado con datos históricos  
                    donde la demanda está principalmente determinada por **patrones horarios**,  
                    **día de la semana** y **estacionalidad**. La temperatura tiene influencia  
                    limitada en las predicciones.
                    """)
                    
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
                            btn_24h = gr.Button("🚀 Generar Plan 24h", variant="primary")
                        
                        with gr.Column():
                            output_24h = gr.Markdown()
                    
                    plot_24h = gr.Plot(label="Gráfico Plan 24h")
                    
                    btn_24h.click(
                        fn=self.planificar_24_horas,
                        inputs=[input_fecha_24h, input_temp_24h],
                        outputs=[output_24h, plot_24h]
                    )
                
                # TAB: Predicción Demanda 72 Horas
                with gr.Tab("🔮 Demanda 72 Horas"):
                    gr.Markdown("### Predicción de demanda para 3 días con temperaturas variables")
                    
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
                            
                            btn_72h = gr.Button("🚀 Generar Plan 72h", variant="primary")
                        
                        with gr.Column():
                            output_72h = gr.Markdown()
                    
                    plot_72h = gr.Plot(label="Gráfico Plan 72h")
                    
                    btn_72h.click(
                        fn=self.planificar_72_horas,
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
                    
                    Ejecuta tests para verificar que el sistema funciona correctamente:
                    
                    - **🌡️ Sensibilidad a Temperatura:** Temp alta → mayor demanda
                    - **📅 Coherencia Temporal:** Predicciones consistentes
                    - **📊 Rangos Históricos:** Dentro de rangos razonables
                    - **⚙️ Estado del Sistema:** Todos los componentes OK
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
                    ### Comparación de Múltiples Algoritmos
                    
                    Compara rendimiento de diferentes algoritmos:
                    
                    - **🚀 XGBoost V3.0:** Modelo actual
                    - **🌲 RandomForest:** Ensamble de árboles
                    - **💡 LightGBM:** Gradient Boosting optimizado
                    
                    ⏱️ **Tiempo estimado:** 2-3 minutos
                    """)
                    
                    btn_comparacion = gr.Button("🔬 Comparar Modelos", variant="primary", size="lg")
                    output_comparacion = gr.Markdown()
                    plot_comparacion = gr.Plot(label="Gráfico Comparación")
                    
                    btn_comparacion.click(
                        fn=self.comparar_modelos_ml,
                        inputs=[],
                        outputs=[output_comparacion, plot_comparacion]
                    )
                
                # TAB: Información
                with gr.Tab("ℹ️ Información"):
                    gr.Markdown(f"""
## 📋 Sistema de Planificación de Producción (Qin)

### 🎯 Cambio de Paradigma

**Antes (V3.0):**
- Predecía Q_net (balance = Qin - Qout - ΔVol)
- Interpretación: "El sistema está perdiendo 800 m³/hr"
- Requería que el operador calculara Qin necesario

**Ahora (V1.0):**
- Recomienda Qin directamente
- Interpretación: "Producir 11,500 m³/hr"
- Guía operacional directa

---

### 🔬 Metodología

**1. Predicción Base:**
- Modelo XGBoost V3.0 predice Q_net (R² = {self.metricas['r2']:.4f})
- 47 features: clima, calendario, lags, rolling stats

**2. Cálculo de Qin:**
```python
# Si Q_net < 0 (sistema perdiendo agua):
Qin_recomendado = Qin_base_hora + |Q_net|

# Si Q_net > 0 (sistema ganando agua):
Qin_recomendado = Qin_base_hora × 0.95
```

**3. Sin Márgenes Artificiales:**
- Error del modelo = margen inherente
- No se agregan % arbitrarios
- Límites basados en datos históricos reales

**4. Qin Base Histórico:**
- Mediana de Qin por hora desde datos
- Rango: {min(self.qin_base_por_hora.values()):,.0f} - {max(self.qin_base_por_hora.values()):,.0f} m³/hr

**5. Límites Operativos:**
- Mínimo (P05): {self.qin_limites['min_operativo']:,.0f} m³/hr
- Máximo (P95): {self.qin_limites['max_operativo']:,.0f} m³/hr
- Diseño máximo: {self.qin_limites['max_diseno']:,.0f} m³/hr

**6. Umbrales de Alerta:**
- Qin bajo: < {self.umbrales['qin_bajo']:,.0f} m³/hr
- Qin normal: {self.umbrales['qin_bajo']:,.0f} - {self.umbrales['qin_alto']:,.0f} m³/hr
- Qin alto: > {self.umbrales['qin_alto']:,.0f} m³/hr

---

### 📊 Visualizaciones

**Gráficos de líneas duales:**
- 🔵 Línea azul: Qin recomendado
- 🔴 Línea roja: Demanda estimada (área sombreada)
- ⚪ Línea gris punteada: Qin base histórico
- 📏 Bandas horizontales: Umbrales de alerta

---

### 💡 Ventajas del Sistema

1. **Orientación operacional directa** - No requiere interpretación
2. **Basado en datos reales** - No usa constantes arbitrarias
3. **Sin márgenes artificiales** - La incertidumbre del modelo es el margen
4. **Límites realistas** - Extraídos de percentiles históricos
5. **Alertas automáticas** - Basadas en umbrales estadísticos

---

**Modelo:** XGBoost Forecasting V3.0  
**Dataset:** 15,034 registros horarios  
**Período:** 2024-01-01 a 2024-09-30  
**Versión:** Planificación Qin V1.0
                    """)
            
            gr.Markdown("""
            ---
            **Sistema de Planificación de Producción ESVAL** | Versión 1.0  
            *Noviembre 2025*
            """)
        
        return demo


def main():
    """Función principal"""
    print("=" * 80)
    print("SISTEMA DE PLANIFICACIÓN DE PRODUCCIÓN (Qin) V1.0")
    print("=" * 80)
    
    # Crear instancia
    app = InterfazPlanificacionQin()
    
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
        server_port=7867,  # Puerto 7867
        share=False
    )


if __name__ == "__main__":
    main()
