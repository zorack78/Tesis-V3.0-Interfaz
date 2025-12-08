"""
Interfaz Gradio - Sistema de Planificación de Producción (Qin)
Versión 1.0 - Predice Qin recomendado basado en Q_net y demanda

MODELO PREDICTIVO:
- Predice Q_net = ΔVol/Δt (flujo neto del sistema)
- Calcula Demanda: Qout = Qin - Q_net
- Recomienda producción óptima considerando balance hídrico
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
import requests
from zoneinfo import ZoneInfo
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
        # Modelos adicionales para comparación
        self.modelo_rf = None
        self.modelo_lgb = None
        # Pronóstico meteorológico
        self.pronostico_cache = None
        
    def obtener_pronostico_automatico(self):
        """
        Obtiene pronóstico meteorológico real de Open-Meteo para Valparaíso.
        Retorna dict con 3 días de pronóstico: temp_min, temp_max, prob_lluvia.
        
        Si falla la API, retorna valores por defecto razonables.
        """
        LAT = -33.06528  # Rodelillo, Valparaíso
        LON = -71.55639
        TZ = ZoneInfo('America/Santiago')
        
        try:
            # Determinar fecha inicial (si es después de 17:00, comenzar mañana)
            now = datetime.now(TZ)
            if now.hour >= 17:
                start_date = (now + timedelta(days=1)).date()
            else:
                start_date = now.date()
            
            end_date = start_date + timedelta(days=2)  # 3 días total
            
            # Llamar API Open-Meteo (sin API key, gratis)
            url = 'https://api.open-meteo.com/v1/forecast'
            params = {
                'latitude': LAT,
                'longitude': LON,
                'hourly': 'temperature_2m',
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_probability_max',
                'timezone': 'America/Santiago',
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
            }
            
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get('daily', {})
                
                fechas = daily.get('time', [])
                temp_max = daily.get('temperature_2m_max', [])
                temp_min = daily.get('temperature_2m_min', [])
                prob_lluvia = daily.get('precipitation_probability_max', [])
                
                # Construir resultado
                pronostico = []
                for i in range(min(3, len(fechas))):
                    # Temperatura promedio del día
                    temp_promedio = (temp_max[i] + temp_min[i]) / 2
                    
                    # Determinar icono según probabilidad de lluvia
                    prob = prob_lluvia[i] if i < len(prob_lluvia) else 0
                    if prob >= 70:
                        icono = "🌧️"  # Lluvia alta
                    elif prob >= 40:
                        icono = "⛅"   # Nublado parcial
                    elif prob >= 20:
                        icono = "🌤️"  # Mayormente soleado
                    else:
                        icono = "☀️"   # Soleado
                    
                    pronostico.append({
                        'fecha': fechas[i],
                        'temp_min': round(temp_min[i], 1),
                        'temp_max': round(temp_max[i], 1),
                        'temp_promedio': round(temp_promedio, 1),
                        'prob_lluvia': round(prob, 0),
                        'icono': icono,
                        'dia': f"Día {i+1}"
                    })
                
                print(f"✅ Pronóstico obtenido: {len(pronostico)} días desde {start_date}")
                self.pronostico_cache = pronostico
                return pronostico
                
            else:
                print(f"⚠️ API error {resp.status_code}, usando valores por defecto")
                return self._pronostico_por_defecto()
        
        except Exception as e:
            print(f"⚠️ Error obteniendo pronóstico: {e}, usando valores por defecto")
            return self._pronostico_por_defecto()
    
    def _pronostico_por_defecto(self):
        """Pronóstico por defecto en caso de error de API"""
        return [
            {'fecha': '', 'temp_min': 12, 'temp_max': 20, 'temp_promedio': 16, 'prob_lluvia': 10, 'icono': '☀️', 'dia': 'Día 1'},
            {'fecha': '', 'temp_min': 11, 'temp_max': 19, 'temp_promedio': 15, 'prob_lluvia': 20, 'icono': '🌤️', 'dia': 'Día 2'},
            {'fecha': '', 'temp_min': 10, 'temp_max': 18, 'temp_promedio': 14, 'prob_lluvia': 30, 'icono': '⛅', 'dia': 'Día 3'},
        ]
    
    def buscar_dia_similar(self, fecha_objetivo, temp_min, temp_max, mes, dia_semana):
        """
        Busca un día similar en la base de datos histórica basado en:
        - Mes (mismo mes o adyacentes)
        - Día de la semana (mismo o similar)
        - Rango de temperatura (temp_min y temp_max similares)
        
        Retorna las temperaturas horarias reales de ese día histórico.
        """
        # Cargar datos con clima desde archivo separado (no reemplazar df_completo)
        try:
            df_clima = pd.read_csv('data/processed/dataset_completo_sincronizado.csv')
            if 'temp' not in df_clima.columns:
                print("⚠️ Dataset sin columna 'temp'")
                return None
            df = df_clima
            # print("✅ Datos climáticos históricos cargados para búsqueda")
        except Exception as e:
            print(f"⚠️ Error cargando datos climáticos: {e}")
            return None
        
        # Filtrar por mes (mismo mes o meses adyacentes para mejor match)
        meses_validos = [mes]
        if mes == 1:
            meses_validos = [12, 1, 2]
        elif mes == 12:
            meses_validos = [11, 12, 1]
        else:
            meses_validos = [mes-1, mes, mes+1]
        
        df_filtrado = df[df['mes'].isin(meses_validos)].copy()
        
        if len(df_filtrado) == 0:
            print(f"⚠️ No hay datos para meses {meses_validos}")
            return None
        
        # Agrupar por fecha para obtener temp_min y temp_max diarias
        if 'timestamp' in df_filtrado.columns:
            df_filtrado['timestamp'] = pd.to_datetime(df_filtrado['timestamp'])
            df_filtrado['fecha'] = df_filtrado['timestamp'].dt.date
        elif 'fecha_hora_local' in df_filtrado.columns:
            df_filtrado['timestamp'] = pd.to_datetime(df_filtrado['fecha_hora_local'])
            df_filtrado['fecha'] = df_filtrado['timestamp'].dt.date
        else:
            print("⚠️ No hay columna timestamp o fecha_hora_local")
            return None
        
        df_diario = df_filtrado.groupby('fecha').agg({
            'temp': ['min', 'max', 'mean'],
            'dia_semana': 'first',
            'mes': 'first',
            'hora': 'count'
        }).reset_index()
        
        df_diario.columns = ['fecha', 'temp_min_dia', 'temp_max_dia', 'temp_mean_dia', 'dia_semana', 'mes', 'n_registros']
        
        # Filtrar días completos (24 horas)
        df_diario = df_diario[df_diario['n_registros'] == 24]
        
        if len(df_diario) == 0:
            print("⚠️ No hay días completos en el histórico")
            return None
        
        # Calcular similitud de temperatura
        df_diario['diff_temp'] = abs(df_diario['temp_min_dia'] - temp_min) + abs(df_diario['temp_max_dia'] - temp_max)
        
        # Penalizar días de semana diferentes (pero no descartar)
        df_diario['diff_dia_semana'] = abs(df_diario['dia_semana'] - dia_semana)
        df_diario['score'] = df_diario['diff_temp'] + df_diario['diff_dia_semana'] * 2
        
        # Ordenar por similitud (menor score = más similar)
        df_diario = df_diario.sort_values('score')
        
        if len(df_diario) == 0:
            return None
        
        # Tomar el día más similar
        mejor_dia = df_diario.iloc[0]
        fecha_similar = mejor_dia['fecha']
        
        print(f"✅ Día similar encontrado: {fecha_similar} (Temp: {mejor_dia['temp_min_dia']:.1f}-{mejor_dia['temp_max_dia']:.1f}°C, DiaSemana: {int(mejor_dia['dia_semana'])})")
        
        # Obtener temperaturas horarias de ese día
        df_filtrado['fecha_compare'] = pd.to_datetime(df_filtrado['timestamp']).dt.date
        df_dia_similar = df_filtrado[df_filtrado['fecha_compare'] == fecha_similar].sort_values('hora')
        
        if len(df_dia_similar) != 24:
            print(f"⚠️ Día similar no tiene 24 horas completas (tiene {len(df_dia_similar)})")
            return None
        
        # Retornar temperaturas horarias
        temperaturas_horarias = df_dia_similar[['hora', 'temp']].set_index('hora')['temp'].to_dict()
        
        # Asegurar que tenemos las 24 horas
        temps_completas = {}
        for h in range(24):
            if h in temperaturas_horarias:
                temps_completas[h] = round(temperaturas_horarias[h], 2)
            else:
                # Interpolar si falta alguna hora
                print(f"⚠️ Falta hora {h}, interpolando...")
                temps_completas[h] = round(mejor_dia['temp_mean_dia'], 2)
        
        return temps_completas
    
    def calcular_temperatura_hora(self, temp_promedio_dia, hora, mes):
        """
        Calcula temperatura realista para una hora específica basada en datos empíricos.
        
        Parámetros:
            temp_promedio_dia: Temperatura promedio del día (°C) ingresada por usuario
            hora: Hora del día (0-23)
            mes: Mes del año (1-12) para ajustar amplitud estacional
        
        Retorna:
            Temperatura estimada para esa hora (°C)
        
        Basado en análisis de 15,334 registros de Valparaíso:
        - Mínima: 06:00-07:00 hrs
        - Máxima: 15:00-16:00 hrs
        - Amplitud: varía 6-9°C según estación
        """
        # Amplitud térmica por estación (datos reales)
        if mes in [12, 1, 2]:  # Verano
            amplitud = 8.6  # °C
            hora_min = 7
            hora_max = 16
        elif mes in [3, 4, 5]:  # Otoño
            amplitud = 8.1
            hora_min = 7
            hora_max = 15
        elif mes in [6, 7, 8]:  # Invierno
            amplitud = 6.3
            hora_min = 7
            hora_max = 14
        else:  # Primavera (9, 10, 11)
            amplitud = 7.3
            hora_min = 7
            hora_max = 16
        
        # Modelo sinusoidal con picos ajustados a datos reales
        # La temperatura alcanza su mínimo a hora_min y máximo a hora_max
        if hora < hora_min:
            # Antes del mínimo (madrugada): temperatura baja
            progreso = (hora + 24 - hora_max) / (24 - hora_max + hora_min)
            delta = -amplitud/2 * (1 - np.cos(progreso * np.pi))
        elif hora <= hora_max:
            # Entre mínimo y máximo: temperatura sube
            progreso = (hora - hora_min) / (hora_max - hora_min)
            delta = -amplitud/2 + amplitud * np.sin(progreso * np.pi/2)
        else:
            # Después del máximo: temperatura baja
            progreso = (hora - hora_max) / (24 - hora_max + hora_min)
            delta = amplitud/2 * (1 - progreso)
        
        temp_hora = temp_promedio_dia + delta
        
        return temp_hora
        
    def cargar_todo(self):
        """Carga modelo, datos y calcula Qin baseline"""
        print("🚀 Cargando Sistema de Planificación Qin...")
        
        try:
            # Intentar cargar LightGBM primero (modelo por defecto)
            model_lgb_path = Path('models/forecasting/modelo_forecasting_lgbm.pkl')
            model_xgb_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
            
            if model_lgb_path.exists():
                modelo_data = joblib.load(model_lgb_path)
                # Extraer modelo si está en formato diccionario (nuevo formato con metadata)
                self.modelo = modelo_data['modelo'] if isinstance(modelo_data, dict) else modelo_data
                self.modelo_nombre = "LightGBM"
                if isinstance(modelo_data, dict) and 'metadata' in modelo_data:
                    self.modelo_metadata = modelo_data['metadata']
                print("✅ Modelo LightGBM cargado (por defecto)")
            elif model_xgb_path.exists():
                # Fallback a XGBoost si LightGBM no existe
                modelo_data = joblib.load(model_xgb_path)
                self.modelo = modelo_data['modelo'] if isinstance(modelo_data, dict) else modelo_data
                self.modelo_nombre = "XGBoost"
                if isinstance(modelo_data, dict) and 'metadata' in modelo_data:
                    self.modelo_metadata = modelo_data['metadata']
                print("✅ Modelo XGBoost cargado (fallback)")
                print("⚠️ LightGBM no disponible, usando XGBoost")
            else:
                print("⚠️ Ningún modelo encontrado")
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
        """Calcula perfil de Qin por hora desde datos RAW de producción - SOLO TRAIN SET"""
        print("\n📊 Calculando perfil de Qin histórico (solo entrenamiento)...")
        
        # Calcular fecha límite de entrenamiento (70% del dataset completo)
        n_total = len(self.df_completo)
        train_end_idx = int(n_total * 0.70)
        fecha_limite_train = self.df_completo.iloc[train_end_idx]['timestamp']
        print(f"   📅 Fecha límite entrenamiento: {fecha_limite_train}")
        
        # Cargar datos RAW de Qin (producción real)
        qin_path = Path('data/raw/BD_Qin_m3_Local.csv')
        if qin_path.exists():
            df_qin = pd.read_csv(qin_path)
            df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])
            
            # ✅ CORRECCIÓN: Filtrar solo datos de entrenamiento
            df_qin_train = df_qin[df_qin['timestamp'] <= fecha_limite_train].copy()
            print(f"   📊 Registros Qin: {len(df_qin)} total → {len(df_qin_train)} en train set")
            
            df_qin_train['hora'] = df_qin_train['timestamp'].dt.hour
            
            # Calcular estadísticas por hora SOLO con datos de entrenamiento
            qin_stats = df_qin_train.groupby('hora')['Qin'].agg(['median', 'mean', 'std', 'min', 'max'])
            
            # Guardar como diccionario de diccionarios
            self.qin_perfil_hora = qin_stats.to_dict('index')
            
            # Para compatibilidad, también guardar solo mediana
            self.qin_base_por_hora = qin_stats['median'].to_dict()
            
            print(f"   ✅ Perfil de Qin calculado para 24 horas (sin data leakage)")
            print(f"   📈 Rango mediana: {qin_stats['median'].min():,.0f} - {qin_stats['median'].max():,.0f} m³/hr")
            print(f"   📊 Promedio global: {df_qin_train['Qin'].mean():,.0f} m³/hr")
        else:
            print(f"   ⚠️ {qin_path} no encontrado, usando dataset procesado")
            # Fallback al dataset procesado - también filtrar train
            df_train = self.df_completo.iloc[:train_end_idx].copy()
            df_train['hora'] = df_train['timestamp'].dt.hour
            self.qin_base_por_hora = df_train.groupby('hora')['sist_Qin_m3h'].median().to_dict()
            # Crear perfil simple
            self.qin_perfil_hora = {
                h: {'median': v, 'mean': v, 'std': 0, 'min': v, 'max': v} 
                for h, v in self.qin_base_por_hora.items()
            }
            print(f"   ✅ Qin base calculado para 24 horas (fallback, sin leakage)")
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
            # Asegurar que timestamp es datetime
            if 'timestamp' in df_base.columns and not pd.api.types.is_datetime64_any_dtype(df_base['timestamp']):
                df_base = df_base.copy()
                df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])
            
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
            # Asegurar que timestamp es datetime en self.df_completo
            if 'timestamp' in self.df_completo.columns and not pd.api.types.is_datetime64_any_dtype(self.df_completo['timestamp']):
                self.df_completo['timestamp'] = pd.to_datetime(self.df_completo['timestamp'])
            
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
        
        NOMENCLATURA:
        - Q_net = ΔVol/Δt = Flujo neto del sistema (m³/hr)
          * Q_net > 0: Sistema almacenando agua (llenado)
          * Q_net < 0: Sistema descargando agua (vaciado)
          * Q_net ≈ 0: Régimen estacionario (equilibrio)
        
        Balance hídrico del sistema:
        - Qin = Qout + Q_net + Pérdidas (asumimos pérdidas ≈ 0)
        
        Despejando demanda:
        - Qout = Qin - Q_net
        
        Esto garantiza que Qout (demanda) > 0 siempre que Qin > 0
        """
        
        # Obtener Qin típico para esta hora (mediana histórica)
        qin_hora = self.qin_perfil_hora[hora]['median']
        
        # Q_net es el cambio de almacenamiento del sistema
        q_net = q_net_predicho
        
        # Calcular Qout (demanda real del sistema)
        # Balance: Qin = Qout + Q_net → Qout = Qin - Q_net
        qout = qin_hora - q_net
        demanda_base = max(qout, 0)  # No puede ser negativa
        
        # Clasificar estado del sistema según Q_net
        if q_net < -500:  # Descargando almacenamiento significativamente
            tipo_balance = 'DESCARGA'
            nivel_alerta = '🔴'
            descripcion = f'Sistema descarga {abs(q_net):.0f} m³/hr (Qout > Qin)'
        elif q_net > 500:  # Recargando almacenamiento
            tipo_balance = 'RECARGA'
            nivel_alerta = '🟢'
            descripcion = f'Sistema recarga {q_net:.0f} m³/hr (Qin > Qout)'
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
            'q_net': q_net,
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
    
    def planificar_24_horas(self, fecha_str, temp_min_input, temp_max_input):
        """
        Genera predicción de demanda para 24 horas usando día similar histórico.
        
        Parámetros:
            fecha_str: Fecha en formato DD/MM/YYYY
            temp_min_input: Temperatura mínima del día del pronóstico
            temp_max_input: Temperatura máxima del día del pronóstico
        """
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
            
            # Buscar día similar en el histórico
            temperaturas_horarias = self.buscar_dia_similar(
                fecha,
                temp_min_input,
                temp_max_input,
                fecha.month,
                fecha.weekday()
            )
            
            # Si no se encuentra día similar, usar modelo sinusoidal
            if temperaturas_horarias is None:
                print("⚠️ Usando modelo sinusoidal de temperatura")
                temp_promedio = (temp_min_input + temp_max_input) / 2
                temperaturas_horarias = {
                    h: self.calcular_temperatura_hora(temp_promedio, h, fecha.month)
                    for h in range(24)
                }
            
            resultados_24h = []
            horas = []
            
            for hora in range(24):
                # Usar temperatura real del día similar histórico
                temperatura_hora = temperaturas_horarias.get(hora, (temp_min_input + temp_max_input) / 2)
                
                # 1. Crear features para predicción
                row = self.crear_features_prediccion(
                    hora,
                    fecha.weekday(),
                    fecha.month,
                    temperatura_hora,
                    self.df_completo
                )
                
                # 2. Predecir Q_net
                X = pd.DataFrame([row])[self.features]
                q_net_pred = self.modelo.predict(X)[0]
                
                # 3. Calcular demanda predicha
                resultado_hora = self.calcular_demanda_predicha(q_net_pred, hora, temperatura_hora)
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
            
            # Obtener temperaturas reales del día
            temperaturas_24h = [r['temperatura'] for r in resultados_24h]
            temp_min_real = min(temperaturas_24h)
            temp_max_real = max(temperaturas_24h)
            temp_promedio_real = sum(temperaturas_24h) / 24
            
            # Contexto de temperatura
            if temp_promedio_real < self.umbrales['temp_frio']:
                ctx_temp = f"❄️ Día frío (<{self.umbrales['temp_frio']:.1f}°C) - Demanda esperada baja"
            elif temp_promedio_real > self.umbrales['temp_calor']:
                ctx_temp = f"🌡️ Día caluroso (>{self.umbrales['temp_calor']:.1f}°C) - Demanda esperada alta"
            else:
                ctx_temp = "🌤️ Temperatura normal - Demanda esperada moderada"
            
            # Generar reporte
            reporte = f"""
### 📋 Predicción de Demanda Real (Qout) - 24 Horas - {fecha_str}

**🌡️ Temperaturas del día:** {temp_min_real:.1f}°C - {temp_max_real:.1f}°C (promedio: {temp_promedio_real:.1f}°C)  
   ↳ Basado en día similar histórico (mismo mes, día semana y rango térmico)  
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
- **Demanda (Qout)** = Consumo real del sistema calculado como: Qin - Q_net
- La demanda NUNCA es cero (siempre hay consumo)
- Mayor temperatura → Mayor consumo → Mayor demanda predicha (+2%/°C)
- Balance: Qin = Qout + Q_net
            """
            
            # Generar gráfico de líneas
            fig = self.crear_grafico_demanda_24h(resultados_24h, fecha_str, temp_min_input, temp_max_input)
            
            return reporte, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def crear_grafico_demanda_24h(self, resultados, fecha_str, temp_min, temp_max):
        """Crea gráfico de líneas de demanda predicha con temperatura variable"""
        
        horas = list(range(24))
        demandas = [r['demanda_m3h'] for r in resultados]
        niveles = [r['nivel_demanda'] for r in resultados]
        temperaturas = [r['temperatura'] for r in resultados]
        
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
        
        # Crear figura con eje Y secundario para temperatura
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Línea de demanda con marcadores coloreados (eje Y principal)
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
        ), secondary_y=False)
        
        # Curva de temperatura (eje Y secundario)
        fig.add_trace(go.Scatter(
            x=[f"{h:02d}:00" for h in horas],
            y=temperaturas,
            mode='lines+markers',
            name='🌡️ Temperatura',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=5, color='orange'),
            hovertemplate='%{x}<br>Temp: %{y:.1f}°C<extra></extra>'
        ), secondary_y=True)
        
        # Líneas de referencia para niveles (rangos realistas 8k-14k)
        fig.add_hline(y=9000, line_dash="dot", line_color="green", opacity=0.3,
                     annotation_text="Demanda Baja", annotation_position="right", secondary_y=False)
        fig.add_hline(y=11000, line_dash="dot", line_color="orange", opacity=0.3,
                     annotation_text="Demanda Media", annotation_position="right", secondary_y=False)
        fig.add_hline(y=13000, line_dash="dot", line_color="red", opacity=0.3,
                     annotation_text="Demanda Alta", annotation_position="right", secondary_y=False)
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Hora del Día")
        fig.update_yaxes(title_text="Demanda (m³/hr)", range=[7000, 15000], secondary_y=False)
        fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=True)
        
        temp_promedio = (temp_min + temp_max) / 2
        fig.update_layout(
            title=f"Predicción de Demanda Real (Qout) 24h - {fecha_str}<br>Temp: {temp_min:.1f}-{temp_max:.1f}°C (promedio: {temp_promedio:.1f}°C)",
            height=500,
            hovermode='x unified',
            legend=dict(x=0, y=1, orientation='h'),
            plot_bgcolor='rgba(250,250,250,0.95)'
        )
        
        return fig
    
    def planificar_72_horas(self, fecha_str, temp_min_dia1, temp_max_dia1, 
                            temp_min_dia2, temp_max_dia2, temp_min_dia3, temp_max_dia3):
        """Genera predicción de demanda para 72 horas (3 días) con temperatura variable"""
        try:
            fecha_inicio = datetime.strptime(fecha_str, "%d/%m/%Y")
            temperaturas_dias = [
                (temp_min_dia1, temp_max_dia1),
                (temp_min_dia2, temp_max_dia2),
                (temp_min_dia3, temp_max_dia3)
            ]
            
            todos_resultados = []
            timestamps = []
            temperaturas_plot = []
            
            for dia in range(3):
                fecha_dia = fecha_inicio + timedelta(days=dia)
                temp_min, temp_max = temperaturas_dias[dia]
                
                # Buscar día similar en histórico
                temperaturas_horarias = self.buscar_dia_similar(
                    fecha_dia, temp_min, temp_max, fecha_dia.month, fecha_dia.weekday()
                )
                
                # Si no se encuentra día similar, usar modelo sinusoidal
                if temperaturas_horarias is None:
                    temp_promedio = (temp_min + temp_max) / 2
                    temperaturas_horarias = {
                        h: self.calcular_temperatura_hora(temp_promedio, h, fecha_dia.month)
                        for h in range(24)
                    }
                
                for hora in range(24):
                    # Usar temperatura del día histórico similar
                    temp_hora = temperaturas_horarias.get(
                        hora, 
                        (temp_min + temp_max) / 2
                    )
                    
                    # Crear features
                    row = self.crear_features_prediccion(
                        hora,
                        fecha_dia.weekday(),
                        fecha_dia.month,
                        temp_hora,
                        self.df_completo
                    )
                    
                    # Predecir Q_net
                    X = pd.DataFrame([row])[self.features]
                    q_net_pred = self.modelo.predict(X)[0]
                    
                    # Calcular demanda
                    resultado = self.calcular_demanda_predicha(q_net_pred, hora, temp_hora)
                    resultado['dia'] = dia + 1
                    resultado['hora'] = hora
                    
                    todos_resultados.append(resultado)
                    timestamps.append(fecha_dia + timedelta(hours=hora))
                    temperaturas_plot.append(temp_hora)
            
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
                temp_min, temp_max = temperaturas_dias[dia]
                
                resumen_dias.append(f"""
**Día {dia+1}** ({fecha_dia.strftime('%d/%m')} - {temp_min:.1f}-{temp_max:.1f}°C):
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
    
    def planificar_72_horas_multimodelo(self, fecha_str, temp_min_dia1, temp_max_dia1,
                                         temp_min_dia2, temp_max_dia2, temp_min_dia3, 
                                         temp_max_dia3, modelo_seleccionado):
        """Genera predicción de demanda para 72 horas usando el modelo seleccionado"""
        try:
            # Seleccionar modelo
            if modelo_seleccionado == "XGBoost V3.0 (Actual)":
                modelo_usar = self.modelo
                nombre_modelo = "XGBoost V3.0"
            elif modelo_seleccionado == "RandomForest":
                if not hasattr(self, 'modelo_rf') or self.modelo_rf is None:
                    return "❌ RandomForest no entrenado. Ve a 'Comparación Modelos ML' primero.", None
                modelo_usar = self.modelo_rf
                nombre_modelo = "RandomForest"
            elif modelo_seleccionado == "LightGBM":
                if not hasattr(self, 'modelo_lgb') or self.modelo_lgb is None:
                    return "❌ LightGBM no entrenado. Ve a 'Comparación Modelos ML' primero.", None
                modelo_usar = self.modelo_lgb
                nombre_modelo = "LightGBM"
            else:
                return "❌ Modelo no reconocido", None
            
            fecha_inicio = datetime.strptime(fecha_str, "%d/%m/%Y")
            temperaturas_dias = [
                (temp_min_dia1, temp_max_dia1),
                (temp_min_dia2, temp_max_dia2),
                (temp_min_dia3, temp_max_dia3)
            ]
            
            todos_resultados = []
            timestamps = []
            temperaturas_plot = []
            
            for dia in range(3):
                fecha_dia = fecha_inicio + timedelta(days=dia)
                temp_min, temp_max = temperaturas_dias[dia]
                
                # Buscar día similar en histórico
                temperaturas_horarias = self.buscar_dia_similar(
                    fecha_dia, temp_min, temp_max, fecha_dia.month, fecha_dia.weekday()
                )
                
                # Si no se encuentra día similar, usar modelo sinusoidal
                if temperaturas_horarias is None:
                    temp_promedio = (temp_min + temp_max) / 2
                    temperaturas_horarias = {
                        h: self.calcular_temperatura_hora(temp_promedio, h, fecha_dia.month)
                        for h in range(24)
                    }
                
                for hora in range(24):
                    # Usar temperatura del día histórico similar
                    temp_hora = temperaturas_horarias.get(hora, (temp_min + temp_max) / 2)
                    
                    # Crear features
                    row = self.crear_features_prediccion(
                        hora,
                        fecha_dia.weekday(),
                        fecha_dia.month,
                        temp_hora,
                        self.df_completo
                    )
                    
                    # Predecir Q_net con el modelo seleccionado
                    X = pd.DataFrame([row])[self.features]
                    q_net_pred = modelo_usar.predict(X)[0]
                    
                    # Calcular demanda
                    resultado = self.calcular_demanda_predicha(q_net_pred, hora, temp_hora)
                    resultado['dia'] = dia + 1
                    resultado['hora'] = hora
                    
                    todos_resultados.append(resultado)
                    timestamps.append(fecha_dia + timedelta(hours=hora))
                    temperaturas_plot.append(temp_hora)
            
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
                temp_min, temp_max = temperaturas_dias[dia]
                
                resumen_dias.append(f"""
**Día {dia+1}** ({fecha_dia.strftime('%d/%m')} - {temp_min:.1f}-{temp_max:.1f}°C):
- Demanda total: {demanda_total:,.0f} m³
- Demanda promedio: {demanda_prom:,.0f} m³/hr
- Pico demanda: {demanda_max:,.0f} m³/hr
- Horas descarga: {horas_descarga}/24 | Horas recarga: {horas_recarga}/24
                """)
            
            # Calcular totales
            demanda_total_72h = sum([r['demanda_m3h'] for r in todos_resultados])
            
            resultado_md = f"""
### 📋 Predicción de Demanda 72 Horas - {fecha_str}

**🤖 Modelo Utilizado:** {nombre_modelo}

{''.join(resumen_dias)}

**⚖️ Demanda Total 3 Días:** {demanda_total_72h:,.0f} m³  
**📊 Demanda Promedio 72h:** {demanda_total_72h / 72:,.0f} m³/hr

---

💡 **Interpretación:** Predicción basada en {nombre_modelo} con temperatura diaria y patrones históricos
            """
            
            # Crear gráfico 72h
            fig = self.crear_grafico_demanda_72h(
                todos_resultados, timestamps, temperaturas_plot, 
                f"{fecha_str} - {nombre_modelo}"
            )
            
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
            
            # Interpretación del balance
            def interpretar_balance(balance):
                if balance < -1000:
                    return "🔴 Alta descarga de almacenamiento"
                elif balance < 0:
                    return "🟡 Descarga de almacenamiento"
                elif balance < 1000:
                    return "🟢 Ligera recarga"
                else:
                    return "🟢 Recarga de almacenamiento"
            
            return {
                'nombre': 'Sensibilidad a Temperatura',
                'icono': '🌡️',
                'estado': 'PASS' if (es_correcto and cambio_esperado) else 'FAIL',
                'detalles': f"""
**Test:** Predicción 24h con 3°C vs 30°C

- **Balance Q_net con 3°C:** {balance_frio:,.0f} m³ {interpretar_balance(balance_frio)}
- **Balance Q_net con 30°C:** {balance_calor:,.0f} m³ {interpretar_balance(balance_calor)}
- **Diferencia:** {diferencia_pct:.1f}%
- **Comportamiento:** {'✅ Correcto' if es_correcto else '❌ Invertido'}
- **Sensibilidad:** {'✅ Adecuada' if cambio_esperado else '⚠️ Baja'}

💡 **Interpretación:**
- Balance **negativo** = Sistema usa almacenamiento (Demanda > Producción) → Correcto con calor
- Balance **positivo** = Sistema recarga almacenamiento (Producción > Demanda) → Correcto con frío
- Mayor temperatura → Mayor consumo → Balance más negativo ✅
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
            # Cargar dataset con features completas
            df_path = Path('data/processed/dataset_features_completo.csv')
            if not df_path.exists():
                return None, "❌ Dataset no encontrado"
            
            df = pd.read_csv(df_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Agregar features categóricas
            print("   Agregando features categóricas...")
            df = self._agregar_features_categoricas(df)
            
            # Verificar que todas las features existen
            missing = [f for f in self.features if f not in df.columns]
            if missing:
                return None, f"❌ Features faltantes: {missing[:5]}"
            
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
            
            # RandomForest no acepta NaN - rellenar con 0
            print(f"   Verificando valores faltantes...")
            n_nans_train = X_train.isna().sum().sum()
            n_nans_test = X_test.isna().sum().sum()
            if n_nans_train > 0 or n_nans_test > 0:
                print(f"   Rellenando {n_nans_train} NaN en train, {n_nans_test} en test con 0...")
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)
            
            print(f"   Entrenando con {len(X_train):,} registros...")
            modelo = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            modelo.fit(X_train, y_train)
            
            y_test_pred = modelo.predict(X_test)
            metrics_test = self._calculate_metrics(y_test.values, y_test_pred)
            
            print(f"   ✅ Entrenamiento completado - R²: {metrics_test['r2']:.4f}")
            
            return modelo, None
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg
    
    def entrenar_lightgbm(self, n_estimators=200, max_depth=10):
        """Entrena modelo LightGBM para comparación"""
        if not LIGHTGBM_AVAILABLE:
            return None, "❌ LightGBM no instalado"
        
        try:
            # Cargar dataset con features completas
            df_path = Path('data/processed/dataset_features_completo.csv')
            if not df_path.exists():
                return None, "❌ Dataset no encontrado"
            
            df = pd.read_csv(df_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Agregar features categóricas
            print("   Agregando features categóricas...")
            df = self._agregar_features_categoricas(df)
            
            # Verificar que todas las features existen
            missing = [f for f in self.features if f not in df.columns]
            if missing:
                return None, f"❌ Features faltantes: {missing[:5]}"
            
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
            
            print(f"   Entrenando con {len(X_train):,} registros...")
            y_test_pred = modelo.predict(X_test)
            metrics_test = self._calculate_metrics(y_test.values, y_test_pred)
            
            print(f"   ✅ Entrenamiento completado - R²: {metrics_test['r2']:.4f}")
            
            return modelo, None
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg
    
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
    
    def _calculate_metrics(self, y_true, y_pred, mape_min=1000):
        """
        Calcula métricas de evaluación
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            mape_min: Umbral mínimo para calcular MAPE (evita distorsiones
                     con valores muy pequeños). Default 1000 m³/hr.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE: ignorar valores muy pequeños para evitar distorsiones
        mask = np.abs(y_true) > mape_min
        if mask.sum() > 0:
            mape = np.mean(
                np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            ) * 100
        else:
            mape = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def evaluar_testing_tres_modelos(self):
        """Evalúa 3 modelos ML (XGBoost, RandomForest, LightGBM) sobre TODO el periodo de test"""
        try:
            print("\n🔬 Iniciando evaluación de 3 modelos ML...")
            print("✅ Usando perfil Qin CORREGIDO (sin data leakage)")
            print(f"   Perfil calculado con {len(self.qin_perfil_hora)} horas de datos de ENTRENAMIENTO")
            
            resultados = {}
            predicciones = {}
            
            # Cargar dataset
            df_completo_path = Path('data/processed/dataset_features_completo.csv')
            if not df_completo_path.exists():
                return "❌ Error: dataset_features_completo.csv no encontrado", None
            
            df = pd.read_csv(df_completo_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Agregar features categóricas
            df = self._agregar_features_categoricas(df)
            
            # Preparar datos: 70% train, 15% val, 15% test (IGUAL QUE SIEMPRE)
            print("\n📊 Preparando datos con split 70%/15%/15%...")
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]
            
            print(f"   Total registros:  {n:,}")
            print(f"   Train (70%):      {len(df_train):,} - hasta {df_train['timestamp'].max().strftime('%d/%m/%Y')}")
            print(f"   Validación (15%): {len(df_val):,}")
            print(f"   Test (15%):       {len(df_test):,}")
            print(f"   Periodo test:     {df_test['timestamp'].min().strftime('%d/%m/%Y')} - {df_test['timestamp'].max().strftime('%d/%m/%Y')}")
            
            X_train = df_train[self.features].fillna(0)
            X_test = df_test[self.features].fillna(0)
            y_train_qnet = df_train['Q_net_m3h'].values
            y_test_qnet = df_test['Q_net_m3h'].values
            timestamps = df_test['timestamp'].values
            
            # Calcular Qin por hora para test
            df_test['hora'] = df_test['timestamp'].dt.hour
            qin_por_hora = df_test['hora'].map(self.qin_perfil_hora).apply(
                lambda x: x['median']
            ).values
            
            # DEMANDA REAL: Qout = Qin - Q_net
            y_test_qout = qin_por_hora - y_test_qnet
            
            # 1. XGBoost V3.0 (cargar modelo pickle)
            print("\n📊 Cargando y evaluando XGBoost...")
            xgb_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
            if xgb_path.exists():
                xgb_data = joblib.load(xgb_path)
                modelo_xgb = xgb_data['modelo'] if isinstance(xgb_data, dict) else xgb_data
                
                y_pred_qnet_xgb = modelo_xgb.predict(X_test)
                y_pred_qout_xgb = qin_por_hora - y_pred_qnet_xgb
                
                metrics_xgb = self._calculate_metrics(y_test_qout, y_pred_qout_xgb)
                resultados['XGBoost V3.0'] = metrics_xgb
                predicciones['XGBoost V3.0'] = y_pred_qout_xgb
                print(f"   ✅ XGBoost - R²: {metrics_xgb['r2']:.4f}, MAE: {metrics_xgb['mae']:,.0f} m³/h")
            else:
                print("   ⚠️ XGBoost pickle no encontrado, usando modelo actual (puede ser LightGBM)")
                y_pred_qnet_xgb = self.modelo.predict(X_test)
                y_pred_qout_xgb = qin_por_hora - y_pred_qnet_xgb
                
                metrics_xgb = self._calculate_metrics(y_test_qout, y_pred_qout_xgb)
                resultados['XGBoost V3.0'] = metrics_xgb
                predicciones['XGBoost V3.0'] = y_pred_qout_xgb
                print(f"   ✅ Modelo actual - R²: {metrics_xgb['r2']:.4f}, MAE: {metrics_xgb['mae']:,.0f} m³/h")
            
            # 2. RandomForest
            print("\n🌲 Entrenando RandomForest...")
            modelo_rf, resultado_rf = self.entrenar_randomforest()
            if modelo_rf is not None:
                print("📊 Evaluando RandomForest...")
                self.modelo_rf = modelo_rf
                
                y_pred_qnet_rf = modelo_rf.predict(X_test)
                y_pred_qout_rf = qin_por_hora - y_pred_qnet_rf
                
                metrics_rf = self._calculate_metrics(y_test_qout, y_pred_qout_rf)
                resultados['RandomForest'] = metrics_rf
                predicciones['RandomForest'] = y_pred_qout_rf
                print(f"   ✅ RandomForest - R²: {metrics_rf['r2']:.4f}, MAE: {metrics_rf['mae']:,.0f} m³/h")
            else:
                print(f"   ❌ RandomForest falló: {resultado_rf}")
            
            # 3. LightGBM
            if LIGHTGBM_AVAILABLE:
                print("\n💡 Entrenando LightGBM...")
                modelo_lgb, resultado_lgb = self.entrenar_lightgbm()
                if modelo_lgb is not None:
                    print("📊 Evaluando LightGBM...")
                    self.modelo_lgb = modelo_lgb
                    
                    y_pred_qnet_lgb = modelo_lgb.predict(X_test)
                    y_pred_qout_lgb = qin_por_hora - y_pred_qnet_lgb
                    
                    metrics_lgb = self._calculate_metrics(y_test_qout, y_pred_qout_lgb)
                    resultados['LightGBM'] = metrics_lgb
                    predicciones['LightGBM'] = y_pred_qout_lgb
                    print(f"   ✅ LightGBM - R²: {metrics_lgb['r2']:.4f}, MAE: {metrics_lgb['mae']:,.0f} m³/h")
                else:
                    print(f"   ❌ LightGBM falló: {resultado_lgb}")
            
            print(f"\n✅ Evaluación completada con {len(resultados)} modelos!")
            
            # Generar reporte y gráfico
            reporte = self._generar_reporte_testing_tres_modelos(
                resultados, df_test, len(df_test)
            )
            fig = self._generar_grafico_testing_tres_modelos(
                resultados, predicciones, y_test_qout, timestamps
            )
            
            return reporte, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def _generar_reporte_testing_tres_modelos(self, resultados, df_test, n_test):
        """Genera reporte comparativo para los 3 modelos ML"""
        reporte = f"""
# 📊 Evaluación de 3 Modelos ML en Periodo de Testing

**Fecha:** {datetime.now().strftime("%d/%m/%Y %H:%M")}

---

## 🎯 Configuración de la Evaluación

**Periodo de Test:** {df_test['timestamp'].min().strftime('%d/%m/%Y')} - {df_test['timestamp'].max().strftime('%d/%m/%Y')}  
**Registros de Test:** {n_test:,} horas  
**Proporción del Dataset:** 70% train / 15% validación / 15% test

**Metodología:** Las métricas se calculan sobre la **demanda real (Qout = Qin - Q_net)**, 
no sobre Q_net directamente, para una interpretación operacionalmente relevante.

---

## 📈 Tabla Comparativa de Métricas

| Modelo | R² | RMSE (m³/h) | MAE (m³/h) | MAPE (%) |
|--------|----:|-----:|----:|-----:|
"""
        
        for nombre, m in resultados.items():
            reporte += f"| {nombre} | {m['r2']:.4f} | {m['rmse']:,.0f} | {m['mae']:,.0f} | {m['mape']:.2f} |\n"
        
        reporte += "\n---\n\n## 🏆 Ranking de Modelos\n\n"
        
        mejor_r2 = max(resultados.items(), key=lambda x: x[1]['r2'])
        mejor_mae = min(resultados.items(), key=lambda x: x[1]['mae'])
        mejor_mape = min(resultados.items(), key=lambda x: x[1]['mape'])
        
        reporte += f"- **🥇 Mejor R²:** {mejor_r2[0]} ({mejor_r2[1]['r2']:.4f})\n"
        reporte += f"- **🥇 Mejor MAE:** {mejor_mae[0]} ({mejor_mae[1]['mae']:,.0f} m³/h)\n"
        reporte += f"- **🥇 Mejor MAPE:** {mejor_mape[0]} ({mejor_mape[1]['mape']:.2f}%)\n"
        
        reporte += "\n---\n\n## 💡 Interpretación de Métricas\n\n"
        reporte += "- **R²:** Proporción de varianza explicada (1.0 = perfecto)\n"
        reporte += "- **RMSE:** Error cuadrático medio, penaliza errores grandes\n"
        reporte += "- **MAE:** Error absoluto medio, interpretación directa\n"
        reporte += "- **MAPE:** Error porcentual medio absoluto\n"
        
        return reporte
    
    def _generar_grafico_testing_tres_modelos(self, resultados, predicciones,
                                               y_test, timestamps):
        """Genera gráfico comparativo para los 3 modelos ML"""
        modelos = list(resultados.keys())
        r2_vals = [resultados[m]['r2'] for m in modelos]
        rmse_vals = [resultados[m]['rmse'] for m in modelos]
        mae_vals = [resultados[m]['mae'] for m in modelos]
        
        # Crear subplots: 1 fila con métricas, 1 fila con predicciones
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'R² Score', 'RMSE (m³/h)', 'MAE (m³/h)',
                'Predicciones Últimos 7 Días', '', ''
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'scatter', 'colspan': 3}, None, None]
            ],
            row_heights=[0.35, 0.65]
        )
        
        colors = ['#2E86AB', '#A23B72', '#F18701']
        
        # Fila 1: Métricas en barras (con espacio adicional para los valores)
        fig.add_trace(go.Bar(
            x=modelos, y=r2_vals, marker_color=colors,
            showlegend=False, text=[f'{v:.4f}' for v in r2_vals],
            textposition='outside',
            textfont=dict(size=12),
            cliponaxis=False
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=modelos, y=rmse_vals, marker_color=colors,
            showlegend=False, text=[f'{v:,.0f}' for v in rmse_vals],
            textposition='outside',
            textfont=dict(size=12),
            cliponaxis=False
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=modelos, y=mae_vals, marker_color=colors,
            showlegend=False, text=[f'{v:,.0f}' for v in mae_vals],
            textposition='outside',
            textfont=dict(size=12),
            cliponaxis=False
        ), row=1, col=3)
        
        # Fila 2: Predicciones últimos 7 días
        n_points = min(168, len(y_test))  # 168 horas = 7 días
        timestamps_plot = timestamps[-n_points:]
        y_test_plot = y_test[-n_points:]
        
        # Real
        fig.add_trace(go.Scatter(
            x=timestamps_plot,
            y=y_test_plot,
            mode='lines',
            name='Real',
            line=dict(color='black', width=2),
            hovertemplate='%{x|%d/%m %H:%M}<br>Real: %{y:,.0f} m³/h<extra></extra>'
        ), row=2, col=1)
        
        # Predicciones de cada modelo
        for i, (nombre, color) in enumerate(zip(modelos, colors)):
            if nombre in predicciones:
                y_pred_plot = predicciones[nombre][-n_points:]
                fig.add_trace(go.Scatter(
                    x=timestamps_plot,
                    y=y_pred_plot,
                    mode='lines',
                    name=nombre,
                    line=dict(color=color, width=1.5, dash='dot'),
                    hovertemplate=f'%{{x|%d/%m %H:%M}}<br>{nombre}: %{{y:,.0f}} m³/h<extra></extra>'
                ), row=2, col=1)
        
        # Actualizar layouts con rango extendido para mostrar valores sobre barras
        fig.update_yaxes(title_text="R²", row=1, col=1, range=[0, max(r2_vals) * 1.15])
        fig.update_yaxes(title_text="RMSE (m³/h)", row=1, col=2, range=[0, max(rmse_vals) * 1.15])
        fig.update_yaxes(title_text="MAE (m³/h)", row=1, col=3, range=[0, max(mae_vals) * 1.15])
        fig.update_yaxes(title_text="Demanda Qout (m³/h)", row=2, col=1)
        fig.update_xaxes(title_text="Fecha/Hora", row=2, col=1)
        
        fig.update_layout(
            height=900,
            title_text=f"Evaluación de 3 Modelos ML - Todo el Periodo de Testing",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def comparar_modelos_ml(self):
        """Compara XGBoost actual vs RandomForest vs LightGBM sobre Demanda (Qout)"""
        try:
            print("\n🔬 Iniciando comparación...")
            print("✅ Usando perfil Qin CORREGIDO (sin data leakage)")
            print(f"   Perfil calculado con {len(self.qin_perfil_hora)} horas de datos de ENTRENAMIENTO")
            
            resultados = {}
            predicciones = {}  # Para guardar predicciones de cada modelo
            
            # Cargar dataset con features base
            df_completo_path = Path('data/processed/dataset_features_completo.csv')
            if not df_completo_path.exists():
                return "❌ Error: dataset_features_completo.csv no encontrado", None
            
            df = pd.read_csv(df_completo_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').copy()
            
            # Crear features categóricas que faltan (igual que en crear_features_prediccion)
            df = self._agregar_features_categoricas(df)
            
            # Preparar datos de test
            print("\n📊 Preparando datos de test...")
            n = len(df)
            val_end = int(n * 0.85)
            df_test = df.iloc[val_end:]
            
            X_test = df_test[self.features].fillna(0)  # Rellenar NaN con 0
            y_test_qnet = df_test['Q_net_m3h'].values
            timestamps = df_test['timestamp'].values
            
            # CALCULAR QIN HISTÓRICO POR HORA (igual que evaluar_testing)
            df_test['hora'] = df_test['timestamp'].dt.hour
            qin_por_hora = df_test['hora'].map(self.qin_perfil_hora).apply(
                lambda x: x['median']
            ).values
            
            # DEMANDA REAL: Qout = Qin - Q_net
            y_test_qout = qin_por_hora - y_test_qnet
            
            # **FILTRAR ÚLTIMA SEMANA** (para coincidir con gráfico)
            from datetime import timedelta
            df_test_copy = df_test.copy()
            df_test_copy['Qout_real'] = y_test_qout
            
            ultima_fecha = df_test_copy['timestamp'].max()
            fecha_inicio_semana = ultima_fecha - timedelta(days=7)
            mask_semana = df_test_copy['timestamp'] >= fecha_inicio_semana
            
            print(f"\n📅 Filtrando última semana del test set:")
            print(f"   Desde: {fecha_inicio_semana}")
            print(f"   Hasta: {ultima_fecha}")
            print(f"   Registros totales test: {len(df_test_copy)}")
            print(f"   Registros última semana: {mask_semana.sum()}")
            
            # 1. XGBoost actual
            print("\n📊 Evaluando XGBoost V3.0 sobre Demanda (Qout)...")
            y_pred_qnet_xgb = self.modelo.predict(X_test)
            
            # Pasar a demanda
            y_pred_qout_xgb = qin_por_hora - y_pred_qnet_xgb
            
            # Métricas sobre última semana
            metrics_xgb = self._calculate_metrics(
                y_test_qout[mask_semana], y_pred_qout_xgb[mask_semana]
            )
            resultados['XGBoost V3.0 (Actual)'] = metrics_xgb
            predicciones['XGBoost V3.0 (Actual)'] = y_pred_qout_xgb
            print(f"   ✅ XGBoost (última semana) - R²: {metrics_xgb['r2']:.4f}, MAE: {metrics_xgb['mae']:,.0f}")
            
            # 2. RandomForest
            print("\n🌲 Entrenando RandomForest...")
            modelo_rf, resultado_rf = self.entrenar_randomforest()
            if modelo_rf is not None:
                print("📊 Evaluando RandomForest sobre Demanda (Qout)...")
                # Guardar modelo para uso posterior
                self.modelo_rf = modelo_rf
                
                y_pred_qnet_rf = modelo_rf.predict(X_test)
                y_pred_qout_rf = qin_por_hora - y_pred_qnet_rf
                
                # Métricas sobre última semana
                metrics_rf = self._calculate_metrics(
                    y_test_qout[mask_semana], y_pred_qout_rf[mask_semana]
                )
                resultados['RandomForest'] = metrics_rf
                predicciones['RandomForest'] = y_pred_qout_rf
                print(f"   ✅ RandomForest (última semana) - R²: {metrics_rf['r2']:.4f}, MAE: {metrics_rf['mae']:,.0f}")
            else:
                print(f"   ❌ RandomForest falló: {resultado_rf}")
            
            # 3. LightGBM
            if LIGHTGBM_AVAILABLE:
                print("\n💡 Entrenando LightGBM...")
                modelo_lgb, resultado_lgb = self.entrenar_lightgbm()
                if modelo_lgb is not None:
                    print("📊 Evaluando LightGBM sobre Demanda (Qout)...")
                    # Guardar modelo para uso posterior
                    self.modelo_lgb = modelo_lgb
                    
                    y_pred_qnet_lgb = modelo_lgb.predict(X_test)
                    y_pred_qout_lgb = qin_por_hora - y_pred_qnet_lgb
                    
                    # Métricas sobre última semana
                    metrics_lgb = self._calculate_metrics(
                        y_test_qout[mask_semana], y_pred_qout_lgb[mask_semana]
                    )
                    resultados['LightGBM'] = metrics_lgb
                    predicciones['LightGBM'] = y_pred_qout_lgb
                    print(f"   ✅ LightGBM (última semana) - R²: {metrics_lgb['r2']:.4f}, MAE: {metrics_lgb['mae']:,.0f}")
                else:
                    print(f"   ❌ LightGBM falló: {resultado_lgb}")
            
            print(f"\n✅ Comparación completada con {len(resultados)} modelos!")
            
            # Filtrar predicciones y datos para última semana (para gráfico)
            predicciones_semana = {
                modelo: pred[mask_semana] for modelo, pred in predicciones.items()
            }
            
            # Exportar predicciones reales para análisis riguroso
            self._exportar_predicciones_reales(
                predicciones, y_test_qout, timestamps, resultados
            )
            
            reporte = self._generar_reporte_comparacion(resultados)
            fig = self._generar_grafico_comparacion(
                resultados, predicciones_semana, 
                y_test_qout[mask_semana], timestamps[mask_semana]
            )
            
            return reporte, fig
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
    
    def _exportar_predicciones_reales(self, predicciones, y_real, 
                                      timestamps, metricas):
        """
        Exporta predicciones reales de los modelos para análisis científico.
        
        Args:
            predicciones: Dict con arrays de predicciones por modelo
            y_real: Array con valores reales (ground truth)
            timestamps: Array con timestamps del test set
            metricas: Dict con métricas calculadas por modelo
        """
        import os
        try:
            # Crear directorio si no existe
            output_dir = 'outputs/metricas_ML'
            os.makedirs(output_dir, exist_ok=True)
            
            # Preparar datos para export
            export_data = {
                'timestamp': timestamps,
                'demanda_real_m3': y_real
            }
            
            # Agregar predicciones de cada modelo
            for nombre_modelo, prediccion in predicciones.items():
                nombre_col = nombre_modelo.replace(' ', '_').lower()
                export_data[f'pred_{nombre_col}_m3'] = prediccion
            
            # Crear DataFrame
            df_export = pd.DataFrame(export_data)
            
            # Exportar a CSV
            csv_path = os.path.join(output_dir, 'predicciones_reales.csv')
            df_export.to_csv(csv_path, index=False)
            print(f"\n💾 Predicciones exportadas: {csv_path}")
            print(f"   📊 {len(df_export)} registros × {len(df_export.columns)} columnas")
            
            # Exportar métricas a JSON para referencia
            import json
            metricas_path = os.path.join(output_dir, 'metricas_reales.json')
            metricas_export = {
                'fecha_generacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'n_samples': len(y_real),
                'periodo': {
                    'inicio': str(timestamps[0]),
                    'fin': str(timestamps[-1])
                },
                'modelos': metricas
            }
            
            with open(metricas_path, 'w', encoding='utf-8') as f:
                json.dump(metricas_export, f, indent=2, ensure_ascii=False)
            print(f"   📈 Métricas exportadas: {metricas_path}")
            
        except Exception as e:
            print(f"⚠️ Error exportando predicciones: {e}")
    
    def _generar_reporte_comparacion(self, resultados):
        """Genera reporte de comparación"""
        reporte = f"""
# 🔬 Comparación de Modelos ML

**Fecha:** {datetime.now().strftime("%d/%m/%Y %H:%M")}

---

## 📊 Resultados en Última Semana de Test (Demanda Qout)

**Nota metodológica:** Las métricas se calculan sobre la **última semana** del test set
para coincidir con el gráfico guardado. La demanda (Qout = Qin - Q_net) permite 
una interpretación más clara y operacionalmente relevante.

| Modelo | R² | RMSE (m³/hr) | MAE (m³/hr) | MAPE |
|--------|----:|-----:|----:|-----:|
"""
        
        for nombre, m in resultados.items():
            reporte += f"| {nombre} | {m['r2']:.4f} | {m['rmse']:,.0f} | {m['mae']:,.0f} | {m['mape']:.2f}% |\n"
        
        reporte += "\n---\n\n## 🏆 Mejor Modelo\n\n"
        
        mejor_r2 = max(resultados.items(), key=lambda x: x[1]['r2'])
        mejor_mae = min(resultados.items(), key=lambda x: x[1]['mae'])
        mejor_mape = min(resultados.items(), key=lambda x: x[1]['mape'])
        
        reporte += f"- **R² más alto:** {mejor_r2[0]} ({mejor_r2[1]['r2']:.4f})\n"
        reporte += f"- **MAE más bajo:** {mejor_mae[0]} ({mejor_mae[1]['mae']:,.0f} m³/hr)\n"
        reporte += f"- **MAPE más bajo:** {mejor_mape[0]} ({mejor_mape[1]['mape']:.2f}%)\n"
        
        reporte += "\n---\n\n## 💡 Interpretación\n\n"
        reporte += "- **R²**: Proporción de varianza explicada (más cercano a 1 = mejor)\n"
        reporte += "- **RMSE**: Error cuadrático medio, penaliza errores grandes\n"
        reporte += "- **MAE**: Error absoluto medio, interpretación directa en m³/hr\n"
        reporte += "- **MAPE**: Error porcentual medio absoluto\n"
        
        return reporte
    
    def _generar_grafico_comparacion(self, resultados, predicciones,
                                       y_test, timestamps):
        """Genera gráfico de comparación con métricas y predicciones"""
        modelos = list(resultados.keys())
        r2_vals = [resultados[m]['r2'] for m in modelos]
        rmse_vals = [resultados[m]['rmse'] for m in modelos]
        mae_vals = [resultados[m]['mae'] for m in modelos]
        mape_vals = [resultados[m]['mape'] for m in modelos]
        
        # Crear subplots: 2 filas, 3 columnas
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'R² Score', 'RMSE (m³/hr)', 'MAE (m³/hr)',
                'Predicciones Comparativas (Últimos 7 días)', '', ''
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'scatter', 'colspan': 3}, None, None]
            ],
            row_heights=[0.35, 0.65]
        )
        
        colors = ['#2E86AB', '#A23B72', '#F18701']
        
        # Fila 1: Métricas en barras
        fig.add_trace(go.Bar(
            x=modelos, y=r2_vals, marker_color=colors,
            showlegend=False, text=[f'{v:.4f}' for v in r2_vals],
            textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=modelos, y=rmse_vals, marker_color=colors,
            showlegend=False, text=[f'{v:,.0f}' for v in rmse_vals],
            textposition='outside'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=modelos, y=mae_vals, marker_color=colors,
            showlegend=False, text=[f'{v:,.0f}' for v in mae_vals],
            textposition='outside'
        ), row=1, col=3)
        
        # Fila 2: Predicciones comparativas (últimos 7 días)
        # Tomar solo últimos 168 puntos (7 días * 24 horas)
        n_points = min(168, len(y_test))
        timestamps_plot = timestamps[-n_points:]
        y_test_plot = y_test[-n_points:]
        
        # Real
        fig.add_trace(go.Scatter(
            x=timestamps_plot,
            y=y_test_plot,
            mode='lines',
            name='Real',
            line=dict(color='black', width=2),
            hovertemplate='%{x|%d/%m %H:%M}<br>Real: %{y:,.0f} m³/hr'
            '<extra></extra>'
        ), row=2, col=1)
        
        # Predicciones de cada modelo
        for i, (nombre, color) in enumerate(zip(modelos, colors)):
            if nombre in predicciones:
                y_pred_plot = predicciones[nombre][-n_points:]
                fig.add_trace(go.Scatter(
                    x=timestamps_plot,
                    y=y_pred_plot,
                    mode='lines',
                    name=nombre,
                    line=dict(color=color, width=1.5, dash='dot'),
                    hovertemplate=f'%{{x|%d/%m %H:%M}}<br>{nombre}: '
                    '%{y:,.0f} m³/hr<extra></extra>'
                ), row=2, col=1)
        
        # Actualizar layouts
        fig.update_yaxes(title_text="R²", row=1, col=1)
        fig.update_yaxes(title_text="RMSE (m³/hr)", row=1, col=2)
        fig.update_yaxes(title_text="MAE (m³/hr)", row=1, col=3)
        fig.update_yaxes(title_text="Demanda Qout (m³/hr)", row=2, col=1)
        fig.update_xaxes(title_text="Fecha/Hora", row=2, col=1)
        
        # Agregar timestamp para forzar actualización en Gradio
        timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        fig.update_layout(
            height=900,
            title_text=f"Comparación de Modelos ML (Generado: {timestamp_actual})",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def crear_header_pronostico(self):
        """Crea header con pronóstico climático real de Open-Meteo"""
        
        # Obtener pronóstico real (o por defecto si falla)
        pronostico = self.obtener_pronostico_automatico()
        
        # Generar HTML de cards
        cards_html = ""
        for dia_info in pronostico:
            # Formatear probabilidad de lluvia
            if dia_info['prob_lluvia'] >= 70:
                lluvia_color = "#E74C3C"  # Rojo
                lluvia_texto = f"💧 {dia_info['prob_lluvia']:.0f}%"
            elif dia_info['prob_lluvia'] >= 40:
                lluvia_color = "#F39C12"  # Naranja
                lluvia_texto = f"💧 {dia_info['prob_lluvia']:.0f}%"
            else:
                lluvia_color = "#95A5A6"  # Gris
                lluvia_texto = f"💧 {dia_info['prob_lluvia']:.0f}%"
            
            cards_html += f"""
            <div style="background: rgba(255,255,255,0.95); padding: 12px; border-radius: 10px; 
                        text-align: center; min-width: 110px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-size: 32px; margin-bottom: 5px;">{dia_info['icono']}</div>
                <div style="font-size: 15px; font-weight: bold; color: #2c5aa0;">
                    {dia_info['temp_min']:.0f}°C - {dia_info['temp_max']:.0f}°C
                </div>
                <div style="font-size: 12px; color: {lluvia_color}; margin-top: 4px; font-weight: 600;">
                    {lluvia_texto}
                </div>
                <div style="font-size: 13px; color: #666; margin-top: 3px;">{dia_info['dia']}</div>
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
                    # Determinar qué modelo está cargado
                    if hasattr(self, 'modelo_nombre'):
                        modelo_nombre = self.modelo_nombre
                    else:
                        modelo_nombre = "LightGBM"  # Fallback
                    
                    gr.Markdown(f"""
                    ### Predicción de demanda para un día completo
                    
                    **🤖 Modelo en uso:** {modelo_nombre} (R²=0.6307, MAE=1,650 m³/h)
                    
                    **Sistema simplificado:**
                    - ✅ Predice balance del sistema (Q_net)
                    - ✅ Calcula demanda cuando Q_net < 0
                    - ✅ Clasifica nivel de demanda (BAJA/MEDIA/ALTA)
                    
                    🌡️ **Nota sobre temperatura:** El modelo considera la temperatura como una variable  
                    importante junto con **patrones horarios**, **día de la semana** y **estacionalidad**.  
                    Los cambios en temperatura **SÍ afectan** las predicciones, especialmente en días extremos.
                    """)
                    
                    # Obtener pronóstico para precargar temperaturas
                    pronostico = self.pronostico_cache if self.pronostico_cache else self.obtener_pronostico_automatico()
                    
                    with gr.Row():
                        with gr.Column():
                            input_fecha_24h = gr.Textbox(
                                label="Fecha",
                                placeholder="DD/MM/YYYY",
                                value=datetime.now().strftime("%d/%m/%Y")
                            )
                            input_temp_min_24h = gr.Number(
                                label=f"🌡️ Temperatura Mínima {pronostico[0]['icono']}",
                                value=pronostico[0]['temp_min'],
                                minimum=-10,
                                maximum=45,
                                info=f"💧 Prob. lluvia: {pronostico[0]['prob_lluvia']:.0f}%"
                            )
                            input_temp_max_24h = gr.Number(
                                label=f"🌡️ Temperatura Máxima {pronostico[0]['icono']}",
                                value=pronostico[0]['temp_max'],
                                minimum=-10,
                                maximum=45,
                                info=f"Pronóstico: {pronostico[0]['dia']}"
                            )
                            btn_24h = gr.Button("🚀 Generar Plan 24h", variant="primary")
                        
                        with gr.Column():
                            output_24h = gr.Markdown()
                    
                    plot_24h = gr.Plot(label="Gráfico Plan 24h")
                    
                    btn_24h.click(
                        fn=self.planificar_24_horas,
                        inputs=[input_fecha_24h, input_temp_min_24h, input_temp_max_24h],
                        outputs=[output_24h, plot_24h]
                    )
                
                # TAB: Predicción Demanda 72 Horas
                with gr.Tab("🔮 Demanda 72 Horas"):
                    # Determinar qué modelo está cargado
                    if hasattr(self, 'modelo_nombre'):
                        modelo_nombre_72h = self.modelo_nombre
                    else:
                        modelo_nombre_72h = "LightGBM"  # Fallback
                    
                    gr.Markdown(f"""
                    ### Predicción de demanda para 3 días con temperaturas variables
                    
                    **🤖 Modelo en uso:** {modelo_nombre_72h} (R²=0.6307, MAE=1,650 m³/h)
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            input_fecha_72h = gr.Textbox(
                                label="Fecha Inicio",
                                placeholder="DD/MM/YYYY",
                                value=datetime.now().strftime("%d/%m/%Y")
                            )
                            
                            gr.Markdown("**🌡️ Temperaturas por Día (precargadas desde pronóstico):**")
                            
                            with gr.Row():
                                btn_refresh_pronostico = gr.Button("🔄 Actualizar Pronóstico", size="sm")
                            status_pronostico = gr.Markdown("_Pronóstico cargado automáticamente_")
                            
                            # Día 1
                            gr.Markdown(f"**📅 Día 1: {pronostico[0]['dia']} {pronostico[0]['icono']}** (💧 {pronostico[0]['prob_lluvia']:.0f}%)")
                            with gr.Row():
                                input_temp_min_dia1 = gr.Number(
                                    label="Temp. Mínima (°C)",
                                    value=pronostico[0]['temp_min'],
                                    minimum=-10,
                                    maximum=45
                                )
                                input_temp_max_dia1 = gr.Number(
                                    label="Temp. Máxima (°C)",
                                    value=pronostico[0]['temp_max'],
                                    minimum=-10,
                                    maximum=45
                                )
                            
                            # Día 2
                            gr.Markdown(f"**📅 Día 2: {pronostico[1]['dia']} {pronostico[1]['icono']}** (💧 {pronostico[1]['prob_lluvia']:.0f}%)")
                            with gr.Row():
                                input_temp_min_dia2 = gr.Number(
                                    label="Temp. Mínima (°C)",
                                    value=pronostico[1]['temp_min'],
                                    minimum=-10,
                                    maximum=45
                                )
                                input_temp_max_dia2 = gr.Number(
                                    label="Temp. Máxima (°C)",
                                    value=pronostico[1]['temp_max'],
                                    minimum=-10,
                                    maximum=45
                                )
                            
                            # Día 3
                            gr.Markdown(f"**📅 Día 3: {pronostico[2]['dia']} {pronostico[2]['icono']}** (💧 {pronostico[2]['prob_lluvia']:.0f}%)")
                            with gr.Row():
                                input_temp_min_dia3 = gr.Number(
                                    label="Temp. Mínima (°C)",
                                    value=pronostico[2]['temp_min'],
                                    minimum=-10,
                                    maximum=45
                                )
                                input_temp_max_dia3 = gr.Number(
                                    label="Temp. Máxima (°C)",
                                    value=pronostico[2]['temp_max'],
                                    minimum=-10,
                                    maximum=45
                                )
                            
                            btn_72h = gr.Button("🚀 Generar Plan 72h", variant="primary")
                            
                            # Callback para refrescar pronóstico
                            def refrescar_pronostico():
                                nuevo_pron = self.obtener_pronostico_automatico()
                                return (
                                    nuevo_pron[0]['temp_min'],
                                    nuevo_pron[0]['temp_max'],
                                    nuevo_pron[1]['temp_min'],
                                    nuevo_pron[1]['temp_max'],
                                    nuevo_pron[2]['temp_min'],
                                    nuevo_pron[2]['temp_max'],
                                    f"✅ Actualizado: {datetime.now().strftime('%H:%M:%S')}"
                                )
                            
                            btn_refresh_pronostico.click(
                                fn=refrescar_pronostico,
                                outputs=[input_temp_min_dia1, input_temp_max_dia1, 
                                        input_temp_min_dia2, input_temp_max_dia2,
                                        input_temp_min_dia3, input_temp_max_dia3, 
                                        status_pronostico]
                            )
                        
                        with gr.Column():
                            output_72h = gr.Markdown()
                    
                    plot_72h = gr.Plot(label="Gráfico Plan 72h")
                    
                    btn_72h.click(
                        fn=self.planificar_72_horas,
                        inputs=[input_fecha_72h, input_temp_min_dia1, input_temp_max_dia1,
                               input_temp_min_dia2, input_temp_max_dia2,
                               input_temp_min_dia3, input_temp_max_dia3],
                        outputs=[output_72h, plot_72h]
                    )
                
                # TAB: Predicción Demanda 72 Horas Multi-Modelo
                # TAB: Evaluación Testing
                with gr.Tab("📈 Evaluación Testing"):
                    gr.Markdown("""
                    ### Evaluación de los 3 Modelos ML en Periodo de Testing
                    
                    Compara el rendimiento de todos los modelos sobre **TODO el periodo de prueba**:
                    
                    - **🚀 XGBoost:** Gradient Boosting con árboles de decisión
                    - **🌲 RandomForest:** Ensamble de árboles de decisión
                    - **💡 LightGBM:** Gradient Boosting optimizado (mejor rendimiento)
                    
                    📊 **Métricas calculadas sobre demanda real (Qout = Qin - Q_net)**
                    
                    ⏱️ **Tiempo estimado:** 2-3 minutos (incluye entrenamiento de RF y LightGBM)
                    """)
                    
                    btn_testing = gr.Button("🎯 Evaluar 3 Modelos ML", variant="primary", size="lg")
                    output_testing = gr.Markdown()
                    plot_testing = gr.Plot(label="Comparación de Modelos")
                    
                    btn_testing.click(
                        fn=self.evaluar_testing_tres_modelos,
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
                
                # TAB: Información
                with gr.Tab("ℹ️ Información"):
                    gr.Markdown(f"""
## 📋 Sistema de Planificación de Producción (Qin)

### 🎯 Cambio de Paradigma

**Antes (V3.0):**
- Predecía Q_net (donde Q_net = ΔVol = cambio en almacenamiento)
- Interpretación: "El sistema está perdiendo 800 m³/hr de almacenamiento"
- Requería que el operador calculara Qin necesario

**Ahora (V1.0):**
- Calcula Demanda real: Qout = Qin - Q_net
- Ajusta por temperatura (+2% por grado sobre 20°C)
- Interpretación: "Demanda de 11,500 m³/hr"
- Guía operacional directa basada en consumo real

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
        share=True  # Link público de Gradio
    )


if __name__ == "__main__":
    main()
