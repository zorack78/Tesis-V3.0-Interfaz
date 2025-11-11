"""
DOCUMENTACIÓN COMPLETA DEL MODELO DE PREDICCIÓN DE DEMANDA

Este script genera una documentación exhaustiva de:
1. Archivos de entrada utilizados
2. Cómo el modelo interpreta cada dato
3. Features utilizadas (21 en total)
4. Archivos generados por el modelo
5. Pipeline completo de datos
"""

import pandas as pd
import json
from pathlib import Path
import pickle

print("=" * 100)
print("DOCUMENTACIÓN COMPLETA: MODELO DE PREDICCIÓN DE DEMANDA")
print("=" * 100)

# =============================================================================
# PARTE 1: ARCHIVOS DE ENTRADA (RAW DATA)
# =============================================================================
print("\n" + "=" * 100)
print("PARTE 1: ARCHIVOS DE ENTRADA (RAW DATA)")
print("=" * 100)

archivos_entrada = {
    'BD_VolTotal_X_Hr_m3_UTC.csv': {
        'ubicacion': 'data/raw/',
        'descripcion': 'Volumen total almacenado en el sistema por hora',
        'columnas_principales': ['timestamp', 'Volumen_Total_m3'],
        'frecuencia': 'Horaria',
        'interpretacion': 'Cantidad de agua almacenada en todos los tanques del sistema en m³',
        'uso_modelo': 'Para calcular ΔVolumen = diferencia horaria de almacenamiento'
    },
    'BD_Qin_m3_UTC.csv': {
        'ubicacion': 'data/raw/',
        'descripcion': 'Caudal de entrada al sistema (producción)',
        'columnas_principales': ['timestamp', 'Qin'],
        'frecuencia': 'Horaria',
        'interpretacion': 'Volumen de agua producido/inyectado al sistema en m³/hr',
        'uso_modelo': 'Para calcular Demanda = Qin - ΔVolumen (balance hídrico)'
    },
    'BD_Clima2024a202509_UTC.csv': {
        'ubicacion': 'data/raw/',
        'descripcion': 'Datos climáticos de la región',
        'columnas_principales': ['timestamp', 'temperatura_promedio', 'precipitacion', 
                                 'humedad', 'velocidad_viento'],
        'frecuencia': 'Horaria',
        'interpretacion': 'Variables meteorológicas que pueden afectar consumo de agua',
        'uso_modelo': 'Actualmente NO usado en el modelo (futuras versiones podrían incluirlo)'
    },
    'calendar_social_ES_COMPLETO_20240101_20250930.csv': {
        'ubicacion': 'data/raw/',
        'descripcion': 'Calendario de eventos sociales y feriados en Chile',
        'columnas_principales': ['fecha', 'es_feriado', 'tipo_evento', 'temporada_turistica'],
        'frecuencia': 'Diaria',
        'interpretacion': 'Feriados nacionales, eventos especiales, temporadas turísticas',
        'uso_modelo': 'Features: feriado (binaria), temporada_turistica_alta (binaria)'
    },
    'BD_Capacidad_89Tks_m3.csv': {
        'ubicacion': 'data/raw/',
        'descripcion': 'Capacidades de los 89 tanques del sistema',
        'columnas_principales': ['tanque_id', 'capacidad_m3', 'ubicacion'],
        'frecuencia': 'Estática (configuración)',
        'interpretacion': 'Capacidad máxima de almacenamiento por tanque',
        'uso_modelo': 'Referencia para validación (capacidad total ~156,500 m³)'
    },
    'Vol_X_TK_Hr_m3_UTC.csv': {
        'ubicacion': 'data/raw/',
        'descripcion': 'Volumen individual por tanque por hora',
        'columnas_principales': ['timestamp', 'tanque_id', 'volumen_m3'],
        'frecuencia': 'Horaria por tanque',
        'interpretacion': 'Desglose del volumen total en cada uno de los 89 tanques',
        'uso_modelo': 'Actualmente NO usado (se usa el agregado Volumen_Total_m3)'
    }
}

print("\n📁 ARCHIVOS DE ENTRADA (6 archivos):\n")

for i, (archivo, info) in enumerate(archivos_entrada.items(), 1):
    print(f"{i}. {archivo}")
    print(f"   📂 Ubicación: {info['ubicacion']}")
    print(f"   📝 Descripción: {info['descripcion']}")
    print(f"   📊 Columnas: {', '.join(info['columnas_principales'])}")
    print(f"   ⏱️  Frecuencia: {info['frecuencia']}")
    print(f"   🔍 Interpretación: {info['interpretacion']}")
    print(f"   🤖 Uso en modelo: {info['uso_modelo']}")
    
    # Verificar si existe
    path = Path(info['ubicacion']) / archivo
    if path.exists():
        df = pd.read_csv(path)
        print(f"   ✅ Estado: Existe ({len(df):,} registros)")
    else:
        print(f"   ⚠️  Estado: No encontrado")
    print()

# =============================================================================
# PARTE 2: ARCHIVOS PROCESADOS (INTERMEDIOS)
# =============================================================================
print("\n" + "=" * 100)
print("PARTE 2: ARCHIVOS PROCESADOS (INTERMEDIOS)")
print("=" * 100)

archivos_procesados = {
    'data_processed_complete.csv': {
        'ubicacion': 'data/processed/',
        'descripcion': 'Datos procesados con todas las features excepto Demanda',
        'origen': 'Merge de BD_VolTotal + BD_Clima + calendar_social',
        'registros': '15,336',
        'periodo': '2024-01-01 a 2025-09-30',
        'columnas_clave': ['timestamp_utc', 'Volumen_Total_m3', 'temperatura_promedio',
                           'es_feriado', 'temporada_turistica'],
        'proceso': 'Pipeline de data_processing.py',
        'uso': 'Base para crear data_processed_con_demanda.csv'
    },
    'data_processed_con_demanda.csv': {
        'ubicacion': 'data/processed/',
        'descripcion': 'Datos completos incluyendo variable Demanda_m3_hr',
        'origen': 'data_processed_complete + BD_Qin + cálculo de Demanda',
        'registros': '15,336 (incluye 114 con NaN)',
        'periodo': '2024-01-01 a 2025-09-30',
        'columnas_clave': ['timestamp_utc', 'Volumen_Total_m3', 'Qin', 
                           'delta_volumen_m3_hr', 'Demanda_m3_hr'],
        'proceso': 'crear_variable_demanda.py',
        'uso': 'Archivo completo con todos los registros (válidos e inválidos)'
    },
    'data_processed_demanda_valid.csv': {
        'ubicacion': 'data/processed/',
        'descripcion': 'DATASET PRINCIPAL - Solo registros válidos para entrenamiento',
        'origen': 'data_processed_con_demanda filtrado (sin NaN)',
        'registros': '15,222 (99.3% del total)',
        'periodo': '2024-01-01 a 2025-09-30',
        'columnas_clave': ['timestamp_utc', 'Demanda_m3_hr', 'hour', 'day_of_week', 
                           'month', 'is_weekend', 'feriado', 'temporada_turistica_alta'],
        'proceso': 'crear_variable_demanda.py (elimina registros con NaN)',
        'uso': '🎯 ESTE ES EL ARCHIVO QUE USA EL MODELO para entrenar y predecir'
    },
    'data_train.csv': {
        'ubicacion': 'data/processed/',
        'descripcion': 'Set de entrenamiento (70%)',
        'origen': 'Split de data_processed_demanda_valid.csv',
        'registros': '10,655',
        'periodo': '2024-01-01 a 2025-05-18 (aprox)',
        'proceso': 'entrenar_modelo_demanda.py',
        'uso': 'Entrenar el modelo XGBoost'
    },
    'data_validation.csv': {
        'ubicacion': 'data/processed/',
        'descripcion': 'Set de validación (15%)',
        'origen': 'Split de data_processed_demanda_valid.csv',
        'registros': '2,283',
        'periodo': '2025-05-19 a 2025-07-22 (aprox)',
        'proceso': 'entrenar_modelo_demanda.py',
        'uso': 'Ajustar hiperparámetros y prevenir overfitting'
    },
    'data_test.csv': {
        'ubicacion': 'data/processed/',
        'descripcion': 'Set de prueba (15%)',
        'origen': 'Split de data_processed_demanda_valid.csv',
        'registros': '2,284',
        'periodo': '2025-07-23 a 2025-09-30',
        'proceso': 'entrenar_modelo_demanda.py',
        'uso': 'Evaluar performance final del modelo (R²=0.9742, MAPE=3.02%)'
    }
}

print("\n📁 ARCHIVOS PROCESADOS (6 archivos):\n")

for i, (archivo, info) in enumerate(archivos_procesados.items(), 1):
    print(f"{i}. {archivo}")
    print(f"   📂 Ubicación: {info['ubicacion']}")
    print(f"   📝 Descripción: {info['descripcion']}")
    print(f"   🔗 Origen: {info['origen']}")
    print(f"   📊 Registros: {info['registros']}")
    print(f"   📅 Período: {info['periodo']}")
    print(f"   🔧 Proceso: {info['proceso']}")
    print(f"   💡 Uso: {info['uso']}")
    
    # Verificar si existe
    path = Path(info['ubicacion']) / archivo
    if path.exists():
        print(f"   ✅ Estado: Existe")
    else:
        print(f"   ⚠️  Estado: No encontrado")
    print()

# =============================================================================
# PARTE 3: FEATURES DEL MODELO (21 FEATURES)
# =============================================================================
print("\n" + "=" * 100)
print("PARTE 3: FEATURES DEL MODELO (21 FEATURES)")
print("=" * 100)

# Cargar features desde el archivo
features_path = Path('models/demanda/features.txt')
if features_path.exists():
    with open(features_path, 'r') as f:
        features_usadas = [line.strip() for line in f if line.strip()]
    print(f"\n✅ Features cargadas desde: {features_path}")
    print(f"   Total: {len(features_usadas)} features\n")
else:
    features_usadas = []
    print(f"\n⚠️  No se encontró {features_path}\n")

# Definición detallada de cada feature
features_detalle = {
    # TEMPORALES BÁSICAS (5)
    'hour': {
        'tipo': 'Temporal básica',
        'rango': '0-23',
        'descripcion': 'Hora del día (0=00:00, 23=23:00)',
        'interpretacion': 'Captura patrones diarios de consumo (pico 12:00, valle 04:00)',
        'importancia_modelo': '14.14%',
        'ranking': 3
    },
    'day_of_week': {
        'tipo': 'Temporal básica',
        'rango': '0-6',
        'descripcion': 'Día de la semana (0=Lunes, 6=Domingo)',
        'interpretacion': 'Diferencia consumo laboral vs fin de semana',
        'importancia_modelo': 'Media',
        'ranking': 7
    },
    'month': {
        'tipo': 'Temporal básica',
        'rango': '1-12',
        'descripcion': 'Mes del año (1=Enero, 12=Diciembre)',
        'interpretacion': 'Captura estacionalidad anual (verano vs invierno)',
        'importancia_modelo': 'Media-Alta',
        'ranking': 5
    },
    'is_weekend': {
        'tipo': 'Temporal binaria',
        'rango': '0-1',
        'descripcion': 'Si es fin de semana (1) o día laboral (0)',
        'interpretacion': 'Consumo diferente sábado/domingo vs lunes-viernes',
        'importancia_modelo': 'Baja',
        'ranking': 15
    },
    'day_of_year': {
        'tipo': 'Temporal continua',
        'rango': '1-366',
        'descripcion': 'Día del año (1=1 enero, 365=31 diciembre)',
        'interpretacion': 'Tendencia continua a lo largo del año',
        'importancia_modelo': 'Baja-Media',
        'ranking': 12
    },
    
    # CÍCLICAS (4)
    'hour_sin': {
        'tipo': 'Cíclica',
        'rango': '-1 a 1',
        'descripcion': 'Seno de la hora (sin(2π × hour/24))',
        'interpretacion': 'Representa continuidad hora 23→0, patrón circular diario',
        'importancia_modelo': 'Media',
        'ranking': 8
    },
    'hour_cos': {
        'tipo': 'Cíclica',
        'rango': '-1 a 1',
        'descripcion': 'Coseno de la hora (cos(2π × hour/24))',
        'interpretacion': 'Complemento de hour_sin para capturar ciclo completo',
        'importancia_modelo': 'Media',
        'ranking': 9
    },
    'day_of_week_sin': {
        'tipo': 'Cíclica',
        'rango': '-1 a 1',
        'descripcion': 'Seno del día de semana (sin(2π × day/7))',
        'interpretacion': 'Continuidad domingo→lunes, patrón circular semanal',
        'importancia_modelo': 'Baja',
        'ranking': 16
    },
    'day_of_week_cos': {
        'tipo': 'Cíclica',
        'rango': '-1 a 1',
        'descripcion': 'Coseno del día de semana (cos(2π × day/7))',
        'interpretacion': 'Complemento de day_of_week_sin',
        'importancia_modelo': 'Baja',
        'ranking': 17
    },
    
    # EVENTOS/CALENDARIO (2)
    'feriado': {
        'tipo': 'Binaria - Evento',
        'rango': '0-1',
        'descripcion': 'Si es feriado nacional (1) o no (0)',
        'interpretacion': 'Consumo diferente en feriados (menos industrial, más residencial)',
        'fuente': 'calendar_social_ES_COMPLETO_20240101_20250930.csv',
        'importancia_modelo': 'Baja-Media',
        'ranking': 11
    },
    'temporada_turistica_alta': {
        'tipo': 'Binaria - Evento',
        'rango': '0-1',
        'descripcion': 'Si es temporada turística alta (1) o no (0)',
        'interpretacion': 'Mayor consumo en verano/vacaciones por turismo',
        'fuente': 'calendar_social_ES_COMPLETO_20240101_20250930.csv',
        'importancia_modelo': 'Baja',
        'ranking': 18
    },
    
    # LAGS - VALORES HISTÓRICOS (4)
    'Demanda_lag_1h': {
        'tipo': 'Lag (histórico)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Demanda hace 1 hora',
        'interpretacion': 'El consumo reciente es el mejor predictor del consumo actual',
        'calculo': 'df["Demanda_m3_hr"].shift(1)',
        'importancia_modelo': 'Media-Alta',
        'ranking': 6
    },
    'Demanda_lag_2h': {
        'tipo': 'Lag (histórico)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Demanda hace 2 horas',
        'interpretacion': 'Inercia del consumo, captura tendencias de corto plazo',
        'calculo': 'df["Demanda_m3_hr"].shift(2)',
        'importancia_modelo': 'Media',
        'ranking': 10
    },
    'Demanda_lag_24h': {
        'tipo': 'Lag (histórico)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Demanda hace 24 horas (mismo hora ayer)',
        'interpretacion': 'Patrón diario muy fuerte, misma hora del día anterior',
        'calculo': 'df["Demanda_m3_hr"].shift(24)',
        'importancia_modelo': 'Media-Alta',
        'ranking': 4
    },
    'Demanda_lag_168h': {
        'tipo': 'Lag (histórico)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Demanda hace 168 horas (mismo hora hace 1 semana)',
        'interpretacion': 'Patrón semanal, mismo día y hora semana anterior',
        'calculo': 'df["Demanda_m3_hr"].shift(168)',
        'importancia_modelo': 'Baja-Media',
        'ranking': 13
    },
    
    # ROLLING STATISTICS (4)
    'Demanda_rolling_mean_6h': {
        'tipo': 'Rolling (estadística móvil)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Media móvil de las últimas 6 horas',
        'interpretacion': 'Tendencia de corto plazo, suaviza fluctuaciones',
        'calculo': 'df["Demanda_m3_hr"].rolling(window=6).mean()',
        'importancia_modelo': 'Media',
        'ranking': 14
    },
    'Demanda_rolling_std_6h': {
        'tipo': 'Rolling (estadística móvil)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Desviación estándar móvil de las últimas 6 horas',
        'interpretacion': 'Volatilidad reciente, cambios bruscos en consumo',
        'calculo': 'df["Demanda_m3_hr"].rolling(window=6).std()',
        'importancia_modelo': 'Baja',
        'ranking': 19
    },
    'Demanda_rolling_mean_24h': {
        'tipo': 'Rolling (estadística móvil)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Media móvil de las últimas 24 horas',
        'interpretacion': 'Tendencia diaria, nivel promedio del día',
        'calculo': 'df["Demanda_m3_hr"].rolling(window=24).mean()',
        'importancia_modelo': 'Baja-Media',
        'ranking': 20
    },
    'Demanda_rolling_std_24h': {
        'tipo': 'Rolling (estadística móvil)',
        'rango': 'Variable (m³/hr)',
        'descripcion': 'Desviación estándar móvil de las últimas 24 horas',
        'interpretacion': 'Variabilidad del día completo',
        'calculo': 'df["Demanda_m3_hr"].rolling(window=24).std()',
        'importancia_modelo': 'Muy Baja',
        'ranking': 21
    },
    
    # DIFERENCIAS (2)
    'Demanda_diff_1h': {
        'tipo': 'Diferencia (rate of change)',
        'rango': 'Variable (Δm³/hr)',
        'descripcion': 'Cambio en demanda respecto a hace 1 hora',
        'interpretacion': 'Velocidad de cambio, aceleración/desaceleración del consumo',
        'calculo': 'df["Demanda_m3_hr"].diff(1)',
        'importancia_modelo': '21.73% ⭐ TOP 1',
        'ranking': 1
    },
    'Demanda_diff_24h': {
        'tipo': 'Diferencia (rate of change)',
        'rango': 'Variable (Δm³/hr)',
        'descripcion': 'Cambio en demanda respecto a hace 24 horas',
        'interpretacion': 'Diferencia vs mismo hora ayer, detecta anomalías diarias',
        'calculo': 'df["Demanda_m3_hr"].diff(24)',
        'importancia_modelo': '14.46% ⭐ TOP 2',
        'ranking': 2
    },
    
    # RATIOS (1)
    'Demanda_ratio_vs_24h': {
        'tipo': 'Ratio (relación)',
        'rango': 'Variable (sin unidad)',
        'descripcion': 'Ratio actual vs hace 24 horas (actual/lag_24h)',
        'interpretacion': 'Proporción vs ayer: >1 aumentó, <1 disminuyó',
        'calculo': 'df["Demanda_m3_hr"] / df["Demanda_lag_24h"]',
        'importancia_modelo': 'Baja',
        'ranking': 14
    }
}

print("\n🎯 TOP 5 FEATURES MÁS IMPORTANTES:\n")
top_features = sorted(features_detalle.items(), 
                     key=lambda x: x[1].get('ranking', 99))[:5]

for i, (nombre, info) in enumerate(top_features, 1):
    print(f"{i}. {nombre}")
    print(f"   Importancia: {info.get('importancia_modelo', 'N/A')}")
    print(f"   Tipo: {info['tipo']}")
    print(f"   Descripción: {info['descripcion']}")
    print(f"   Interpretación: {info['interpretacion']}")
    print()

print("\n📋 TODAS LAS FEATURES POR CATEGORÍA:\n")

categorias = {
    'Temporales Básicas': ['hour', 'day_of_week', 'month', 'is_weekend', 'day_of_year'],
    'Cíclicas': ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'],
    'Eventos/Calendario': ['feriado', 'temporada_turistica_alta'],
    'Lags (Valores Históricos)': ['Demanda_lag_1h', 'Demanda_lag_2h', 
                                   'Demanda_lag_24h', 'Demanda_lag_168h'],
    'Rolling Statistics': ['Demanda_rolling_mean_6h', 'Demanda_rolling_std_6h',
                          'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h'],
    'Diferencias': ['Demanda_diff_1h', 'Demanda_diff_24h'],
    'Ratios': ['Demanda_ratio_vs_24h']
}

for categoria, features in categorias.items():
    print(f"\n📌 {categoria} ({len(features)} features):")
    for feat in features:
        if feat in features_detalle:
            info = features_detalle[feat]
            importancia = info.get('importancia_modelo', 'N/A')
            ranking = info.get('ranking', '?')
            print(f"   • {feat:<30} [Rank #{ranking:>2}] {importancia}")
        else:
            print(f"   • {feat:<30} [No info]")

# =============================================================================
# PARTE 4: ARCHIVOS GENERADOS POR EL MODELO
# =============================================================================
print("\n" + "=" * 100)
print("PARTE 4: ARCHIVOS GENERADOS POR EL MODELO")
print("=" * 100)

archivos_modelo = {
    'demanda_xgboost_model.pkl': {
        'ubicacion': 'models/demanda/',
        'descripcion': 'Modelo XGBoost entrenado (archivo pickle)',
        'tipo': 'Modelo serializado',
        'tamaño_aprox': '~2-5 MB',
        'contiene': 'Árbol de decisión XGBoost con 21 features de entrada',
        'metricas': 'R²=0.9742, RMSE=1,303 m³/hr, MAE=285 m³/hr, MAPE=3.02%',
        'generado_por': 'entrenar_modelo_demanda.py',
        'uso': '🎯 Cargado por interfaz_demanda_v1.py para hacer predicciones'
    },
    'features.txt': {
        'ubicacion': 'models/demanda/',
        'descripcion': 'Lista de features utilizadas por el modelo',
        'tipo': 'Texto plano',
        'contenido': 'Una feature por línea (21 líneas)',
        'generado_por': 'entrenar_modelo_demanda.py',
        'uso': 'La interfaz lee este archivo para saber qué features crear'
    },
    'metricas.json': {
        'ubicacion': 'models/demanda/',
        'descripcion': 'Métricas de evaluación del modelo',
        'tipo': 'JSON',
        'contenido': 'R², RMSE, MAE, MAPE para train/val/test',
        'generado_por': 'entrenar_modelo_demanda.py',
        'uso': 'Documentación de performance del modelo'
    },
    'demanda_info.json': {
        'ubicacion': 'data/processed/',
        'descripcion': 'Estadísticas descriptivas de la variable Demanda',
        'tipo': 'JSON',
        'contenido': 'Media, std, min, max, percentiles, horario de inflexión máximo/valle',
        'generado_por': 'crear_variable_demanda.py',
        'uso': 'Referencia para validación de predicciones'
    },
    'importancia_features.png': {
        'ubicacion': 'outputs/figures/',
        'descripcion': 'Gráfico de importancia de features',
        'tipo': 'Imagen PNG',
        'contenido': 'Barras horizontales mostrando importancia de cada feature',
        'generado_por': 'entrenar_modelo_demanda.py',
        'uso': 'Visualización para interpretar el modelo'
    },
    'predicciones_test.csv': {
        'ubicacion': 'outputs/',
        'descripcion': 'Predicciones del modelo en el set de prueba',
        'tipo': 'CSV',
        'columnas': ['timestamp_utc', 'Demanda_real', 'Demanda_predicha', 'error'],
        'registros': '2,284',
        'generado_por': 'entrenar_modelo_demanda.py',
        'uso': 'Análisis post-mortem de errores y validación'
    }
}

print("\n📁 ARCHIVOS GENERADOS (6 archivos):\n")

for i, (archivo, info) in enumerate(archivos_modelo.items(), 1):
    print(f"{i}. {archivo}")
    print(f"   📂 Ubicación: {info['ubicacion']}")
    print(f"   📝 Descripción: {info['descripcion']}")
    print(f"   🔧 Tipo: {info['tipo']}")
    print(f"   📦 Contenido: {info.get('contenido', info.get('contiene', 'N/A'))}")
    print(f"   ⚙️  Generado por: {info['generado_por']}")
    print(f"   💡 Uso: {info['uso']}")
    
    # Verificar si existe
    path = Path(info['ubicacion']) / archivo
    if path.exists():
        size = path.stat().st_size
        size_kb = size / 1024
        if size_kb > 1024:
            size_str = f"{size_kb/1024:.2f} MB"
        else:
            size_str = f"{size_kb:.2f} KB"
        print(f"   ✅ Estado: Existe ({size_str})")
    else:
        print(f"   ⚠️  Estado: No encontrado")
    print()

# =============================================================================
# PARTE 5: PIPELINE COMPLETO
# =============================================================================
print("\n" + "=" * 100)
print("PARTE 5: PIPELINE COMPLETO DE DATOS")
print("=" * 100)

pipeline = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DE DATOS Y MODELO                          │
└─────────────────────────────────────────────────────────────────────────────┘

📥 FASE 1: DATOS RAW (data/raw/)
   ├─ BD_VolTotal_X_Hr_m3_UTC.csv          [15,336 registros]
   ├─ BD_Qin_m3_UTC.csv                    [15,336 registros]
   ├─ BD_Clima2024a202509_UTC.csv          [15,336 registros]
   ├─ calendar_social_ES_COMPLETO...csv    [638 días]
   ├─ BD_Capacidad_89Tks_m3.csv            [89 tanques]
   └─ Vol_X_TK_Hr_m3_UTC.csv               [15,336 × 89 registros]

             │
             ▼
             
🔧 FASE 2: PROCESAMIENTO (data_processing.py)
   ├─ Merge de archivos por timestamp
   ├─ Limpieza de datos (NaN, duplicados)
   ├─ Creación de features temporales (hour, day_of_week, month)
   ├─ Features cíclicas (sin/cos)
   └─ Join con calendario (feriados, temporada turística)
   
             │
             ▼
             
💾 data_processed_complete.csv [15,336 registros]

             │
             ▼
             
🔧 FASE 3: CÁLCULO DE DEMANDA (crear_variable_demanda.py)
   ├─ Merge con BD_Qin_m3_UTC.csv
   ├─ Calcular: delta_volumen = Volumen_Total.diff()
   ├─ Calcular: Demanda = Qin - delta_volumen
   └─ Identificar registros con NaN (114 registros)
   
             │
             ├─────────────────────┬───────────────────────┐
             ▼                     ▼                       ▼
             
💾 data_processed_con_demanda.csv    demanda_info.json
   [15,336 registros - TODOS]        [Estadísticas]
   
             │
             ▼ (Filtrar NaN)
             
💾 data_processed_demanda_valid.csv
   [15,222 registros - VÁLIDOS]
   🎯 DATASET PRINCIPAL
   
             │
             ▼
             
🔧 FASE 4: FEATURE ENGINEERING (entrenar_modelo_demanda.py)
   ├─ Crear Lags: 1h, 2h, 24h, 168h
   ├─ Crear Rolling: mean/std 6h, 24h
   ├─ Crear Diferencias: diff 1h, 24h
   ├─ Crear Ratios: vs 24h
   └─ Total: 21 FEATURES
   
             │
             ▼
             
🔀 FASE 5: SPLIT TEMPORAL
   ├─ Train (70%): 10,655 registros → data_train.csv
   ├─ Val (15%):    2,283 registros → data_validation.csv
   └─ Test (15%):   2,284 registros → data_test.csv
   
             │
             ▼
             
🤖 FASE 6: ENTRENAMIENTO (XGBoost)
   ├─ Algoritmo: XGBoost Regressor
   ├─ Input: 21 features
   ├─ Output: Demanda_m3_hr (m³/hr)
   ├─ Optimización: GridSearch con validación cruzada
   └─ Early stopping para prevenir overfitting
   
             │
             ├───────────────────────┬──────────────────┬─────────────────┐
             ▼                       ▼                  ▼                 ▼
             
💾 demanda_xgboost_model.pkl  features.txt    metricas.json   predicciones_test.csv
   [Modelo entrenado]         [21 features]   [Performance]   [2,284 predicciones]
   
             │
             ▼
             
🌐 FASE 7: INTERFAZ (interfaz_demanda_v1.py)
   ├─ Puerto: 7870
   ├─ Input usuario: Fecha + Hora
   ├─ Carga modelo: demanda_xgboost_model.pkl
   ├─ Carga features: features.txt
   ├─ Carga históricos: data_processed_demanda_valid.csv
   ├─ Estima LAGs desde históricos (filtro por hora+día+mes)
   ├─ Crea 21 features
   ├─ Ejecuta predicción
   └─ Output: Demanda predicha + interpretación
   
             │
             ▼
             
📊 PREDICCIÓN FINAL
   "Para el 25/12/2025 a las 12:00:
    Demanda predicha: 18,245 m³/hr
    Horario de inflexión máximo
    +43% vs demanda promedio
    Temporada: Verano (alta demanda)"
"""

print(pipeline)

# =============================================================================
# PARTE 6: CÓMO EL MODELO ENTIENDE CADA DATO
# =============================================================================
print("\n" + "=" * 100)
print("PARTE 6: CÓMO EL MODELO INTERPRETA CADA TIPO DE DATO")
print("=" * 100)

interpretaciones = {
    'Datos Temporales': {
        'ejemplos': ['hour=12', 'day_of_week=1', 'month=7'],
        'interpretacion': 'El modelo aprende PATRONES RECURRENTES. Por ejemplo:\n'
                         '   • hour=12 → históricamente alta demanda (17,381 m³/hr promedio)\n'
                         '   • hour=4 → históricamente baja demanda (5,315 m³/hr promedio)\n'
                         '   • month=1 (enero) → verano en Chile → mayor consumo\n'
                         '   • month=7 (julio) → invierno en Chile → menor consumo',
        'mecanismo': 'XGBoost crea reglas tipo "Si hour entre 11-14 → predicción alta"\n'
                    '                        "Si hour entre 2-5 → predicción baja"'
    },
    
    'Datos Cíclicos': {
        'ejemplos': ['hour_sin=0.5', 'hour_cos=-0.866'],
        'interpretacion': 'El modelo entiende que el tiempo es CIRCULAR:\n'
                         '   • hour=23 y hour=0 están cerca (no 23 unidades de distancia)\n'
                         '   • Evita discontinuidad artificial en la medianoche\n'
                         '   • day_of_week_sin/cos: domingo(6) → lunes(0) son consecutivos',
        'mecanismo': 'Transforma valores lineales (0-23) en coordenadas polares\n'
                    '   sin/cos capturan la periodicidad natural del tiempo'
    },
    
    'Datos de Eventos': {
        'ejemplos': ['feriado=1', 'temporada_turistica_alta=1'],
        'interpretacion': 'El modelo ajusta predicción según CONTEXTO SOCIAL:\n'
                         '   • feriado=1 → menos consumo industrial, más residencial\n'
                         '   • temporada_turistica_alta=1 → mayor consumo (hoteles, turistas)\n'
                         '   • Modifica el patrón base según el día especial',
        'mecanismo': 'Features binarias (0/1) que actúan como "interruptores"\n'
                    '   Modifican la predicción base cuando están activas'
    },
    
    'Lags (Históricos)': {
        'ejemplos': ['Demanda_lag_1h=15000', 'Demanda_lag_24h=14500'],
        'interpretacion': 'El modelo usa INERCIA del sistema:\n'
                         '   • lag_1h: "Si hace 1h fue alto, probablemente siga alto"\n'
                         '   • lag_24h: "Mismo hora ayer es muy predictivo"\n'
                         '   • lag_168h: "Mismo día/hora semana pasada es referencia"\n'
                         '   • INERCIA: Los sistemas de consumo cambian gradualmente',
        'mecanismo': 'XGBoost aprende: "Si lag_1h > 15000 Y hour=12 → demanda ~16000"\n'
                    '   Combina valor histórico con contexto temporal'
    },
    
    'Rolling Statistics': {
        'ejemplos': ['rolling_mean_6h=12000', 'rolling_std_6h=2500'],
        'interpretacion': 'El modelo entiende TENDENCIAS Y VOLATILIDAD:\n'
                         '   • rolling_mean_6h: "¿Va en aumento o disminución?"\n'
                         '   • rolling_std_6h: "¿Está estable o fluctuando mucho?"\n'
                         '   • Alta volatilidad → más incertidumbre\n'
                         '   • Tendencia ascendente → próxima hora probablemente suba',
        'mecanismo': 'Suaviza ruido, captura momentum del sistema\n'
                    '   Detecta si el sistema está en "modo creciente" o "decreciente"'
    },
    
    'Diferencias': {
        'ejemplos': ['Demanda_diff_1h=-2000', 'Demanda_diff_24h=+1500'],
        'interpretacion': 'El modelo mide VELOCIDAD DE CAMBIO:\n'
                         '   • diff_1h=-2000: "Está bajando rápido (-2000 m³/hr)"\n'
                         '   • diff_1h=+3000: "Está subiendo rápido (+3000 m³/hr)"\n'
                         '   • diff_24h: "¿Hoy es día especial? Más o menos que ayer?"\n'
                         '   • TOP 1 y TOP 2 en importancia (21% y 14%)',
        'mecanismo': 'Captura aceleración/desaceleración\n'
                    '   Detecta anomalías: "Hoy 18:00 muy diferente a ayer 18:00"'
    },
    
    'Ratios': {
        'ejemplos': ['ratio_vs_24h=1.15', 'ratio_vs_24h=0.87'],
        'interpretacion': 'El modelo compara PROPORCIONALMENTE:\n'
                         '   • ratio=1.15: "Hoy 15% más alto que ayer (misma hora)"\n'
                         '   • ratio=0.87: "Hoy 13% más bajo que ayer"\n'
                         '   • Normaliza: Un aumento de 2000 es grande si base es 5000\n'
                         '                pero pequeño si base es 20000',
        'mecanismo': 'Captura cambios relativos, no absolutos\n'
                    '   Ajusta por el nivel base de demanda'
    }
}

for categoria, info in interpretaciones.items():
    print(f"\n🔍 {categoria}:")
    print(f"   Ejemplos: {', '.join(info['ejemplos'])}")
    print(f"\n   📖 Interpretación:\n{info['interpretacion']}")
    print(f"\n   ⚙️  Mecanismo:\n   {info['mecanismo']}")
    print()

# =============================================================================
# PARTE 7: RESUMEN EJECUTIVO
# =============================================================================
print("\n" + "=" * 100)
print("RESUMEN EJECUTIVO")
print("=" * 100)

resumen = """
📊 ARCHIVOS DE ENTRADA: 6 archivos RAW
   └─ Más importante: BD_Qin_m3_UTC.csv + BD_VolTotal_X_Hr_m3_UTC.csv
      (Para calcular Demanda = Qin - ΔVolumen)

🔧 ARCHIVOS PROCESADOS: 6 archivos intermedios
   └─ Más importante: data_processed_demanda_valid.csv (15,222 registros)
      🎯 ESTE ES EL QUE USA EL MODELO

🤖 MODELO: XGBoost Regressor
   ├─ Input: 21 features
   ├─ Output: Demanda_m3_hr (m³/hr)
   ├─ Performance: R²=0.9742, MAPE=3.02%
   └─ Archivos generados: 6 (modelo.pkl, features.txt, metricas.json, etc.)

🎯 FEATURES: 21 en total
   ├─ Top 1: Demanda_diff_1h (21.73%)        → Velocidad de cambio
   ├─ Top 2: Demanda_diff_24h (14.46%)       → Comparación vs ayer
   ├─ Top 3: hour (14.14%)                   → Hora del día
   ├─ Top 4: Demanda_lag_24h (%)             → Mismo hora ayer
   └─ Top 5: month (%)                       → Estacionalidad

💡 INTERPRETACIÓN DEL MODELO:
   El modelo combina:
   1️⃣ Patrones temporales (hora/día/mes) → cuándo ocurre
   2️⃣ Valores históricos (lags) → qué pasó antes
   3️⃣ Tendencias (rolling, diff) → hacia dónde va
   4️⃣ Contexto social (feriados) → eventos especiales
   
   Para predecir: "¿Cuánta agua se consumirá en la próxima hora?"

🌐 INTERFAZ: interfaz_demanda_v1.py (Puerto 7870)
   Input: Fecha + Hora
   Output: Demanda predicha (m³/hr) + interpretación + contexto

📈 VALIDACIÓN:
   ✅ Patrones correctos: Pico 12:00, Valle 04:00
   ✅ Estacionalidad correcta: Verano > Invierno (14.9%)
   ✅ Métricas excelentes: R²=0.9742, MAPE=3.02%
   ✅ 99.3% de datos utilizados (15,222/15,336)
"""

print(resumen)

print("\n" + "=" * 100)
print("FIN DE LA DOCUMENTACIÓN")
print("=" * 100)
