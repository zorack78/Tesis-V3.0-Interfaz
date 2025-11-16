"""
Sistema de Entrenamiento: Modelos Forecasting vs Explicativo
=====================================================

MODELO A - Forecasting (sin Qin actual):
  - Para predicción 24h-72h adelante
  - Solo usa: clima, calendario, lags de demanda
  - NO usa Qin(t) para evitar leakage causal

MODELO B - Explicativo (con Qin):
  - Para análisis histórico y validación
  - Incluye: Qin(t), Qin_delta, regímenes operacionales
  - Ayuda a entender decisiones operacionales

Autor: Sistema de predicción demanda agua potable Gran Valparaíso
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime

# Configuración
np.random.seed(42)
sns.set_style("whitegrid")


def cargar_features_prometedoras():
    """Carga las 31 features prometedoras identificadas"""
    features_df = pd.read_csv('outputs/features_prometedoras_Q_net.csv')
    return features_df['feature'].tolist()


def cargar_datos():
    """Carga dataset completo con todas las features"""
    print("=" * 100)
    print("CARGANDO DATOS")
    print("=" * 100)
    
    df = pd.read_csv('data/processed/dataset_features_completo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ordenar por timestamp (CRÍTICO para series temporales)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nDataset cargado: {df.shape}")
    print(f"Rango temporal: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Target (Q_net_m3h): media={df['Q_net_m3h'].mean():.1f}, "
          f"std={df['Q_net_m3h'].std():.1f}")
    
    return df


def ingenieria_features_adicionales(df):
    """Crea features adicionales basadas en hallazgos"""
    print("\n" + "=" * 100)
    print("INGENIERÍA DE FEATURES ADICIONALES")
    print("=" * 100)
    
    df = df.copy()
    
    # ===== UMBRALES CATEGÓRICOS =====
    print("\n1. Creando categorías basadas en umbrales descubiertos...")
    
    # Temperatura (umbrales: 10.5°C, 16.3°C)
    df['temp_nivel'] = pd.cut(
        df['clima_temp_c'],
        bins=[-np.inf, 10.5, 16.3, np.inf],
        labels=['Frio', 'Normal', 'Calor']
    )
    
    # Qin (umbrales: 10,998, 12,614)
    df['qin_nivel'] = pd.cut(
        df['sist_Qin_m3h'],
        bins=[-np.inf, 10998, 12614, np.inf],
        labels=['Bajo', 'Normal', 'Alto']
    )
    
    # Hora del día (períodos críticos)
    df['periodo_dia'] = pd.cut(
        df['timestamp'].dt.hour,
        bins=[0, 6, 9, 18, 22, 24],
        labels=['Madrugada', 'Manana_critica', 'Dia', 'Noche', 'Noche_tardia'],
        include_lowest=True
    )
    
    # Horas bisagra (7-9h, 22-23h)
    df['es_hora_bisagra'] = df['timestamp'].dt.hour.isin([7, 8, 9, 22, 23]).astype(int)
    
    print(f"   temp_nivel: {df['temp_nivel'].value_counts().to_dict()}")
    print(f"   qin_nivel: {df['qin_nivel'].value_counts().to_dict()}")
    print(f"   periodo_dia: {df['periodo_dia'].value_counts().to_dict()}")
    
    # ===== FEATURES DE QIN (evitar leakage causal) =====
    print("\n2. Creando features de Qin (lags y deltas)...")
    
    # Lags (valores pasados - no tienen leakage)
    df['Qin_lag_1h'] = df['sist_Qin_m3h'].shift(1)
    df['Qin_lag_3h'] = df['sist_Qin_m3h'].shift(3)
    df['Qin_lag_6h'] = df['sist_Qin_m3h'].shift(6)
    
    # Deltas (cambios - señal de decisión operacional)
    df['Qin_delta_1h'] = df['sist_Qin_m3h'] - df['sist_Qin_m3h'].shift(1)
    df['Qin_delta_3h'] = df['sist_Qin_m3h'] - df['sist_Qin_m3h'].shift(3)
    df['Qin_delta_6h'] = df['sist_Qin_m3h'] - df['sist_Qin_m3h'].shift(6)
    
    # Variabilidad (detecta régimen estable)
    df['Qin_std_24h'] = df['sist_Qin_m3h'].rolling(window=24, min_periods=12).std()
    df['Qin_cambios_24h'] = (df['Qin_delta_1h'].abs() > 100).rolling(window=24).sum()
    
    print(f"   Qin_delta_1h: media={df['Qin_delta_1h'].mean():.1f}, "
          f"std={df['Qin_delta_1h'].std():.1f}")
    print(f"   Qin_std_24h: media={df['Qin_std_24h'].mean():.1f}")
    
    # ===== DETECCIÓN DE REGÍMENES OPERACIONALES =====
    print("\n3. Detectando regímenes operacionales...")
    
    # Régimen 1: Equilibrio estable (Qin constante)
    df['regimen_equilibrio'] = ((df['Qin_std_24h'] < 200) & 
                                 (df['Qin_cambios_24h'] < 3)).astype(int)
    
    # Régimen 2: Recuperación anticipada (reduciendo Qin mientras recupera)
    df['regimen_recuperacion_anticipada'] = (
        (df['Qin_delta_1h'] < -50) & 
        (df['Q_net_m3h'] > 500)
    ).astype(int)
    
    # Régimen 3: Déficit persistente (Qin alto + demanda sostenida)
    df['regimen_deficit'] = (
        (df['sist_Qin_m3h'] > 12614) & 
        (df['Q_net_m3h'].rolling(window=6).mean() < -1000)
    ).astype(int)
    
    print(f"   Equilibrio: {df['regimen_equilibrio'].sum()} horas "
          f"({df['regimen_equilibrio'].mean()*100:.1f}%)")
    print(f"   Recuperación anticipada: {df['regimen_recuperacion_anticipada'].sum()} horas")
    print(f"   Déficit persistente: {df['regimen_deficit'].sum()} horas")
    
    # ===== INTERACCIONES CLAVE =====
    print("\n4. Creando interacciones clave...")
    
    # Temperatura × hora (patrón diario varía con temperatura)
    df['temp_x_hora'] = df['clima_temp_c'] * df['timestamp'].dt.hour
    
    # Delta temperatura × hora (cambios térmicos en horas críticas)
    if 'clima_temp_delta_6h' in df.columns:
        df['delta_temp_6h_x_hora'] = df['clima_temp_delta_6h'] * df['timestamp'].dt.hour
    
    # Temperatura × fin de semana
    if 'cal_es_fin_de_semana' in df.columns:
        df['temp_x_finde'] = df['clima_temp_c'] * df['cal_es_fin_de_semana']
    
    # Qin × temperatura (producción ajustada a clima)
    df['qin_x_temp'] = df['sist_Qin_m3h'] * df['clima_temp_c']
    
    print("   Interacciones creadas: temp_x_hora, delta_temp_6h_x_hora, temp_x_finde, qin_x_temp")
    
    # ===== FEATURES TEMPORALES BÁSICAS =====
    print("\n5. Creando features temporales básicas...")
    
    df['hora'] = df['timestamp'].dt.hour
    df['dia_semana'] = df['timestamp'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['mes'] = df['timestamp'].dt.month
    df['es_fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    print(f"   Features temporales: hora, dia_semana, mes, es_fin_de_semana")
    
    # Verificar NaNs generados por shifts y rolling
    n_antes = len(df)
    
    # Analizar NaNs por columna
    nan_counts = df.isnull().sum()
    cols_con_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    
    if len(cols_con_nan) > 0:
        print(f"\n   Columnas con NaN (Top 10):")
        for col, count in cols_con_nan.head(10).items():
            print(f"      {col}: {count} NaN ({count/len(df)*100:.1f}%)")
    
    # ESTRATEGIA: Solo eliminar filas donde features CRÍTICAS tienen NaN
    # No eliminar por features rolling muy largos (ej: 168h)
    
    # Features críticas que NO pueden tener NaN
    features_criticas = [
        'Q_net_m3h', 'clima_temp_c', 'sist_Qin_m3h',
        'hora', 'dia_semana', 'mes',
        'temp_nivel', 'qin_nivel', 'periodo_dia'
    ]
    
    # Verificar que existen
    features_criticas = [f for f in features_criticas if f in df.columns]
    
    # Eliminar solo filas con NaN en features críticas
    df = df.dropna(subset=features_criticas)
    
    n_despues = len(df)
    print(f"\n   Registros eliminados por NaN en features críticas: {n_antes - n_despues}")
    print(f"   Registros finales: {n_despues}")
    
    # Para otras features con NaN, rellenar con estrategia inteligente
    # Lags: forward fill (último valor conocido)
    lag_cols = [c for c in df.columns if 'lag' in c.lower()]
    for col in lag_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill')
    
    # Rolling: forward fill también (último valor calculado)
    rolling_cols = [c for c in df.columns if any(x in c.lower() for x in ['std', 'cambios', 'ema', 'rolling'])]
    for col in rolling_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill')
    
    # Regímenes: rellenar con 0 (no está en régimen)
    regimen_cols = [c for c in df.columns if 'regimen' in c.lower()]
    for col in regimen_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    
    # Interacciones: si algún componente es NaN, resultado es 0
    interaccion_cols = [c for c in df.columns if '_x_' in c]
    for col in interaccion_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    
    # Verificar que no quedan NaNs críticos
    nan_restantes = df.isnull().sum().sum()
    if nan_restantes > 0:
        print(f"\n   ⚠️ Advertencia: {nan_restantes} NaN restantes")
        
        # Rellenar NaNs por tipo de columna
        for col in df.columns:
            if df[col].isnull().any():
                # Categóricas: rellenar con moda o 'Desconocido'
                if df[col].dtype.name == 'category' or df[col].dtype == 'object':
                    if col in ['fecha_feriado_observado', 'nombre_feriado', 
                               'fecha_feriado_oficial']:
                        # Estas son textuales, dejarlas como están (no se usan)
                        continue
                    else:
                        # Usar moda
                        moda = df[col].mode()
                        if len(moda) > 0:
                            df[col] = df[col].fillna(moda[0])
                
                # Numéricas: rellenar con 0
                elif np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(0)
    
    print(f"   ✅ Dataset limpio: {len(df)} registros")
    
    return df


def preparar_features_modelo_a(df, features_prometedoras):
    """
    Modelo A - FORECASTING (sin Qin actual)
    Para predicción 24-72h adelante
    """
    print("\n" + "=" * 100)
    print("MODELO A - FORECASTING (sin Qin actual)")
    print("=" * 100)
    
    # Features base prometedoras (excluir las que tienen Qin en nombre)
    features_sin_qin = [f for f in features_prometedoras 
                        if 'Qin' not in f and 'qin' not in f.lower()]
    
    # Agregar features creadas
    features_adicionales = [
        # Temporales
        'hora', 'dia_semana', 'mes', 'es_fin_de_semana',
        
        # Umbrales categóricos (one-hot)
        'temp_nivel', 'periodo_dia', 'es_hora_bisagra',
        
        # Interacciones (sin Qin)
        'temp_x_hora', 'delta_temp_6h_x_hora', 'temp_x_finde',
        
        # Regímenes (sin usar Qin directamente, solo patrones)
        'regimen_equilibrio'
    ]
    
    # Combinar
    features_modelo_a = list(set(features_sin_qin + features_adicionales))
    
    # Verificar que existen
    features_modelo_a = [f for f in features_modelo_a if f in df.columns]
    
    print(f"\nFeatures seleccionadas: {len(features_modelo_a)}")
    print("\nCategorías:")
    print(f"  - Prometedoras sin Qin: {len(features_sin_qin)}")
    print(f"  - Temporales: 4")
    print(f"  - Umbrales: 3")
    print(f"  - Interacciones: 3")
    print(f"  - Regímenes: 1")
    
    return features_modelo_a


def preparar_features_modelo_b(df, features_modelo_a):
    """
    Modelo B - EXPLICATIVO (con Qin)
    Para análisis histórico y validación
    """
    print("\n" + "=" * 100)
    print("MODELO B - EXPLICATIVO (con Qin)")
    print("=" * 100)
    
    # Partir del Modelo A y agregar features de Qin
    features_qin = [
        'sist_Qin_m3h',  # Qin actual
        'Qin_lag_1h', 'Qin_lag_3h', 'Qin_lag_6h',  # Lags
        'Qin_delta_1h', 'Qin_delta_3h', 'Qin_delta_6h',  # Deltas
        'Qin_std_24h', 'Qin_cambios_24h',  # Variabilidad
        'qin_nivel',  # Categórico
        'qin_x_temp',  # Interacción
        'regimen_recuperacion_anticipada',  # Regímenes con Qin
        'regimen_deficit'
    ]
    
    # Verificar que existen
    features_qin = [f for f in features_qin if f in df.columns]
    
    features_modelo_b = features_modelo_a + features_qin
    
    print(f"\nFeatures seleccionadas: {len(features_modelo_b)}")
    print(f"  - Del Modelo A: {len(features_modelo_a)}")
    print(f"  - Qin adicionales: {len(features_qin)}")
    
    return features_modelo_b


def one_hot_encoding(df, features, categorical_cols):
    """Aplica one-hot encoding a columnas categóricas"""
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in features:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            features.remove(col)
            features.extend(dummies.columns.tolist())
    
    return df_encoded, features


def split_temporal(df, train_ratio=0.70, val_ratio=0.15):
    """Split temporal (no aleatorio) para series de tiempo"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    print(f"\nSplit temporal:")
    print(f"  Train: {len(train)} ({len(train)/n*100:.1f}%) - "
          f"{train['timestamp'].min()} → {train['timestamp'].max()}")
    print(f"  Val:   {len(val)} ({len(val)/n*100:.1f}%) - "
          f"{val['timestamp'].min()} → {val['timestamp'].max()}")
    print(f"  Test:  {len(test)} ({len(test)/n*100:.1f}%) - "
          f"{test['timestamp'].min()} → {test['timestamp'].max()}")
    
    return train, val, test


def entrenar_xgboost(X_train, y_train, X_val, y_val, nombre_modelo):
    """Entrena modelo XGBoost con early stopping"""
    print(f"\n{'=' * 100}")
    print(f"ENTRENANDO: {nombre_modelo}")
    print(f"{'=' * 100}")
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    print(f"\nHiperparámetros:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    modelo = xgb.XGBRegressor(**params)
    
    print(f"\nEntrenando con early stopping...")
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    best_iteration = modelo.best_iteration if modelo.best_iteration else modelo.n_estimators
    print(f"\nMejor iteración: {best_iteration}")
    
    # Obtener métricas finales
    rmse_train = np.sqrt(mean_squared_error(y_train, modelo.predict(X_train)))
    rmse_val = np.sqrt(mean_squared_error(y_val, modelo.predict(X_val)))
    print(f"RMSE train: {rmse_train:.2f}")
    print(f"RMSE val: {rmse_val:.2f}")
    
    return modelo


def evaluar_modelo(modelo, X_test, y_test, test_df, nombre_modelo):
    """Evaluación completa del modelo"""
    print(f"\n{'=' * 100}")
    print(f"EVALUACIÓN: {nombre_modelo}")
    print(f"{'=' * 100}")
    
    y_pred = modelo.predict(X_test)
    
    # Métricas globales
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE (cuidado con divisiones por cero)
    mask_no_cero = y_test != 0
    mape = np.mean(np.abs((y_test[mask_no_cero] - y_pred[mask_no_cero]) / 
                          y_test[mask_no_cero])) * 100
    
    print(f"\nMÉTRICAS GLOBALES:")
    print(f"  RMSE: {rmse:.2f} m³/hr")
    print(f"  MAE:  {mae:.2f} m³/hr")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Métricas por segmento
    test_df = test_df.copy()
    test_df['prediccion'] = y_pred
    test_df['residuo'] = y_test.values - y_pred
    test_df['residuo_abs'] = np.abs(test_df['residuo'])
    
    print(f"\nMÉTRICAS POR TEMPERATURA:")
    for nivel in ['Frio', 'Normal', 'Calor']:
        if nivel in test_df['temp_nivel'].values:
            mask = test_df['temp_nivel'] == nivel
            mae_seg = test_df[mask]['residuo_abs'].mean()
            n_seg = mask.sum()
            print(f"  {nivel:10s}: MAE={mae_seg:7.2f} m³/hr  (n={n_seg:,})")
    
    print(f"\nMÉTRICAS POR PERÍODO DEL DÍA:")
    for periodo in test_df['periodo_dia'].unique():
        if pd.notna(periodo):
            mask = test_df['periodo_dia'] == periodo
            mae_seg = test_df[mask]['residuo_abs'].mean()
            n_seg = mask.sum()
            print(f"  {periodo:20s}: MAE={mae_seg:7.2f} m³/hr  (n={n_seg:,})")
    
    print(f"\nMÉTRICAS POR FIN DE SEMANA:")
    for es_finde in [0, 1]:
        mask = test_df['es_fin_de_semana'] == es_finde
        mae_seg = test_df[mask]['residuo_abs'].mean()
        n_seg = mask.sum()
        tipo = "Fin de semana" if es_finde else "Día laboral"
        print(f"  {tipo:20s}: MAE={mae_seg:7.2f} m³/hr  (n={n_seg:,})")
    
    # Feature importance
    print(f"\nTOP 20 FEATURES MÁS IMPORTANTES:")
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    
    metricas = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'n_test': len(y_test)
    }
    
    return metricas, importance_df, test_df


def comparar_modelos(metricas_a, metricas_b):
    """Compara Modelo A vs Modelo B"""
    print(f"\n{'=' * 100}")
    print(f"COMPARACIÓN MODELO A (Forecasting) vs MODELO B (Explicativo)")
    print(f"{'=' * 100}")
    
    print(f"\n{'Métrica':<10s} {'Modelo A':<15s} {'Modelo B':<15s} {'Diferencia':<15s} {'Mejora %':<10s}")
    print("-" * 100)
    
    for metrica in ['rmse', 'mae', 'r2', 'mape']:
        val_a = metricas_a[metrica]
        val_b = metricas_b[metrica]
        dif = val_b - val_a
        
        if metrica == 'r2':
            mejora = (val_b - val_a) / abs(val_a) * 100
            print(f"{metrica.upper():<10s} {val_a:<15.4f} {val_b:<15.4f} "
                  f"{dif:+15.4f} {mejora:+10.2f}%")
        else:
            mejora = (val_a - val_b) / abs(val_a) * 100  # Menor es mejor
            print(f"{metrica.upper():<10s} {val_a:<15.2f} {val_b:<15.2f} "
                  f"{dif:+15.2f} {mejora:+10.2f}%")
    
    print("\nINTERPRETACIÓN:")
    mejora_rmse = (metricas_a['rmse'] - metricas_b['rmse']) / metricas_a['rmse'] * 100
    
    if mejora_rmse > 15:
        print(f"  ✅ Modelo B es SIGNIFICATIVAMENTE mejor ({mejora_rmse:.1f}% mejora)")
        print(f"     → Qin aporta información valiosa NO capturada por clima/calendario")
        print(f"     → Para forecasting largo plazo, considerar predecir Qin primero")
    elif mejora_rmse > 5:
        print(f"  ⚠️  Modelo B es MODERADAMENTE mejor ({mejora_rmse:.1f}% mejora)")
        print(f"     → Qin aporta algo de información adicional")
        print(f"     → Modelo A es suficiente para forecasting")
    else:
        print(f"  ✅ Modelos SIMILARES ({mejora_rmse:.1f}% diferencia)")
        print(f"     → Clima/calendario capturan la mayoría de la varianza")
        print(f"     → Qin es redundante, solo refleja lo que clima predice")


def guardar_modelo(modelo, features, metricas, importance_df, carpeta, nombre):
    """Guarda modelo y metadatos"""
    os.makedirs(carpeta, exist_ok=True)
    
    # Guardar modelo
    modelo_path = os.path.join(carpeta, f'{nombre}.pkl')
    joblib.dump(modelo, modelo_path)
    print(f"\n✅ Modelo guardado: {modelo_path}")
    
    # Guardar features
    features_path = os.path.join(carpeta, 'features.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    print(f"✅ Features guardadas: {features_path}")
    
    # Guardar métricas
    metricas_path = os.path.join(carpeta, 'metricas.json')
    with open(metricas_path, 'w') as f:
        json.dump(metricas, f, indent=2)
    print(f"✅ Métricas guardadas: {metricas_path}")
    
    # Guardar feature importance
    importance_path = os.path.join(carpeta, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"✅ Feature importance guardada: {importance_path}")
    
    # Guardar umbrales
    umbrales = {
        'temp_frio': 10.5,
        'temp_calor': 16.3,
        'qin_bajo': 10998,
        'qin_alto': 12614,
        'hora_max_demanda': 12,
        'hora_max_recuperacion': 4
    }
    umbrales_path = os.path.join(carpeta, 'umbrales.pkl')
    joblib.dump(umbrales, umbrales_path)
    print(f"✅ Umbrales guardados: {umbrales_path}")


def main():
    """Pipeline principal de entrenamiento"""
    print("=" * 100)
    print("SISTEMA DE ENTRENAMIENTO - MODELOS FORECASTING Y EXPLICATIVO")
    print("=" * 100)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Cargar datos
    features_prometedoras = cargar_features_prometedoras()
    print(f"\nFeatures prometedoras cargadas: {len(features_prometedoras)}")
    
    df = cargar_datos()
    
    # 2. Ingeniería de features
    df = ingenieria_features_adicionales(df)
    
    # 3. Preparar features para ambos modelos
    features_modelo_a = preparar_features_modelo_a(df, features_prometedoras)
    features_modelo_b = preparar_features_modelo_b(df, features_modelo_a)
    
    # 4. One-hot encoding (aplicar a TODAS las categóricas desde el inicio)
    # Esto crea columnas para temp_nivel, qin_nivel, periodo_dia
    categorical_cols = ['temp_nivel', 'qin_nivel', 'periodo_dia']
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    # Actualizar listas de features para incluir dummies
    features_modelo_a_encoded = []
    for f in features_modelo_a:
        if f in categorical_cols:
            # Reemplazar categórica por sus dummies
            dummies_cols = [c for c in df_encoded.columns if c.startswith(f + '_')]
            features_modelo_a_encoded.extend(dummies_cols)
        else:
            features_modelo_a_encoded.append(f)
    
    features_modelo_b_encoded = []
    for f in features_modelo_b:
        if f in categorical_cols:
            # Reemplazar categórica por sus dummies
            dummies_cols = [c for c in df_encoded.columns if c.startswith(f + '_')]
            features_modelo_b_encoded.extend(dummies_cols)
        else:
            features_modelo_b_encoded.append(f)
    
    # 5. Verificar que todas las features existen en df_encoded
    print(f"\n{'=' * 100}")
    print("VERIFICANDO FEATURES DISPONIBLES")
    print(f"{'=' * 100}")
    
    missing_a = [f for f in features_modelo_a_encoded if f not in df_encoded.columns]
    missing_b = [f for f in features_modelo_b_encoded if f not in df_encoded.columns]
    
    if missing_a:
        print(f"\n⚠️ Features faltantes en Modelo A: {missing_a[:10]}")
    if missing_b:
        print(f"\n⚠️ Features faltantes en Modelo B: {missing_b[:10]}")
    
    # Filtrar solo features que existen
    features_modelo_a_encoded = [f for f in features_modelo_a_encoded 
                                  if f in df_encoded.columns]
    features_modelo_b_encoded = [f for f in features_modelo_b_encoded 
                                  if f in df_encoded.columns]
    
    print(f"\nFeatures finales Modelo A: {len(features_modelo_a_encoded)}")
    print(f"Features finales Modelo B: {len(features_modelo_b_encoded)}")
    
    # 6. Split temporal
    train, val, test = split_temporal(df_encoded)
    
    # 7. Preparar datasets
    target = 'Q_net_m3h'
    
    # Modelo A
    X_train_a = train[features_modelo_a_encoded]
    y_train_a = train[target]
    X_val_a = val[features_modelo_a_encoded]
    y_val_a = val[target]
    X_test_a = test[features_modelo_a_encoded]
    y_test_a = test[target]
    
    # Modelo B
    X_train_b = train[features_modelo_b_encoded]
    y_train_b = train[target]
    X_val_b = val[features_modelo_b_encoded]
    y_val_b = val[target]
    X_test_b = test[features_modelo_b_encoded]
    y_test_b = test[target]
    
    # 7. Entrenar Modelo A (Forecasting)
    modelo_a = entrenar_xgboost(X_train_a, y_train_a, X_val_a, y_val_a, 
                                 "MODELO A - Forecasting (sin Qin)")
    
    # 8. Entrenar Modelo B (Explicativo)
    modelo_b = entrenar_xgboost(X_train_b, y_train_b, X_val_b, y_val_b, 
                                 "MODELO B - Explicativo (con Qin)")
    
    # 9. Evaluar Modelo A
    metricas_a, importance_a, test_df_a = evaluar_modelo(
        modelo_a, X_test_a, y_test_a, test, "MODELO A - Forecasting"
    )
    
    # 10. Evaluar Modelo B
    metricas_b, importance_b, test_df_b = evaluar_modelo(
        modelo_b, X_test_b, y_test_b, test, "MODELO B - Explicativo"
    )
    
    # 11. Comparar modelos
    comparar_modelos(metricas_a, metricas_b)
    
    # 12. Guardar modelos
    print(f"\n{'=' * 100}")
    print("GUARDANDO MODELOS Y METADATOS")
    print("=" * 100)
    
    guardar_modelo(modelo_a, features_modelo_a_encoded, metricas_a, importance_a,
                   'models/forecasting', 'modelo_forecasting_xgboost')
    
    guardar_modelo(modelo_b, features_modelo_b_encoded, metricas_b, importance_b,
                   'models/explicativo', 'modelo_explicativo_xgboost')
    
    print(f"\n{'=' * 100}")
    print("ENTRENAMIENTO COMPLETADO!")
    print("=" * 100)
    print("\nModelos disponibles:")
    print("  1. models/forecasting/modelo_forecasting_xgboost.pkl")
    print("     → Usar para predicción 24h-72h adelante")
    print("\n  2. models/explicativo/modelo_explicativo_xgboost.pkl")
    print("     → Usar para análisis histórico y validación")


if __name__ == '__main__':
    main()
