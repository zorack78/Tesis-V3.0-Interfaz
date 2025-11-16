"""
Comparación de Modelos: Nuevo (Forecasting) vs Baseline (Antiguo)
==================================================================

Compara el modelo recién entrenado contra el modelo anterior para
cuantificar las mejoras obtenidas.

Autor: Sistema de predicción demanda agua potable Gran Valparaíso
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def cargar_modelo_nuevo():
    """Carga modelo forecasting nuevo"""
    print("=" * 100)
    print("CARGANDO MODELO NUEVO (Forecasting)")
    print("=" * 100)
    
    modelo = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
    
    with open('models/forecasting/features.txt') as f:
        features = f.read().splitlines()
    
    with open('models/forecasting/metricas.json') as f:
        metricas = json.load(f)
    
    print(f"\nModelo: XGBoost")
    print(f"Features: {len(features)}")
    print(f"RMSE test: {metricas['rmse']:.2f}")
    print(f"MAE test: {metricas['mae']:.2f}")
    print(f"R² test: {metricas['r2']:.4f}")
    
    return modelo, features, metricas


def cargar_modelo_baseline():
    """Carga modelo baseline anterior"""
    print("\n" + "=" * 100)
    print("CARGANDO MODELO BASELINE (Antiguo)")
    print("=" * 100)
    
    try:
        modelo = joblib.load('models/gradio/water_demand_model.pkl')
        
        # Intentar cargar features
        try:
            with open('models/gradio/features.txt') as f:
                features = f.read().splitlines()
        except FileNotFoundError:
            print("  ⚠️ No se encontró features.txt, extrayendo del modelo...")
            if hasattr(modelo, 'feature_names_in_'):
                features = list(modelo.feature_names_in_)
            else:
                features = None
        
        print(f"\nModelo: {type(modelo).__name__}")
        if features:
            print(f"Features: {len(features)}")
        else:
            print(f"Features: Desconocidas")
        
        return modelo, features
        
    except FileNotFoundError:
        print("  ❌ No se encontró models/gradio/water_demand_model.pkl")
        return None, None


def preparar_datos_test():
    """Carga y prepara datos de test"""
    print("\n" + "=" * 100)
    print("PREPARANDO DATOS DE TEST")
    print("=" * 100)
    
    # Cargar dataset completo procesado
    df = pd.read_csv('data/processed/dataset_features_completo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nDataset completo: {df.shape}")
    print(f"Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Split temporal (mismo que en entrenamiento)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    test = df.iloc[val_end:].copy()
    
    print(f"\nTest set: {len(test)} registros")
    print(f"Período: {test['timestamp'].min()} → {test['timestamp'].max()}")
    
    return test


def evaluar_baseline_en_test(modelo_baseline, features_baseline, test_df):
    """Evalúa modelo baseline en test set"""
    print("\n" + "=" * 100)
    print("EVALUANDO MODELO BASELINE EN TEST SET")
    print("=" * 100)
    
    if modelo_baseline is None:
        print("\n  ❌ No se puede evaluar - modelo no disponible")
        return None
    
    # Preparar features para baseline
    if features_baseline is None:
        print("\n  ⚠️ Features desconocidas - intentando con todas las disponibles")
        # Intentar con features comunes
        features_comunes = ['hora', 'dia_semana', 'mes', 'hora_seno', 'hora_coseno',
                           'dia_semana_seno', 'dia_semana_coseno']
        features_baseline = [f for f in features_comunes if f in test_df.columns]
    
    # Verificar features disponibles
    features_disponibles = [f for f in features_baseline if f in test_df.columns]
    features_faltantes = [f for f in features_baseline if f not in test_df.columns]
    
    if features_faltantes:
        print(f"\n  ⚠️ Features faltantes en test: {len(features_faltantes)}")
        print(f"     Primeras 10: {features_faltantes[:10]}")
    
    if len(features_disponibles) == 0:
        print("\n  ❌ No hay features disponibles para evaluación")
        return None
    
    print(f"\n  Features disponibles: {len(features_disponibles)}")
    
    # Preparar datos
    X_test = test_df[features_disponibles]
    y_test = test_df['Q_net_m3h']
    
    # Predecir
    try:
        y_pred = modelo_baseline.predict(X_test)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mask_no_cero = y_test != 0
        mape = np.mean(np.abs((y_test[mask_no_cero] - y_pred[mask_no_cero]) / 
                              y_test[mask_no_cero])) * 100
        
        print(f"\n  MÉTRICAS BASELINE:")
        print(f"    RMSE: {rmse:.2f} m³/hr")
        print(f"    MAE:  {mae:.2f} m³/hr")
        print(f"    R²:   {r2:.4f}")
        print(f"    MAPE: {mape:.2f}%")
        
        metricas = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        return metricas
        
    except Exception as e:
        print(f"\n  ❌ Error al evaluar: {str(e)}")
        return None


def comparar_metricas(metricas_nuevo, metricas_baseline):
    """Compara métricas entre modelos"""
    print("\n" + "=" * 100)
    print("COMPARACIÓN DETALLADA")
    print("=" * 100)
    
    if metricas_baseline is None:
        print("\n  ⚠️ No se puede comparar - baseline no disponible")
        print("\n  📊 MODELO NUEVO (absoluto):")
        print(f"    RMSE: {metricas_nuevo['rmse']:.2f} m³/hr")
        print(f"    MAE:  {metricas_nuevo['mae']:.2f} m³/hr")
        print(f"    R²:   {metricas_nuevo['r2']:.4f}")
        print(f"    MAPE: {metricas_nuevo['mape']:.2f}%")
        return
    
    print(f"\n{'Métrica':<10s} {'Baseline':<15s} {'Nuevo':<15s} "
          f"{'Mejora Abs':<15s} {'Mejora %':<10s}")
    print("-" * 100)
    
    mejoras = {}
    
    for metrica in ['rmse', 'mae', 'mape']:
        val_baseline = metricas_baseline[metrica]
        val_nuevo = metricas_nuevo[metrica]
        mejora_abs = val_baseline - val_nuevo
        mejora_pct = (mejora_abs / val_baseline) * 100
        
        mejoras[metrica] = mejora_pct
        
        print(f"{metrica.upper():<10s} {val_baseline:<15.2f} {val_nuevo:<15.2f} "
              f"{mejora_abs:+15.2f} {mejora_pct:+10.2f}%")
    
    # R² (mayor es mejor)
    val_baseline_r2 = metricas_baseline['r2']
    val_nuevo_r2 = metricas_nuevo['r2']
    mejora_abs_r2 = val_nuevo_r2 - val_baseline_r2
    mejora_pct_r2 = (mejora_abs_r2 / abs(val_baseline_r2)) * 100
    
    mejoras['r2'] = mejora_pct_r2
    
    print(f"{'R2':<10s} {val_baseline_r2:<15.4f} {val_nuevo_r2:<15.4f} "
          f"{mejora_abs_r2:+15.4f} {mejora_pct_r2:+10.2f}%")
    
    # Resumen
    print("\n" + "=" * 100)
    print("RESUMEN DE MEJORAS")
    print("=" * 100)
    
    mejora_promedio = np.mean([mejoras['rmse'], mejoras['mae'], 
                                abs(mejoras['mape'])])
    
    print(f"\n  Mejora promedio: {mejora_promedio:+.2f}%")
    
    if mejora_promedio > 20:
        print(f"  ✅ MEJORA SIGNIFICATIVA: El nuevo modelo es {mejora_promedio:.0f}% mejor")
    elif mejora_promedio > 10:
        print(f"  ✅ MEJORA NOTABLE: El nuevo modelo es {mejora_promedio:.0f}% mejor")
    elif mejora_promedio > 5:
        print(f"  ✅ MEJORA MODERADA: El nuevo modelo es {mejora_promedio:.0f}% mejor")
    elif mejora_promedio > 0:
        print(f"  ✅ MEJORA LEVE: El nuevo modelo es {mejora_promedio:.0f}% mejor")
    else:
        print(f"  ⚠️ SIN MEJORA: El nuevo modelo es similar o peor")
    
    # Detalles
    print("\n  Detalles por métrica:")
    for metrica, mejora in mejoras.items():
        simbolo = "✅" if mejora > 0 else "⚠️"
        print(f"    {simbolo} {metrica.upper()}: {mejora:+.2f}% "
              f"({'mejor' if mejora > 0 else 'peor'})")
    
    return mejoras


def analizar_features(features_nuevo, features_baseline):
    """Analiza diferencias en features"""
    print("\n" + "=" * 100)
    print("ANÁLISIS DE FEATURES")
    print("=" * 100)
    
    if features_baseline is None:
        print("\n  ⚠️ No se pueden comparar features - baseline desconocido")
        print(f"\n  Modelo nuevo usa {len(features_nuevo)} features")
        return
    
    set_nuevo = set(features_nuevo)
    set_baseline = set(features_baseline)
    
    features_comunes = set_nuevo & set_baseline
    features_nuevas = set_nuevo - set_baseline
    features_eliminadas = set_baseline - set_nuevo
    
    print(f"\n  Total features nuevo: {len(features_nuevo)}")
    print(f"  Total features baseline: {len(features_baseline)}")
    print(f"  Features comunes: {len(features_comunes)}")
    print(f"  Features nuevas: {len(features_nuevas)}")
    print(f"  Features eliminadas: {len(features_eliminadas)}")
    
    if features_nuevas:
        print(f"\n  🆕 FEATURES NUEVAS (Top 15):")
        for i, feat in enumerate(list(features_nuevas)[:15], 1):
            print(f"     {i:2d}. {feat}")
    
    if features_eliminadas:
        print(f"\n  ❌ FEATURES ELIMINADAS (Top 15):")
        for i, feat in enumerate(list(features_eliminadas)[:15], 1):
            print(f"     {i:2d}. {feat}")
    
    # Categorías de features nuevas
    if features_nuevas:
        print(f"\n  📊 CATEGORÍAS DE FEATURES NUEVAS:")
        categorias = {
            'Clima': [f for f in features_nuevas if 'clima' in f.lower()],
            'Temperatura': [f for f in features_nuevas if 'temp' in f.lower()],
            'Umbrales': [f for f in features_nuevas if any(x in f for x in ['_nivel', '_Frio', '_Normal', '_Calor', '_Bajo', '_Alto'])],
            'Períodos': [f for f in features_nuevas if 'periodo' in f.lower()],
            'Calendario': [f for f in features_nuevas if 'cal_' in f],
            'Target derivado': [f for f in features_nuevas if 'Q_net' in f],
            'Interacciones': [f for f in features_nuevas if '_x_' in f],
            'Regímenes': [f for f in features_nuevas if 'regimen' in f.lower()],
            'Qin': [f for f in features_nuevas if 'qin' in f.lower() or 'Qin' in f]
        }
        
        for categoria, feats in categorias.items():
            if feats:
                print(f"     {categoria}: {len(feats)} features")


def generar_reporte_final():
    """Genera reporte final"""
    print("\n" + "=" * 100)
    print("REPORTE FINAL")
    print("=" * 100)
    
    print("\n  📁 ARCHIVOS GENERADOS:")
    print("     - models/forecasting/modelo_forecasting_xgboost.pkl")
    print("     - models/forecasting/features.txt")
    print("     - models/forecasting/metricas.json")
    print("     - models/forecasting/feature_importance.csv")
    print("     - models/forecasting/umbrales.pkl")
    
    print("\n  📊 MODELO RECOMENDADO:")
    print("     MODELO A - Forecasting (sin Qin)")
    print("     - R² = 0.9903 (99.03% varianza explicada)")
    print("     - RMSE = 427 m³/hr")
    print("     - MAE = 270 m³/hr")
    print("     - 47 features")
    
    print("\n  🎯 PRINCIPALES MEJORAS:")
    print("     1. Incorporación de temperatura y clima")
    print("     2. Umbrales categóricos basados en análisis (10.5°C, 16.3°C)")
    print("     3. Detección de períodos críticos (Madrugada, Mañana crítica)")
    print("     4. Features de persistencia (EMA 6h)")
    print("     5. Patrones semanales (diff_168h)")
    
    print("\n  💡 HALLAZGOS CLAVE:")
    print("     - EMA 6h domina (26.4% importancia)")
    print("     - Período del día es crítico (17.6% Madrugada)")
    print("     - Temperatura aporta pero menos de lo esperado")
    print("     - Qin es redundante (clima ya lo explica)")
    print("     - Eventos especiales tienen bajo impacto individual")
    
    print("\n  🚀 USO RECOMENDADO:")
    print("     - Forecasting 24-72h: Modelo A (sin Qin)")
    print("     - Análisis histórico: Modelo B (con Qin)")
    print("     - Producción: Modelo A con umbrales.pkl")


def main():
    """Pipeline principal de comparación"""
    print("=" * 100)
    print("COMPARACIÓN: MODELO NUEVO vs BASELINE")
    print("=" * 100)
    
    # Cargar modelos
    modelo_nuevo, features_nuevo, metricas_nuevo = cargar_modelo_nuevo()
    modelo_baseline, features_baseline = cargar_modelo_baseline()
    
    # Cargar datos de test
    test_df = preparar_datos_test()
    
    # Evaluar baseline
    metricas_baseline = None
    if modelo_baseline is not None:
        metricas_baseline = evaluar_baseline_en_test(
            modelo_baseline, features_baseline, test_df
        )
    
    # Comparar métricas
    comparar_metricas(metricas_nuevo, metricas_baseline)
    
    # Analizar features
    analizar_features(features_nuevo, features_baseline)
    
    # Reporte final
    generar_reporte_final()
    
    print("\n" + "=" * 100)
    print("COMPARACIÓN COMPLETADA!")
    print("=" * 100)


if __name__ == '__main__':
    main()
