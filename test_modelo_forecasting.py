"""
Test Rápido: Modelo Forecasting V3.0
====================================
Verifica que el modelo cargue y prediga correctamente.
"""

import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

def test_carga_modelo():
    """Test 1: Verificar carga de modelo y metadatos"""
    print("=" * 80)
    print("TEST 1: CARGA DE MODELO")
    print("=" * 80)
    
    try:
        # Cargar modelo
        modelo = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
        print("✅ Modelo cargado correctamente")
        
        # Cargar features
        with open('models/forecasting/features.txt') as f:
            features = [line.strip() for line in f]
        print(f"✅ Features cargadas: {len(features)}")
        
        # Cargar métricas
        with open('models/forecasting/metricas.json') as f:
            metricas = json.load(f)
        print(f"✅ Métricas cargadas:")
        print(f"   - RMSE: {metricas['rmse']:.2f} m³/hr")
        print(f"   - MAE: {metricas['mae']:.2f} m³/hr")
        print(f"   - R²: {metricas['r2']:.4f}")
        print(f"   - MAPE: {metricas['mape']:.2f}%")
        
        # Cargar umbrales
        umbrales = joblib.load('models/forecasting/umbrales.pkl')
        print(f"✅ Umbrales cargados:")
        print(f"   - Temp frío: {umbrales['temp_frio']:.1f}°C")
        print(f"   - Temp calor: {umbrales['temp_calor']:.1f}°C")
        print(f"   - Qin bajo: {umbrales['qin_bajo']:.0f} m³/hr")
        print(f"   - Qin alto: {umbrales['qin_alto']:.0f} m³/hr")
        
        return True, modelo, features, metricas, umbrales
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None, None, None, None


def test_datos_disponibles():
    """Test 2: Verificar disponibilidad de datos"""
    print("\n" + "=" * 80)
    print("TEST 2: DATOS DISPONIBLES")
    print("=" * 80)
    
    try:
        # Cargar dataset
        df = pd.read_csv('data/processed/dataset_features_completo.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"✅ Dataset cargado: {df.shape}")
        print(f"   - Registros: {len(df):,}")
        print(f"   - Features: {df.shape[1]}")
        print(f"   - Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        # Verificar target
        if 'Q_net_m3h' in df.columns:
            print(f"   - Target (Q_net): {df['Q_net_m3h'].notna().sum():,} valores válidos")
            print(f"   - Media: {df['Q_net_m3h'].mean():.2f} m³/hr")
            print(f"   - Std: {df['Q_net_m3h'].std():.2f} m³/hr")
        
        return True, df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def test_prediccion_simple(modelo, features, df):
    """Test 3: Predicción con última observación"""
    print("\n" + "=" * 80)
    print("TEST 3: PREDICCIÓN SIMPLE")
    print("=" * 80)
    
    try:
        # Verificar features disponibles
        features_disponibles = [f for f in features if f in df.columns]
        features_faltantes = [f for f in features if f not in df.columns]
        
        print(f"   Features disponibles: {len(features_disponibles)}/{len(features)}")
        
        if features_faltantes:
            print(f"   ⚠️ Features faltantes: {len(features_faltantes)}")
            print(f"      Primeras 5: {features_faltantes[:5]}")
            return False
        
        # Tomar últimas 5 observaciones
        X_test = df[features].tail(5)
        y_real = df['Q_net_m3h'].tail(5).values
        
        # Predecir
        y_pred = modelo.predict(X_test)
        
        print(f"\n   Predicciones últimas 5 horas:")
        print(f"   {'Hora':<8s} {'Real':<15s} {'Predicción':<15s} {'Error':<15s}")
        print(f"   {'-'*60}")
        
        for i in range(5):
            timestamp = df['timestamp'].tail(5).iloc[i]
            real = y_real[i]
            pred = y_pred[i]
            error = abs(real - pred)
            
            print(f"   {timestamp.strftime('%H:%M'):<8s} "
                  f"{real:>10.0f} m³/hr "
                  f"{pred:>10.0f} m³/hr "
                  f"{error:>10.0f} m³/hr")
        
        # Métricas
        mae = np.mean(np.abs(y_real - y_pred))
        rmse = np.sqrt(np.mean((y_real - y_pred)**2))
        
        print(f"\n   MAE (5 muestras): {mae:.2f} m³/hr")
        print(f"   RMSE (5 muestras): {rmse:.2f} m³/hr")
        
        print(f"\n✅ Predicciones ejecutadas correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_importance(features):
    """Test 4: Analizar importancia de features"""
    print("\n" + "=" * 80)
    print("TEST 4: FEATURE IMPORTANCE")
    print("=" * 80)
    
    try:
        # Cargar importancias
        df_imp = pd.read_csv('models/forecasting/feature_importance.csv')
        
        print(f"✅ Importancias cargadas: {len(df_imp)} features")
        print(f"\n   TOP 10 Features más importantes:")
        print(f"   {'#':<4s} {'Feature':<45s} {'Importancia':<12s}")
        print(f"   {'-'*65}")
        
        for i, row in df_imp.head(10).iterrows():
            print(f"   {i+1:<4d} {row['feature']:<45s} {row['importance']:>10.2f}%")
        
        # Categorías
        categorias = {
            'Target': df_imp[df_imp['feature'].str.contains('Q_net', na=False)]['importance'].sum(),
            'Clima': df_imp[df_imp['feature'].str.contains('clima', na=False)]['importance'].sum(),
            'Calendario': df_imp[df_imp['feature'].str.contains('cal_', na=False)]['importance'].sum(),
            'Temporal': df_imp[df_imp['feature'].str.contains('hora|dia_semana|mes', regex=True, na=False)]['importance'].sum(),
            'Períodos': df_imp[df_imp['feature'].str.contains('periodo_dia', na=False)]['importance'].sum(),
        }
        
        print(f"\n   Importancia por categoría:")
        for cat, imp in sorted(categorias.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat:<15s}: {imp:>6.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_umbrales_temperatura(df, umbrales):
    """Test 5: Verificar distribución por umbrales"""
    print("\n" + "=" * 80)
    print("TEST 5: UMBRALES DE TEMPERATURA")
    print("=" * 80)
    
    try:
        # Clasificar por temperatura
        if 'clima_temp_c' not in df.columns:
            print("⚠️ No hay datos de temperatura en dataset")
            return False
        
        temp = df['clima_temp_c'].dropna()
        
        frio = (temp < umbrales['temp_frio']).sum()
        normal = ((temp >= umbrales['temp_frio']) & (temp <= umbrales['temp_calor'])).sum()
        calor = (temp > umbrales['temp_calor']).sum()
        
        total = frio + normal + calor
        
        print(f"   Distribución de temperatura:")
        print(f"   - Frío (<{umbrales['temp_frio']:.1f}°C):    {frio:>6,} registros ({frio/total*100:>5.1f}%)")
        print(f"   - Normal ({umbrales['temp_frio']:.1f}-{umbrales['temp_calor']:.1f}°C): {normal:>6,} registros ({normal/total*100:>5.1f}%)")
        print(f"   - Calor (>{umbrales['temp_calor']:.1f}°C):   {calor:>6,} registros ({calor/total*100:>5.1f}%)")
        
        # Stats por categoría
        print(f"\n   Demanda promedio por temperatura:")
        if 'Q_net_m3h' in df.columns:
            df_temp = df[['clima_temp_c', 'Q_net_m3h']].dropna()
            
            q_frio = df_temp[df_temp['clima_temp_c'] < umbrales['temp_frio']]['Q_net_m3h'].mean()
            q_normal = df_temp[(df_temp['clima_temp_c'] >= umbrales['temp_frio']) & 
                              (df_temp['clima_temp_c'] <= umbrales['temp_calor'])]['Q_net_m3h'].mean()
            q_calor = df_temp[df_temp['clima_temp_c'] > umbrales['temp_calor']]['Q_net_m3h'].mean()
            
            print(f"   - Frío:   {q_frio:>8.0f} m³/hr")
            print(f"   - Normal: {q_normal:>8.0f} m³/hr")
            print(f"   - Calor:  {q_calor:>8.0f} m³/hr")
            print(f"   - Delta frío→calor: {q_calor - q_frio:>8.0f} m³/hr")
        
        print(f"\n✅ Umbrales verificados correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n" + "🧪" * 40)
    print("TEST SUITE: MODELO FORECASTING V3.0")
    print("🧪" * 40 + "\n")
    
    resultados = []
    
    # Test 1: Carga
    success, modelo, features, metricas, umbrales = test_carga_modelo()
    resultados.append(("Carga de Modelo", success))
    
    if not success:
        print("\n❌ Tests abortados - modelo no disponible")
        return
    
    # Test 2: Datos
    success, df = test_datos_disponibles()
    resultados.append(("Datos Disponibles", success))
    
    if not success:
        print("\n❌ Tests abortados - datos no disponibles")
        return
    
    # Test 3: Predicción
    success = test_prediccion_simple(modelo, features, df)
    resultados.append(("Predicción Simple", success))
    
    # Test 4: Feature Importance
    success = test_feature_importance(features)
    resultados.append(("Feature Importance", success))
    
    # Test 5: Umbrales
    success = test_umbrales_temperatura(df, umbrales)
    resultados.append(("Umbrales Temperatura", success))
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE TESTS")
    print("=" * 80)
    
    total = len(resultados)
    exitosos = sum(1 for _, s in resultados if s)
    
    for nombre, success in resultados:
        simbolo = "✅" if success else "❌"
        print(f"   {simbolo} {nombre}")
    
    print(f"\n   Total: {exitosos}/{total} tests exitosos ({exitosos/total*100:.0f}%)")
    
    if exitosos == total:
        print("\n🎉 TODOS LOS TESTS PASARON - MODELO LISTO PARA USO")
    else:
        print("\n⚠️ ALGUNOS TESTS FALLARON - REVISAR ERRORES")
    
    print("\n" + "🧪" * 40 + "\n")


if __name__ == '__main__':
    main()
