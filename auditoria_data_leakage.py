"""
AUDITORÍA COMPLETA DE DATA LEAKAGE
===================================
Script para verificar que NO existe filtración de datos del test set
en el proceso de entrenamiento y predicción.

Proyecto: Sistema Predictivo de Demanda de Agua Potable - Gran Valparaíso
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

print("="*80)
print("AUDITORÍA DE DATA LEAKAGE - SISTEMA ML")
print("="*80)

# ==============================================================================
# 1. VERIFICAR SPLITS DE DATOS
# ==============================================================================
print("\n1️⃣ VERIFICACIÓN DE SPLITS DE DATOS")
print("-" * 80)

# Cargar dataset completo
df = pd.read_csv('data/processed/dataset_features_completo.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp')
n = len(df)

# Calcular splits 70/15/15
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f"✅ Dataset total: {n:,} registros")
print(f"✅ Train: {len(df_train):,} registros ({len(df_train)/n*100:.1f}%)")
print(f"   └─ Periodo: {df_train['timestamp'].min()} → {df_train['timestamp'].max()}")
print(f"✅ Validación: {len(df_val):,} registros ({len(df_val)/n*100:.1f}%)")
print(f"   └─ Periodo: {df_val['timestamp'].min()} → {df_val['timestamp'].max()}")
print(f"✅ Test: {len(df_test):,} registros ({len(df_test)/n*100:.1f}%)")
print(f"   └─ Periodo: {df_test['timestamp'].min()} → {df_test['timestamp'].max()}")

# Verificar que no haya overlap
assert df_train['timestamp'].max() < df_val['timestamp'].min(), "❌ OVERLAP: Train y Val"
assert df_val['timestamp'].max() < df_test['timestamp'].min(), "❌ OVERLAP: Val y Test"
print("\n✅ No hay overlap entre train/val/test")

# ==============================================================================
# 2. VERIFICAR FEATURES CONTAMINADAS
# ==============================================================================
print("\n2️⃣ VERIFICACIÓN DE FEATURES CONTAMINADAS")
print("-" * 80)

# Cargar features del modelo
modelo = joblib.load('models/forecasting/modelo_forecasting_lgbm.pkl')
features_modelo = modelo.feature_name_

# Features prohibidas (que podrían causar leakage)
features_prohibidas = [
    'Q_net_m3h__lag',  # LAG de Q_net
    'Q_net_m3h__ema',  # EMA de Q_net
    'Q_net_m3h__diff', # Diferencias de Q_net
    'Q_net_m3h__rolling', # Rolling de Q_net
    'Q_net_m3h__shift', # Shift de Q_net
    'Qout_',  # Cualquier feature derivada de Qout
    'demanda_',  # Cualquier feature de demanda
]

print(f"Total de features en modelo: {len(features_modelo)}")
print("\nBuscando features contaminadas...")

features_sospechosas = []
for feat in features_modelo:
    for prohibida in features_prohibidas:
        if prohibida in feat:
            features_sospechosas.append(feat)
            break

if features_sospechosas:
    print(f"\n❌ ALERTA: {len(features_sospechosas)} features sospechosas encontradas:")
    for feat in features_sospechosas:
        print(f"   ⚠️  {feat}")
else:
    print("\n✅ No se encontraron features contaminadas")

# ==============================================================================
# 3. VERIFICAR PERFIL QIN (DEBE USAR SOLO TRAIN)
# ==============================================================================
print("\n3️⃣ VERIFICACIÓN DE PERFIL QIN")
print("-" * 80)

# Cargar datos de Qin
df_qin = pd.read_csv('data/raw/BD_Qin_m3_Local.csv', parse_dates=['timestamp'])
df_qin = df_qin.sort_values('timestamp')

# Fecha límite de train (igual que en interfaz)
fecha_limite_train = pd.Timestamp('2025-03-22 22:00:00', tz='UTC')

# Hacer timezone aware si es necesario
if df_qin['timestamp'].dt.tz is None:
    df_qin['timestamp'] = df_qin['timestamp'].dt.tz_localize('UTC')

# Filtrar solo datos de entrenamiento
df_qin_train = df_qin[df_qin['timestamp'] <= fecha_limite_train]

print(f"Total registros Qin disponibles: {len(df_qin):,}")
print(f"Registros Qin en periodo TRAIN: {len(df_qin_train):,}")
print(f"Fecha límite train: {fecha_limite_train}")

# Calcular perfil usando solo train
perfil_qin = df_qin_train.groupby(df_qin_train['timestamp'].dt.hour)['Qin'].median()

print(f"\n✅ Perfil Qin calculado con {len(df_qin_train):,} registros de TRAIN")
print(f"   Rango: {perfil_qin.min():.0f} - {perfil_qin.max():.0f} m³/h")

# Verificar que no haya datos de val/test en el perfil
registros_fuera_train = len(df_qin) - len(df_qin_train)
print(f"✅ Registros Qin EXCLUIDOS del perfil: {registros_fuera_train:,}")

# ==============================================================================
# 4. VERIFICAR LAG FEATURES (NO DEBEN CRUZAR FRONTERA TRAIN/TEST)
# ==============================================================================
print("\n4️⃣ VERIFICACIÓN DE LAG TEMPORAL")
print("-" * 80)

# Buscar cualquier feature con "lag" en el nombre
lag_features = [f for f in features_modelo if 'lag' in f.lower()]

if lag_features:
    print(f"⚠️  Se encontraron {len(lag_features)} features con LAG:")
    for feat in lag_features:
        print(f"   • {feat}")
    
    print("\n⚠️  ADVERTENCIA: Features con LAG pueden causar data leakage si no se manejan correctamente")
else:
    print("✅ No hay features con LAG en el modelo")

# ==============================================================================
# 5. VERIFICAR TIPOS DE FEATURES
# ==============================================================================
print("\n5️⃣ CLASIFICACIÓN DE FEATURES")
print("-" * 80)

# Clasificar features por tipo
features_climaticas = [f for f in features_modelo if any(x in f.lower() for x in 
    ['temp', 'hum', 'precip', 'viento', 'presion', 'cdh', 'api', 'vpd'])]
features_temporales = [f for f in features_modelo if any(x in f.lower() for x in 
    ['hora', 'dia', 'mes', 'semana', 'year', 'dayofweek', 'dayofyear'])]
features_calendario = [f for f in features_modelo if any(x in f.lower() for x in 
    ['festivo', 'evento', 'temporada', 'es_', 'fin_semana'])]

print(f"✅ Features climáticas: {len(features_climaticas)} (disponibles a priori)")
print(f"✅ Features temporales: {len(features_temporales)} (conocidas)")
print(f"✅ Features calendario: {len(features_calendario)} (conocidas)")

otras = len(features_modelo) - len(features_climaticas) - len(features_temporales) - len(features_calendario)
if otras > 0:
    print(f"\n📊 Otras features: {otras}")
    features_otras = [f for f in features_modelo if f not in 
        features_climaticas + features_temporales + features_calendario]
    print("   Primeras 10 features:")
    for feat in features_otras[:10]:
        print(f"   • {feat}")

# ==============================================================================
# 6. VERIFICAR MÉTRICAS DE VALIDACIÓN
# ==============================================================================
print("\n6️⃣ VERIFICACIÓN DE MÉTRICAS EN TEST SET")
print("-" * 80)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Preparar datos de test
X_test = df_test[features_modelo].fillna(0)
y_test_qnet = df_test['Q_net_m3h'].values

# Predecir Q_net
y_pred_qnet = modelo.predict(X_test)

# Calcular Qout = Qin - Q_net usando perfil Qin de TRAIN
horas = df_test['timestamp'].dt.hour
qin_por_hora = horas.map(perfil_qin).values

# Demanda real y predicha
y_real_qout = qin_por_hora - y_test_qnet
y_pred_qout = qin_por_hora - y_pred_qnet

# Métricas sobre Qout (demanda)
mae = mean_absolute_error(y_real_qout, y_pred_qout)
rmse = np.sqrt(mean_squared_error(y_real_qout, y_pred_qout))
r2 = r2_score(y_real_qout, y_pred_qout)

# Calcular MAPE
mask = y_real_qout != 0
mape = np.mean(np.abs((y_real_qout[mask] - y_pred_qout[mask]) / y_real_qout[mask])) * 100

print(f"Métricas sobre DEMANDA (Qout) en test set:")
print(f"  • MAE:  {mae:,.0f} m³/h")
print(f"  • RMSE: {rmse:,.0f} m³/h")
print(f"  • R²:   {r2:.4f}")
print(f"  • MAPE: {mape:.2f}%")

# Análisis de métricas
print("\n📊 Análisis de métricas:")
if r2 > 0.95:
    print("⚠️  ALERTA: R² muy alto (>0.95) puede indicar data leakage")
    alerta_r2 = True
elif r2 > 0.80:
    print("⚠️  PRECAUCIÓN: R² alto (>0.80) verificar si es realista")
    alerta_r2 = False
else:
    print("✅ R² dentro de rango esperado para predicción exógena")
    alerta_r2 = False

if mae < 500:
    print("⚠️  ALERTA: MAE muy bajo (<500) puede indicar data leakage")
    alerta_mae = True
elif mae < 1000:
    print("⚠️  PRECAUCIÓN: MAE bajo (<1000) verificar modelo")
    alerta_mae = False
else:
    print("✅ MAE dentro de rango esperado")
    alerta_mae = False

# ==============================================================================
# 7. ANÁLISIS DE RESIDUOS
# ==============================================================================
print("\n7️⃣ ANÁLISIS DE RESIDUOS")
print("-" * 80)

residuos = y_pred_qout - y_real_qout

print(f"Estadísticas de residuos:")
print(f"  • Media:        {residuos.mean():,.0f} m³/h")
print(f"  • Desv. Std:    {residuos.std():,.0f} m³/h")
print(f"  • Mínimo:       {residuos.min():,.0f} m³/h")
print(f"  • Máximo:       {residuos.max():,.0f} m³/h")
print(f"  • Percentil 5:  {np.percentile(residuos, 5):,.0f} m³/h")
print(f"  • Percentil 95: {np.percentile(residuos, 95):,.0f} m³/h")

# La media de residuos debe estar cerca de 0
if abs(residuos.mean()) > 100:
    print("\n⚠️  ADVERTENCIA: Media de residuos alejada de 0 (sesgo)")
else:
    print("\n✅ Media de residuos cercana a 0 (sin sesgo)")

# ==============================================================================
# 8. GENERAR REPORTE DE AUDITORÍA
# ==============================================================================
print("\n8️⃣ GENERACIÓN DE REPORTE")
print("-" * 80)

reporte = {
    "fecha_auditoria": pd.Timestamp.now().isoformat(),
    "dataset": {
        "total_registros": int(n),
        "train": {
            "registros": int(len(df_train)),
            "porcentaje": f"{len(df_train)/n*100:.1f}%",
            "periodo": f"{df_train['timestamp'].min()} → {df_train['timestamp'].max()}"
        },
        "validacion": {
            "registros": int(len(df_val)),
            "porcentaje": f"{len(df_val)/n*100:.1f}%",
            "periodo": f"{df_val['timestamp'].min()} → {df_val['timestamp'].max()}"
        },
        "test": {
            "registros": int(len(df_test)),
            "porcentaje": f"{len(df_test)/n*100:.1f}%",
            "periodo": f"{df_test['timestamp'].min()} → {df_test['timestamp'].max()}"
        }
    },
    "features": {
        "total": len(features_modelo),
        "climaticas": len(features_climaticas),
        "temporales": len(features_temporales),
        "calendario": len(features_calendario),
        "otras": otras,
        "con_lag": len(lag_features),
        "sospechosas": len(features_sospechosas),
        "lista_sospechosas": features_sospechosas
    },
    "perfil_qin": {
        "registros_usados": int(len(df_qin_train)),
        "registros_totales": int(len(df_qin)),
        "usa_solo_train": True,
        "registros_excluidos": int(registros_fuera_train),
        "rango_mediana": f"{perfil_qin.min():.0f} - {perfil_qin.max():.0f} m³/h"
    },
    "metricas_test": {
        "mae_m3h": float(mae),
        "rmse_m3h": float(rmse),
        "r2": float(r2),
        "mape_pct": float(mape)
    },
    "residuos": {
        "media": float(residuos.mean()),
        "std": float(residuos.std()),
        "min": float(residuos.min()),
        "max": float(residuos.max())
    },
    "validacion": {
        "sin_overlap": bool(True),
        "sin_features_contaminadas": bool(len(features_sospechosas) == 0),
        "perfil_qin_limpio": bool(True),
        "r2_realista": bool(not alerta_r2),
        "mae_realista": bool(not alerta_mae),
        "sin_sesgo": bool(abs(residuos.mean()) < 100)
    }
}

# Guardar reporte
Path('outputs').mkdir(exist_ok=True)
with open('outputs/AUDITORIA_DATA_LEAKAGE.json', 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False)

print("✅ Reporte guardado en: outputs/AUDITORIA_DATA_LEAKAGE.json")

# ==============================================================================
# 9. CONCLUSIÓN
# ==============================================================================
print("\n" + "="*80)
print("CONCLUSIÓN DE LA AUDITORÍA")
print("="*80)

todas_verificaciones = [
    ("Sin overlap train/val/test", True),
    ("Sin features contaminadas", len(features_sospechosas) == 0),
    ("Perfil Qin usa solo train", True),
    ("R² realista (< 0.95)", not alerta_r2),
    ("MAE realista (> 500)", not alerta_mae),
    ("Sin sesgo en residuos", abs(residuos.mean()) < 100),
]

print("\nResultado de verificaciones:")
for nombre, resultado in todas_verificaciones:
    icono = "✅" if resultado else "❌"
    print(f"{icono} {nombre}")

# Contar aprobadas
aprobadas = sum(1 for _, r in todas_verificaciones if r)
total = len(todas_verificaciones)

print(f"\n📊 Score: {aprobadas}/{total} verificaciones aprobadas ({aprobadas/total*100:.0f}%)")

if aprobadas == total:
    print("\n" + "🎉" * 40)
    print("✅ AUDITORÍA EXITOSA: NO SE DETECTÓ DATA LEAKAGE")
    print("🎉" * 40)
    print("\nEl sistema está LIMPIO y las métricas son CONFIABLES.")
elif aprobadas >= total * 0.8:
    print("\n" + "⚠️ " * 40)
    print("⚠️  ADVERTENCIA: Se detectaron algunas alertas menores")
    print("⚠️ " * 40)
    print("\nRevisar los puntos marcados con ⚠️")
else:
    print("\n" + "❌" * 40)
    print("❌ ALERTA CRÍTICA: POSIBLE DATA LEAKAGE DETECTADO")
    print("❌" * 40)
    print("\nAcción requerida: Revisar features sospechosas y métricas anómalas.")

print("\n" + "="*80)
print("Fin de la auditoría")
print("="*80)
