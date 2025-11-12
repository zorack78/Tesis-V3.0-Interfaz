"""
REENTRENAMIENTO CON 253 CASOS NEGATIVOS REINCORPORADOS
=======================================================

OBJETIVO:
Validar si al reincorporar los 253 casos negativos (recuperación legítima),
la temperatura gana importancia predictiva.

HIPÓTESIS:
Los 253 casos negativos tienen fuerte correlación con frío/invierno.
Al reincorporarlos, el modelo debería aprender mejor el efecto de temperatura.

PROCESO:
1. Cargar datos completos
2. Filtrar SOLO 86 outliers positivos (>30,000)
3. Mantener 253 negativos
4. Crear features climáticas avanzadas
5. Entrenar 2 modelos: con/sin clima
6. Comparar importancia de temperatura
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 80)
print("REENTRENAMIENTO: VALIDANDO IMPORTANCIA CLIMA CON 253 CASOS NEGATIVOS")
print("=" * 80)
print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================================

print("\n" + "=" * 80)
print("[1/6] CARGANDO Y FILTRANDO DATOS")
print("=" * 80)

# Cargar datos completos
df = pd.read_csv('data/processed/data_processed_complete.csv')
print(f"\n✅ Datos cargados: {len(df)} registros")

# Calcular demanda si no existe
if 'Demanda_m3_hr' not in df.columns:
    print("   Calculando Demanda_m3_hr...")
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    df['Demanda_m3_hr'] = -df[' Volumen_Total_m3_diff_1h']
    
demanda_col = 'Demanda_m3_hr'
print(f"   Columna demanda: {demanda_col}")

# Estadísticas ANTES del filtrado
print(f"\n📊 ESTADÍSTICAS ANTES DEL FILTRADO:")
print(f"   Total registros: {len(df)}")
print(f"   Negativos (<0): {(df[demanda_col] < 0).sum()}")
print(f"   Positivos extremos (>30k): {(df[demanda_col] > 30000).sum()}")
print(f"   Rango: {df[demanda_col].min():,.0f} a {df[demanda_col].max():,.0f} m³/hr")

# FILTRADO SELECTIVO: Solo eliminar positivos extremos
df_filtered = df[df[demanda_col] <= 30000].copy()

# Estadísticas DESPUÉS del filtrado
negativos_mantenidos = (df_filtered[demanda_col] < 0).sum()
print(f"\n📊 ESTADÍSTICAS DESPUÉS DEL FILTRADO:")
print(f"   Total registros: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")
print(f"   Registros filtrados: {len(df) - len(df_filtered)} (solo positivos >30k)")
print(f"   ✅ Negativos MANTENIDOS: {negativos_mantenidos}")
print(f"   Rango: {df_filtered[demanda_col].min():,.0f} a {df_filtered[demanda_col].max():,.0f} m³/hr")

# Guardar dataset filtrado
df_filtered.to_csv('data/processed/data_con_negativos_mantenidos.csv', index=False)
print(f"\n✅ Dataset guardado: data/processed/data_con_negativos_mantenidos.csv")

# ============================================================================
# 2. CARGAR DATOS CLIMÁTICOS
# ============================================================================

print("\n" + "=" * 80)
print("[2/6] CARGANDO DATOS CLIMÁTICOS")
print("=" * 80)

df_clima = pd.read_csv('data/processed/clima_chile_v3.csv')
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'], utc=True)

# Renombrar columnas
df_clima = df_clima.rename(columns={
    'temp': 'temperatura',
    'HR': 'humedad_relativa',
    'mmhr': 'precipitacion'
})

print(f"✅ Datos climáticos cargados: {len(df_clima)} registros")

# Merge con datos principales
df_filtered['timestamp_utc'] = pd.to_datetime(df_filtered['timestamp_utc'], utc=True)
df_full = df_filtered.merge(df_clima[['timestamp', 'temperatura', 'humedad_relativa', 'precipitacion']], 
                             left_on='timestamp_utc', right_on='timestamp', how='left')

print(f"✅ Datos combinados: {len(df_full)} registros")

# ============================================================================
# 3. CREAR FEATURES CLIMÁTICAS AVANZADAS
# ============================================================================

print("\n" + "=" * 80)
print("[3/6] CREANDO FEATURES CLIMÁTICAS AVANZADAS")
print("=" * 80)

print(f"\n⏳ Creando features climáticas avanzadas...")

# Ordenar por timestamp
df_full = df_full.sort_values('timestamp_utc').reset_index(drop=True)

# LAGs de temperatura (solo cortos para no perder datos)
for lag in [1, 2, 24]:
    df_full[f'temp_lag_{lag}h'] = df_full['temperatura'].shift(lag)

# Rolling de temperatura (ventanas cortas)
for window in [6, 24]:
    df_full[f'temp_rolling_mean_{window}h'] = df_full['temperatura'].rolling(window).mean()
    df_full[f'temp_rolling_std_{window}h'] = df_full['temperatura'].rolling(window).std()
    df_full[f'temp_rolling_max_{window}h'] = df_full['temperatura'].rolling(window).max()
    df_full[f'temp_rolling_min_{window}h'] = df_full['temperatura'].rolling(window).min()

# Diferencias de temperatura
df_full['temp_diff_1h'] = df_full['temperatura'].diff(1)
df_full['temp_diff_24h'] = df_full['temperatura'].diff(24)

# Temperaturas extremas
df_full['temp_muy_baja'] = (df_full['temperatura'] < 12).astype(int)
df_full['temp_baja'] = ((df_full['temperatura'] >= 12) & (df_full['temperatura'] < 15)).astype(int)
df_full['temp_alta'] = (df_full['temperatura'] > 22).astype(int)

# LAGs de humedad (solo cortos)
for lag in [1, 2, 24]:
    df_full[f'hr_lag_{lag}h'] = df_full['humedad_relativa'].shift(lag)

# Rolling de humedad (ventanas cortas)
for window in [6, 24]:
    df_full[f'hr_rolling_mean_{window}h'] = df_full['humedad_relativa'].rolling(window).mean()

# Humedad extrema
df_full['hr_muy_baja'] = (df_full['humedad_relativa'] < 30).astype(int)
df_full['hr_alta'] = (df_full['humedad_relativa'] >= 70).astype(int)

# Precipitación
for lag in [1, 6]:
    df_full[f'precip_lag_{lag}h'] = df_full['precipitacion'].shift(lag)

for window in [3, 6, 24]:
    df_full[f'precip_acum_{window}h'] = df_full['precipitacion'].rolling(window).sum()

df_full['lluvia_intensa'] = (df_full['precipitacion'] > 2).astype(int)

# Interacciones
df_full['sensacion_termica'] = df_full['temperatura'] * (1 - df_full['humedad_relativa']/100)
df_full['condiciones_adversas'] = ((df_full['lluvia_intensa'] == 1) | 
                                     (df_full['temp_muy_baja'] == 1) | 
                                     (df_full['temp_alta'] == 1)).astype(int)

# Eliminar NaNs generados por LAGs/Rolling (solo perderemos ~24 horas)
df_full = df_full.dropna()

print(f"✅ Features climáticas creadas")
print(f"   Registros finales (sin NaN): {len(df_full)}")

# ============================================================================
# 4. PREPARAR FEATURES PARA ENTRENAMIENTO
# ============================================================================

print("\n" + "=" * 80)
print("[4/6] PREPARANDO FEATURES PARA ENTRENAMIENTO")
print("=" * 80)

# Features base (temporales)
features_base = [
    'hora', 'dia_semana', 'hora_seno', 'hora_coseno',
    'dia_semana_seno', 'dia_semana_coseno',
    'feriado', 'feriado_irrenunciable',
    'vacaciones_escolares', 'temporada_turistica_alta'
]

# Features climáticas
features_clima = [col for col in df_full.columns if any(x in col for x in 
    ['temp_', 'hr_', 'precip_', 'sensacion', 'condiciones', 'lluvia_intensa'])]

print(f"\n📊 FEATURES DISPONIBLES:")
print(f"   Features base (temporales): {len(features_base)}")
print(f"   Features climáticas: {len(features_clima)}")

# Verificar cuáles existen
features_base_disponibles = [f for f in features_base if f in df_full.columns]
print(f"\n   ✅ Features base disponibles: {len(features_base_disponibles)}")

# Variable objetivo
y = df_full[demanda_col]

# ============================================================================
# 5. ENTRENAR MODELOS: CON Y SIN CLIMA
# ============================================================================

print("\n" + "=" * 80)
print("[5/6] ENTRENANDO MODELOS: CON Y SIN CLIMA")
print("=" * 80)

resultados = {}

# MODELO 1: Sin clima (solo temporales)
print("\n🔄 [MODELO 1] Entrenando SIN clima (solo temporales)...")
X_sin_clima = df_full[features_base_disponibles]

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_sin_clima, y, test_size=0.2, random_state=42, shuffle=False
)

model_sin_clima = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_sin_clima.fit(X_train_1, y_train_1)

y_pred_train_1 = model_sin_clima.predict(X_train_1)
y_pred_test_1 = model_sin_clima.predict(X_test_1)

resultados['sin_clima'] = {
    'model': model_sin_clima,
    'r2_train': r2_score(y_train_1, y_pred_train_1),
    'r2_test': r2_score(y_test_1, y_pred_test_1),
    'mape_train': mean_absolute_percentage_error(y_train_1, y_pred_train_1) * 100,
    'mape_test': mean_absolute_percentage_error(y_test_1, y_pred_test_1) * 100,
    'mae_test': mean_absolute_error(y_test_1, y_pred_test_1),
    'features': len(features_base_disponibles)
}

print(f"✅ Modelo 1 entrenado")
print(f"   R² Test: {resultados['sin_clima']['r2_test']:.4f}")
print(f"   MAPE Test: {resultados['sin_clima']['mape_test']:.2f}%")

# MODELO 2: Con clima (temporales + climáticas)
print("\n🔄 [MODELO 2] Entrenando CON clima (temporales + 37 climáticas)...")
features_completas = features_base_disponibles + features_clima
X_con_clima = df_full[features_completas]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_con_clima, y, test_size=0.2, random_state=42, shuffle=False
)

model_con_clima = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_con_clima.fit(X_train_2, y_train_2)

y_pred_train_2 = model_con_clima.predict(X_train_2)
y_pred_test_2 = model_con_clima.predict(X_test_2)

resultados['con_clima'] = {
    'model': model_con_clima,
    'r2_train': r2_score(y_train_2, y_pred_train_2),
    'r2_test': r2_score(y_test_2, y_pred_test_2),
    'mape_train': mean_absolute_percentage_error(y_train_2, y_pred_train_2) * 100,
    'mape_test': mean_absolute_percentage_error(y_test_2, y_pred_test_2) * 100,
    'mae_test': mean_absolute_error(y_test_2, y_pred_test_2),
    'features': len(features_completas)
}

print(f"✅ Modelo 2 entrenado")
print(f"   R² Test: {resultados['con_clima']['r2_test']:.4f}")
print(f"   MAPE Test: {resultados['con_clima']['mape_test']:.2f}%")

# ============================================================================
# 6. ANÁLISIS DE IMPORTANCIA DE FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("[6/6] ANÁLISIS DE IMPORTANCIA DE FEATURES")
print("=" * 80)

# Obtener importancias del modelo CON clima
importances = model_con_clima.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features_completas,
    'importance': importances
}).sort_values('importance', ascending=False)

# Clasificar features
feature_importance_df['tipo'] = feature_importance_df['feature'].apply(
    lambda x: 'Clima' if any(c in x for c in ['temp_', 'hr_', 'precip_', 'sensacion', 'condiciones', 'lluvia']) else 'Temporal'
)

# Calcular importancia por tipo
importancia_por_tipo = feature_importance_df.groupby('tipo')['importance'].sum()

print(f"\n📊 IMPORTANCIA POR TIPO DE FEATURE:")
for tipo, imp in importancia_por_tipo.items():
    print(f"   {tipo}: {imp*100:.2f}%")

# Top 20 features
print(f"\n📊 TOP 20 FEATURES MÁS IMPORTANTES:")
print(f"{'Rank':<6} {'Feature':<35} {'Importancia':<12} {'Tipo':<10}")
print("-" * 65)
for i, row in feature_importance_df.head(20).iterrows():
    print(f"{feature_importance_df.index.get_loc(i)+1:<6} {row['feature']:<35} {row['importance']*100:>10.2f}% {row['tipo']:<10}")

# Features climáticas específicas
print(f"\n📊 TOP 10 FEATURES CLIMÁTICAS:")
clima_features = feature_importance_df[feature_importance_df['tipo'] == 'Clima'].head(10)
for i, (idx, row) in enumerate(clima_features.iterrows(), 1):
    print(f"   {i}. {row['feature']:<35} {row['importance']*100:.2f}%")

# ============================================================================
# 7. EVALUACIÓN EN CASOS NEGATIVOS
# ============================================================================

print("\n" + "=" * 80)
print("EVALUACIÓN ESPECÍFICA EN CASOS NEGATIVOS")
print("=" * 80)

# Identificar casos negativos en test
negativos_mask = y_test_2 < 0

if negativos_mask.sum() > 0:
    y_neg_real = y_test_2[negativos_mask]
    y_neg_pred = y_pred_test_2[negativos_mask]
    
    mae_neg = mean_absolute_error(y_neg_real, y_neg_pred)
    
    print(f"\n📊 DESEMPEÑO EN CASOS NEGATIVOS:")
    print(f"   Casos negativos en test: {negativos_mask.sum()}")
    print(f"   MAE en negativos: {mae_neg:,.0f} m³/hr")
    print(f"   Rango real: {y_neg_real.min():,.0f} a {y_neg_real.max():,.0f}")
    print(f"   Rango predicho: {y_neg_pred.min():,.0f} a {y_neg_pred.max():,.0f}")
    
    # Comparar con MAE general
    mejora = (mae_neg / resultados['con_clima']['mae_test']) * 100
    print(f"\n   MAE negativos vs general: {mejora:.1f}%")
    if mejora < 100:
        print(f"   ✅ Mejor predicción en negativos ({100-mejora:.1f}% mejor)")
    else:
        print(f"   ⚠️ Peor predicción en negativos ({mejora-100:.1f}% peor)")
else:
    print("\n⚠️ No hay casos negativos en el conjunto de test")

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================

print("\n" + "=" * 80)
print("GENERANDO VISUALIZACIONES")
print("=" * 80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Reentrenamiento con 253 Casos Negativos Reincorporados', 
             fontsize=16, fontweight='bold')

# 1. Comparación de métricas
ax1 = fig.add_subplot(gs[0, 0])
modelos = ['Sin Clima', 'Con Clima']
r2_values = [resultados['sin_clima']['r2_test'], resultados['con_clima']['r2_test']]
mape_values = [resultados['sin_clima']['mape_test'], resultados['con_clima']['mape_test']]

x = np.arange(len(modelos))
width = 0.35

ax1_twin = ax1.twinx()
bars1 = ax1.bar(x - width/2, r2_values, width, label='R²', color='steelblue', alpha=0.7)
bars2 = ax1_twin.bar(x + width/2, mape_values, width, label='MAPE', color='coral', alpha=0.7)

ax1.set_ylabel('R²', fontweight='bold')
ax1_twin.set_ylabel('MAPE (%)', fontweight='bold')
ax1.set_title('Comparación de Métricas')
ax1.set_xticks(x)
ax1.set_xticklabels(modelos)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, r2_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, val in zip(bars2, mape_values):
    height = bar.get_height()
    ax1_twin.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Importancia por tipo
ax2 = fig.add_subplot(gs[0, 1])
colors_tipo = ['steelblue', 'coral']
wedges, texts, autotexts = ax2.pie(importancia_por_tipo.values, 
                                     labels=importancia_por_tipo.index,
                                     autopct='%1.1f%%',
                                     colors=colors_tipo,
                                     startangle=90)
ax2.set_title('Importancia por Tipo de Feature')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 3. Top 15 features
ax3 = fig.add_subplot(gs[0, 2])
top15 = feature_importance_df.head(15)
colors_features = ['coral' if t == 'Clima' else 'steelblue' for t in top15['tipo']]
ax3.barh(range(len(top15)), top15['importance']*100, color=colors_features, alpha=0.7)
ax3.set_yticks(range(len(top15)))
ax3.set_yticklabels(top15['feature'], fontsize=8)
ax3.set_xlabel('Importancia (%)')
ax3.set_title('Top 15 Features Más Importantes')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Leyenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Temporal'),
                   Patch(facecolor='coral', alpha=0.7, label='Clima')]
ax3.legend(handles=legend_elements, loc='lower right')

# 4. Predicciones vs Reales (con clima)
ax4 = fig.add_subplot(gs[1, :])
sample = np.random.choice(len(y_test_2), min(500, len(y_test_2)), replace=False)
colors_scatter = ['red' if val < 0 else 'blue' for val in y_test_2.iloc[sample]]
ax4.scatter(y_test_2.iloc[sample], y_pred_test_2[sample], 
           alpha=0.5, c=colors_scatter, s=20)
ax4.plot([y_test_2.min(), y_test_2.max()], 
         [y_test_2.min(), y_test_2.max()], 
         'k--', lw=2, label='Ideal')
ax4.set_xlabel('Demanda Real (m³/hr)')
ax4.set_ylabel('Demanda Predicha (m³/hr)')
ax4.set_title(f'Predicciones vs Reales (Modelo CON Clima) - R²={resultados["con_clima"]["r2_test"]:.4f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Leyenda de colores
legend_elements_scatter = [Patch(facecolor='red', alpha=0.5, label='Negativos (recuperación)'),
                          Patch(facecolor='blue', alpha=0.5, label='Positivos (consumo)')]
ax4.legend(handles=legend_elements_scatter, loc='upper left')

# 5. Distribución de errores
ax5 = fig.add_subplot(gs[2, 0])
errors = y_test_2 - y_pred_test_2
ax5.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Error (Real - Predicho)')
ax5.set_ylabel('Frecuencia')
ax5.set_title('Distribución de Errores')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Top features climáticas
ax6 = fig.add_subplot(gs[2, 1])
top_clima = feature_importance_df[feature_importance_df['tipo'] == 'Clima'].head(10)
ax6.barh(range(len(top_clima)), top_clima['importance']*100, color='coral', alpha=0.7)
ax6.set_yticks(range(len(top_clima)))
ax6.set_yticklabels(top_clima['feature'], fontsize=8)
ax6.set_xlabel('Importancia (%)')
ax6.set_title('Top 10 Features Climáticas')
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')

# 7. Mejora absoluta
ax7 = fig.add_subplot(gs[2, 2])
mejora_r2 = (resultados['con_clima']['r2_test'] - resultados['sin_clima']['r2_test']) * 100
mejora_mape = resultados['sin_clima']['mape_test'] - resultados['con_clima']['mape_test']

metrics = ['Mejora R²\n(puntos %)', 'Reducción MAPE\n(puntos %)']
values = [mejora_r2, mejora_mape]
colors_bars = ['green' if v > 0 else 'red' for v in values]

bars = ax7.bar(metrics, values, color=colors_bars, alpha=0.7, edgecolor='black')
ax7.axhline(0, color='black', linewidth=1)
ax7.set_ylabel('Mejora (%)')
ax7.set_title('Mejora por Agregar Clima')
ax7.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.2f}%', ha='center', 
            va='bottom' if val > 0 else 'top',
            fontweight='bold')

plt.savefig('outputs/reentrenamiento_con_negativos_reincorporados.png', 
            dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: outputs/reentrenamiento_con_negativos_reincorporados.png")

# ============================================================================
# 9. GUARDAR MODELOS Y RESULTADOS
# ============================================================================

print("\n" + "=" * 80)
print("GUARDANDO MODELOS Y RESULTADOS")
print("=" * 80)

joblib.dump(model_sin_clima, 'models/modelo_negativos_reincorp_sin_clima.pkl')
joblib.dump(model_con_clima, 'models/modelo_negativos_reincorp_con_clima.pkl')
print("✅ Modelos guardados")

# Guardar importancias
feature_importance_df.to_csv('outputs/importancia_features_negativos_reincorp.csv', index=False)
print("✅ Importancias guardadas")

# ============================================================================
# 10. REPORTE FINAL
# ============================================================================

print("\n" + "=" * 80)
print("📊 REPORTE FINAL")
print("=" * 80)

print(f"\n{'='*80}")
print("COMPARACIÓN DE MODELOS")
print(f"{'='*80}")

print(f"\n{'Métrica':<30} {'Sin Clima':<20} {'Con Clima':<20} {'Diferencia':<15}")
print("-" * 85)
print(f"{'Features':<30} {resultados['sin_clima']['features']:<20} {resultados['con_clima']['features']:<20} {resultados['con_clima']['features'] - resultados['sin_clima']['features']:<15}")
print(f"{'R² Test':<30} {resultados['sin_clima']['r2_test']:<20.4f} {resultados['con_clima']['r2_test']:<20.4f} {(resultados['con_clima']['r2_test'] - resultados['sin_clima']['r2_test'])*100:+.2f} pp")
print(f"{'MAPE Test (%)':<30} {resultados['sin_clima']['mape_test']:<20.2f} {resultados['con_clima']['mape_test']:<20.2f} {resultados['sin_clima']['mape_test'] - resultados['con_clima']['mape_test']:+.2f} pp")
print(f"{'MAE Test (m³/hr)':<30} {resultados['sin_clima']['mae_test']:<20.0f} {resultados['con_clima']['mae_test']:<20.0f} {resultados['sin_clima']['mae_test'] - resultados['con_clima']['mae_test']:+.0f}")

print(f"\n{'='*80}")
print("IMPORTANCIA DE FEATURES CLIMÁTICAS")
print(f"{'='*80}")

imp_clima = importancia_por_tipo.get('Clima', 0) * 100
print(f"\n📊 Importancia total clima: {imp_clima:.2f}%")

if imp_clima > 5:
    print(f"   ✅ SIGNIFICATIVA: Clima tiene impacto importante (>{5}%)")
    print(f"   → Reincorporar negativos AUMENTÓ importancia de clima")
elif imp_clima > 2:
    print(f"   ⚠️  MODERADA: Clima tiene impacto menor (2-5%)")
elif imp_clima > 1:
    print(f"   ℹ️  BAJA: Clima tiene impacto mínimo (1-2%)")
else:
    print(f"   ❌ MÍNIMA: Clima tiene impacto despreciable (<1%)")

print(f"\n{'='*80}")
print("CONCLUSIONES")
print(f"{'='*80}")

mejora_r2_pct = (resultados['con_clima']['r2_test'] - resultados['sin_clima']['r2_test']) * 100

if mejora_r2_pct > 1:
    print(f"\n✅ CLIMA MEJORA SIGNIFICATIVAMENTE el modelo")
    print(f"   • Mejora R²: +{mejora_r2_pct:.2f} puntos porcentuales")
    print(f"   • Importancia clima: {imp_clima:.2f}%")
    print(f"\n💡 RECOMENDACIÓN: USAR modelo CON clima")
    print(f"   Los 253 casos negativos reincorporados SÍ ayudaron a")
    print(f"   que el modelo aprendiera mejor el efecto de temperatura")
elif mejora_r2_pct > 0.3:
    print(f"\n⚠️  CLIMA MEJORA MODERADAMENTE el modelo")
    print(f"   • Mejora R²: +{mejora_r2_pct:.2f} puntos porcentuales")
    print(f"   • Importancia clima: {imp_clima:.2f}%")
    print(f"\n💡 RECOMENDACIÓN: Considerar modelo CON clima")
    print(f"   Ganancia marginal pero podría ayudar en casos extremos")
else:
    print(f"\n❌ CLIMA NO MEJORA significativamente el modelo")
    print(f"   • Mejora R²: {mejora_r2_pct:+.2f} puntos porcentuales (mínima)")
    print(f"   • Importancia clima: {imp_clima:.2f}%")
    print(f"\n💡 RECOMENDACIÓN: MANTENER modelo SIN clima")
    print(f"   Incluso con 253 casos negativos, clima sigue siendo")
    print(f"   factor secundario en predicción horaria")

print(f"\n{'='*80}")
print("✅ ANÁLISIS COMPLETADO")
print(f"{'='*80}")
