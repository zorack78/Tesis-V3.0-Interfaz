"""
Validación del Impacto de la Temperatura en las Predicciones

Prueba que temperatura alta → demanda alta
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("VALIDACIÓN: IMPACTO DE TEMPERATURA EN PREDICCIONES")
print("="*80)

# ==================== 1. CARGAR MODELO ====================
print("\n[1/4] Cargando modelo con temperatura...")

model_path = Path('models/demanda_temperatura/demanda_temperatura_xgboost_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

features_path = Path('models/demanda_temperatura/features.txt')
with open(features_path, 'r') as f:
    features = [line.strip() for line in f.readlines()]

print(f"✅ Modelo cargado")
print(f"✅ Features: {len(features)}")

# ==================== 2. CARGAR DATOS HISTÓRICOS ====================
print("\n[2/4] Cargando datos históricos...")

data_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(data_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

print(f"✅ Datos cargados: {len(df)} registros")

# ==================== 3. CREAR FUNCIÓN DE PREDICCIÓN ====================

def crear_features_con_temperatura(fecha, hora, temperatura, df_historico):
    """Crea features para predicción con temperatura específica"""
    
    # Features básicas
    features_dict = {
        'hour': hora,
        'day_of_week': fecha.weekday(),
        'month': fecha.month,
        'is_weekend': 1 if fecha.weekday() >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * hora / 24),
        'hour_cos': np.cos(2 * np.pi * hora / 24),
        'day_of_week_sin': np.sin(2 * np.pi * fecha.weekday() / 7),
        'day_of_week_cos': np.cos(2 * np.pi * fecha.weekday() / 7),
        'feriado': 0,
        'temporada_turistica_alta': 1 if fecha.month in [1, 2, 7, 12] else 0,
    }
    
    # Estimar LAGs basados en datos similares
    datos_similares = df_historico[
        (df_historico['timestamp_utc'].dt.hour == hora) &
        (df_historico['timestamp_utc'].dt.month == fecha.month)
    ]
    
    if len(datos_similares) > 0:
        demanda_mediana = float(datos_similares['Demanda_m3_hr'].median())
        demanda_std = float(datos_similares['Demanda_m3_hr'].std())
    else:
        demanda_mediana = 12000.0
        demanda_std = 2000.0
    
    # LAGs
    features_dict.update({
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
        'Demanda_ratio_vs_24h': 1.0,
    })
    
    # TEMPERATURA
    features_dict['temperatura'] = temperatura
    
    return features_dict

# ==================== 4. PRUEBAS CON DIFERENTES TEMPERATURAS ====================
print("\n[3/4] Generando predicciones con diferentes temperaturas...")

# Fecha de prueba: 15 de diciembre (verano)
fecha_prueba = pd.to_datetime('2025-12-15')

# Rango de temperaturas a probar
temperaturas = [5, 10, 15, 20, 25, 30, 35]
colores_temp = ['#0000ff', '#3366ff', '#66b3ff', '#ffcc00', '#ff9933', '#ff6600', '#ff0000']

# Generar predicciones para cada temperatura (24 horas)
resultados = {}

for temp in temperaturas:
    predicciones_24h = []
    
    for hora in range(24):
        features_dict = crear_features_con_temperatura(fecha_prueba, hora, temp, df)
        X = pd.DataFrame([features_dict])[features]
        pred = model.predict(X)[0]
        predicciones_24h.append(pred)
    
    resultados[temp] = {
        'predicciones': predicciones_24h,
        'promedio': np.mean(predicciones_24h),
        'total': np.sum(predicciones_24h)
    }
    
    print(f"   🌡️  {temp:>2}°C → Promedio: {resultados[temp]['promedio']:>8,.0f} m³/hr | Total: {resultados[temp]['total']:>10,.0f} m³/día")

# ==================== 5. ANÁLISIS DE RESULTADOS ====================
print("\n[4/4] Análisis de resultados...")

print("\n" + "="*80)
print("ANÁLISIS: IMPACTO DE LA TEMPERATURA")
print("="*80)

# Comparar extremos
demanda_frio = resultados[5]['promedio']
demanda_calor = resultados[35]['promedio']
diferencia_abs = demanda_calor - demanda_frio
diferencia_pct = (diferencia_abs / demanda_frio) * 100

print(f"\n📊 Comparación Temperaturas Extremas:")
print(f"   Frío (5°C):   {demanda_frio:,.0f} m³/hr")
print(f"   Calor (35°C): {demanda_calor:,.0f} m³/hr")
print(f"   Diferencia:   {diferencia_abs:+,.0f} m³/hr ({diferencia_pct:+.1f}%)")

if demanda_calor > demanda_frio:
    print(f"\n✅ VALIDACIÓN CORRECTA: Temperatura alta → Demanda mayor")
else:
    print(f"\n❌ VALIDACIÓN FALLIDA: Temperatura no aumenta demanda")

# Calcular tendencia
temperaturas_arr = np.array(temperaturas)
demandas_promedio = np.array([resultados[t]['promedio'] for t in temperaturas])

# Correlación
from scipy.stats import pearsonr
corr, p_value = pearsonr(temperaturas_arr, demandas_promedio)

print(f"\n📈 Correlación Temperatura vs Demanda Predicha:")
print(f"   Pearson r: {corr:.4f}")
print(f"   p-value:   {p_value:.2e}")

if corr > 0:
    print(f"   ✅ Correlación POSITIVA (esperada)")
else:
    print(f"   ❌ Correlación NEGATIVA (inesperada)")

# ==================== 6. VISUALIZACIÓN ====================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIÓN")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Validación: Impacto de Temperatura en Predicciones', fontsize=16, fontweight='bold')

# Subplot 1: Series temporales por temperatura
ax1 = axes[0, 0]
for temp, color in zip(temperaturas, colores_temp):
    ax1.plot(range(24), resultados[temp]['predicciones'], 
             label=f'{temp}°C', linewidth=2, color=color, marker='o', markersize=4)

ax1.set_xlabel('Hora del día', fontsize=12)
ax1.set_ylabel('Demanda (m³/hr)', fontsize=12)
ax1.set_title('Demanda Predicha por Hora según Temperatura', fontsize=14)
ax1.legend(title='Temperatura', loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

# Subplot 2: Demanda promedio vs temperatura
ax2 = axes[0, 1]
scatter = ax2.scatter(temperaturas_arr, demandas_promedio, 
                     c=temperaturas_arr, cmap='coolwarm', s=200, edgecolors='black', linewidths=2)
ax2.plot(temperaturas_arr, demandas_promedio, 'k--', alpha=0.5, linewidth=2)

# Agregar línea de tendencia
z = np.polyfit(temperaturas_arr, demandas_promedio, 1)
p = np.poly1d(z)
ax2.plot(temperaturas_arr, p(temperaturas_arr), "r-", linewidth=2, label=f'Tendencia (y={z[0]:.1f}x+{z[1]:.0f})')

ax2.set_xlabel('Temperatura (°C)', fontsize=12)
ax2.set_ylabel('Demanda Promedio (m³/hr)', fontsize=12)
ax2.set_title(f'Demanda vs Temperatura (r={corr:.3f})', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.colorbar(scatter, ax=ax2, label='Temperatura (°C)')

# Subplot 3: Demanda total diaria vs temperatura
ax3 = axes[1, 0]
totales = [resultados[t]['total'] for t in temperaturas]
bars = ax3.bar(temperaturas, totales, color=colores_temp, edgecolor='black', linewidth=1.5)

# Agregar valores en las barras
for bar, total in zip(bars, totales):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{total/1000:.0f}k',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xlabel('Temperatura (°C)', fontsize=12)
ax3.set_ylabel('Demanda Total Diaria (m³)', fontsize=12)
ax3.set_title('Demanda Total Diaria según Temperatura', fontsize=14)
ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Incremento relativo respecto a 15°C (baseline)
ax4 = axes[1, 1]
baseline_temp = 15
baseline_demanda = resultados[baseline_temp]['promedio']
incrementos = [(resultados[t]['promedio'] - baseline_demanda) / baseline_demanda * 100 for t in temperaturas]

colors_incremento = ['blue' if inc < 0 else 'red' for inc in incrementos]
bars2 = ax4.bar(temperaturas, incrementos, color=colors_incremento, alpha=0.7, edgecolor='black', linewidth=1.5)

# Agregar valores
for bar, inc in zip(bars2, incrementos):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{inc:+.1f}%',
            ha='center', va='bottom' if inc > 0 else 'top', fontsize=10, fontweight='bold')

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Temperatura (°C)', fontsize=12)
ax4.set_ylabel('Variación Demanda (%)', fontsize=12)
ax4.set_title(f'Variación % respecto a {baseline_temp}°C (baseline)', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Guardar
output_path = Path('outputs/validacion_temperatura_predicciones.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {output_path}")

print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)

if corr > 0.1 and demanda_calor > demanda_frio:
    print("\n✅ VALIDACIÓN EXITOSA:")
    print(f"   • Temperatura alta ({max(temperaturas)}°C) produce demanda mayor")
    print(f"   • Temperatura baja ({min(temperaturas)}°C) produce demanda menor")
    print(f"   • Correlación positiva: r={corr:.3f}")
    print(f"   • Incremento del {diferencia_pct:.1f}% entre extremos")
    print(f"\n   El modelo responde correctamente a la temperatura")
elif abs(corr) < 0.05:
    print("\n⚠️  EFECTO MÍNIMO:")
    print(f"   • Correlación muy débil: r={corr:.3f}")
    print(f"   • Temperatura tiene poco impacto en predicciones")
    print(f"   • Esto es esperado dado el bajo ranking (#15 de 22 features)")
else:
    print("\n❌ COMPORTAMIENTO INESPERADO:")
    print(f"   • Correlación: r={corr:.3f}")
    print(f"   • Revisar lógica del modelo")

print("\n" + "="*80)
