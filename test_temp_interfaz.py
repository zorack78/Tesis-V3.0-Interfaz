"""
Análisis directo: verificar si temperatura influye en las predicciones
"""
from interfaz_planificacion_qin_v1 import InterfazPlanificacionQin
from datetime import datetime

# Crear interfaz
print("Cargando interfaz...")
app = InterfazPlanificacionQin()

# Hora fija: 14:00
hora = 14
dia_semana = 1  # Martes
mes = 6  # Junio

print(f"\n✅ Features del modelo: {len(app.features)}")
print(f"📋 Primeras 10 features: {app.features[:10]}")

# Probar con diferentes temperaturas
temperaturas = [15, 18, 20, 25, 28, 33]
resultados = []

print("\n" + "="*70)
print("🔬 PRUEBA DE SENSIBILIDAD A TEMPERATURA (hora 14:00, martes, junio)")
print("="*70)

for temp in temperaturas:
    # Crear features
    row = app.crear_features_prediccion(hora, dia_semana, mes, temp, app.df_completo)
    
    # Verificar temperatura en features
    temp_feature = row.get('clima_temp_c__mean_win_12h', None)
    
    # Predecir
    import pandas as pd
    X = pd.DataFrame([row])[app.features]
    q_net = app.modelo.predict(X)[0]
    
    # Calcular demanda
    demanda = abs(q_net) if q_net < 0 else 0
    
    resultados.append({
        'temperatura_input': temp,
        'temperatura_feature': temp_feature,
        'q_net': q_net,
        'demanda': demanda
    })
    
    print(f"\n🌡️  Temperatura: {temp}°C")
    print(f"   Feature temp: {temp_feature}°C")
    print(f"   Q_net: {q_net:,.0f} m³/hr")
    print(f"   Demanda: {demanda:,.0f} m³/hr")

# Análisis final
print("\n" + "="*70)
print("📊 RESUMEN")
print("="*70)

import pandas as pd
df_res = pd.DataFrame(resultados)
print(df_res[['temperatura_input', 'q_net', 'demanda']].to_string(index=False))

corr = df_res['temperatura_input'].corr(df_res['demanda'])
print(f"\n📈 Correlación Temperatura-Demanda: {corr:.3f}")

if corr > 0.1:
    print("✅ CORRECTO: Mayor temperatura → Mayor demanda")
elif corr < -0.1:
    print("❌ INVERTIDO: Mayor temperatura → Menor demanda")
else:
    print("⚠️  NEUTRAL: Temperatura no influye")
