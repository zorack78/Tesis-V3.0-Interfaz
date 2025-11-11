"""
Script de diagnóstico para encontrar el error exacto
"""
import sys
sys.path.insert(0, '.')

from interfaz_volumen_total_v4 import SistemaVolumenTotal

# Crear instancia
sistema = SistemaVolumenTotal()

# Cargar modelo y datos
print("1. Cargando modelo...")
if not sistema.cargar_modelo_y_datos():
    print("❌ Error cargando modelo")
    exit(1)

print("\n2. Probando predicción simple...")
try:
    pred, info = sistema.predecir_volumen("2025-12-25", 8, 15.0, 70.0, 0.0)
    print(f"✅ Predicción: {pred}")
    print(f"Info:\n{info}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Probando crear_features directamente...")
try:
    from datetime import datetime
    fecha = datetime(2025, 12, 25)
    print(f"Fecha: {fecha}")
    print(f"Fecha.weekday(): {fecha.weekday()}")
    print(f"Tipo weekday: {type(fecha.weekday())}")
    
    features = sistema.crear_features(fecha, 8, 15.0, 70.0, 0.0)
    print(f"✅ Features creadas: {len(features)} features")
    print(f"Primeras 5: {list(features.items())[:5]}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Probando _estimar_lags directamente...")
try:
    from datetime import datetime
    fecha = datetime(2025, 12, 25)
    lags = sistema._estimar_lags(fecha, 8)
    print(f"✅ LAGs estimados: {lags}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
