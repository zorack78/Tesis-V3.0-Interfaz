"""
Script de prueba para identificar el error específico
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Simular el problema
fecha_str = "2025-12-25"
hora = 8

fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
print(f"Fecha tipo: {type(fecha)}")
print(f"Fecha valor: {fecha}")
print(f"Weekday tipo: {type(fecha.weekday())}")
print(f"Weekday valor: {fecha.weekday()}")

# Crear features como en el código
features = {}
features['day_of_week'] = fecha.weekday()
features['day_of_week_sin'] = np.sin(2 * np.pi * fecha.weekday() / 7)

print(f"\nFeatures creadas:")
print(features)

# Crear DataFrame
X = pd.DataFrame([features])
print(f"\nDataFrame:")
print(X)
print(f"Tipo de columna day_of_week: {type(X['day_of_week'])}")
print(f"Valor de day_of_week: {X['day_of_week'].iloc[0]}")

# Verificar si hay problema con Series
try:
    test = X['day_of_week']
    print(f"\nTest Series: {test}")
    print(f"Test tipo: {type(test)}")
    # Intentar llamar como función (esto causaría el error)
    # test()  # Esto daría 'Series' object is not callable
except Exception as e:
    print(f"Error: {e}")

print("\n✅ No hay error en este script de prueba")
