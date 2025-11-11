import pandas as pd

df = pd.read_csv('data/raw/BD_VolTotal_X_Hr_m3_UTC.csv')

print("=" * 70)
print("ANÁLISIS DE TIMESTAMPS")
print("=" * 70)

print(f"\nTotal de registros: {len(df)}")
print(f"\nPrimeras 20 filas:")
print(df.head(20))

print(f"\nÚltimas 20 filas:")
print(df.tail(20))

# Si cada registro es una hora, deberían ser registros consecutivos
total_hours_expected = len(df)
date_range = pd.date_range(start='2024-01-01', periods=total_hours_expected, freq='h')

print(f"\n📅 Si son datos horarios desde 2024-01-01:")
print(f"   - Registros esperados: {total_hours_expected}")
print(f"   - Fecha final esperada: {date_range[-1]}")
print(f"   - Meses cubiertos: {total_hours_expected / 24 / 30:.1f} meses")

print("\n✅ SOLUCIÓN:")
print("   Reconstruir timestamps con frecuencia horaria desde 2024-01-01")
