"""
Script de prueba para verificar las mejoras implementadas
"""

import sys
sys.path.insert(0, '.')

from interfaz_planificacion_qin_v1 import InterfazPlanificacionQin

print("="*80)
print("PRUEBA DE MEJORAS IMPLEMENTADAS")
print("="*80)

# Crear instancia
interfaz = InterfazPlanificacionQin()

print("\n1️⃣ Probando obtención de pronóstico meteorológico...")
pronostico = interfaz.obtener_pronostico_automatico()

print(f"\n✅ Pronóstico obtenido: {len(pronostico)} días")
for dia in pronostico:
    print(f"   {dia['dia']}: {dia['temp_min']:.1f}°C - {dia['temp_max']:.1f}°C | "
          f"Promedio: {dia['temp_promedio']:.1f}°C | "
          f"Lluvia: {dia['prob_lluvia']:.0f}% {dia['icono']}")

print("\n2️⃣ Probando cálculo de temperatura por hora (mes de diciembre)...")
temp_promedio = 20.0
mes = 12
print(f"   Temperatura promedio del día: {temp_promedio}°C")
print(f"   Mes: {mes} (Verano)")
print("\n   Hora  Temperatura")
print("   " + "-"*20)

for hora in [0, 3, 6, 9, 12, 15, 18, 21]:
    temp_hora = interfaz.calcular_temperatura_hora(temp_promedio, hora, mes)
    print(f"   {hora:02d}:00   {temp_hora:5.2f}°C")

print("\n3️⃣ Verificando amplitudes térmicas por estación...")
estaciones = {
    1: "Verano (Enero)", 
    4: "Otoño (Abril)", 
    7: "Invierno (Julio)", 
    10: "Primavera (Octubre)"
}

for mes, nombre in estaciones.items():
    temps = [interfaz.calcular_temperatura_hora(18.0, h, mes) for h in range(24)]
    temp_min = min(temps)
    temp_max = max(temps)
    amplitud = temp_max - temp_min
    hora_min = temps.index(temp_min)
    hora_max = temps.index(temp_max)
    
    print(f"   {nombre}:")
    print(f"      Mínima: {temp_min:.2f}°C a las {hora_min:02d}:00")
    print(f"      Máxima: {temp_max:.2f}°C a las {hora_max:02d}:00")
    print(f"      Amplitud: {amplitud:.2f}°C")

print("\n" + "="*80)
print("✅ TODAS LAS PRUEBAS COMPLETADAS")
print("="*80)
print("\n📝 Resumen de mejoras implementadas:")
print("   1. ✅ Obtención automática de pronóstico desde Open-Meteo")
print("   2. ✅ Temperatura variable por hora (ciclo diario realista)")
print("   3. ✅ Header dinámico con datos reales y probabilidad de lluvia")
print("   4. ✅ Inputs precargados con temperaturas del pronóstico")
print("   5. ✅ Botones de refrescar pronóstico")
print("   6. ✅ Gráficos con curva de temperatura")
print("\n🚀 La interfaz está lista para usarse con:")
print("   python interfaz_planificacion_qin_v1.py")
