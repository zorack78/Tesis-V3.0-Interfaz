"""
Script para corregir terminología en todo el proyecto:
- "hora pico" / "hora valle" → "horario de inflexión máximo/mínimo"
- "hora_pico" / "hora_valle" (variables) → mantener por compatibilidad pero actualizar outputs
"""

import os
import re
from pathlib import Path

# Archivos a corregir
archivos_corregir = [
    'interfaz_demanda_v1.py',
    'documentacion_completa_modelo.py',
    'investigacion_registros_perdidos.py',
    'analisis_determinacion_horas.py',
    'crear_variable_demanda.py',
    'entrenar_modelo_demanda.py'
]

print("="*80)
print("CORRECCIÓN DE TERMINOLOGÍA: PICO/VALLE → INFLEXIÓN")
print("="*80)

print("\n📝 Archivos a corregir:")
for archivo in archivos_corregir:
    print(f"   • {archivo}")

print("\n🔧 Cambios a realizar:")
print("   • 'Hora pico' → 'Horario de inflexión máximo'")
print("   • 'Hora valle' → 'Horario de inflexión mínimo'")
print("   • 'Pico' (en contexto) → 'Máximo'")
print("   • 'Valle' (en contexto) → 'Mínimo'")
print("   • Variables 'hora_pico'/'hora_valle' → mantener (compatibilidad)")

print("\n" + "="*80)
print("INICIO DE CORRECCIONES")
print("="*80)

for archivo in archivos_corregir:
    path = Path(archivo)
    if not path.exists():
        print(f"\n❌ {archivo}: No encontrado")
        continue
    
    print(f"\n📄 Procesando: {archivo}")
    
    with open(path, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    contenido_original = contenido
    cambios = 0
    
    # Mapeo de cambios (orden importa para evitar conflictos)
    reemplazos = [
        # Outputs y mensajes (case-insensitive para textos visibles)
        (r'\bHora pico\b', 'Horario de inflexión máximo'),
        (r'\bHora PICO\b', 'HORARIO DE INFLEXIÓN MÁXIMO'),
        (r'\bhora pico\b', 'horario de inflexión máximo'),
        (r'\bHora valle\b', 'Horario de inflexión mínimo'),
        (r'\bHora VALLE\b', 'HORARIO DE INFLEXIÓN MÍNIMO'),
        (r'\bhora valle\b', 'horario de inflexión mínimo'),
        
        # Labels y títulos
        (r'horas pico/valle', 'horarios de inflexión'),
        (r'Horas pico/valle', 'Horarios de inflexión'),
        (r'horas pico y valle', 'horarios de inflexión'),
        
        # Contextuales
        (r'\bPico máximo\b', 'Inflexión máxima'),
        (r'\bValle mínimo\b', 'Inflexión mínima'),
        (r'\bpico máximo\b', 'inflexión máxima'),
        (r'\bvalle mínimo\b', 'inflexión mínima'),
        
        # En labels de gráficos
        (r"label=f'Pico:", "label=f'Máximo:"),
        (r"label=f'Valle:", "label=f'Mínimo:"),
        
        # Comentarios técnicos
        (r'# Identificar horas pico', '# Identificar horarios de inflexión'),
        (r'# Hora pico', '# Horario de inflexión máximo'),
        (r'# Hora valle', '# Horario de inflexión mínimo'),
    ]
    
    for patron, reemplazo in reemplazos:
        nuevos_cambios = len(re.findall(patron, contenido))
        if nuevos_cambios > 0:
            contenido = re.sub(patron, reemplazo, contenido)
            cambios += nuevos_cambios
            print(f"   ✓ '{patron}' → '{reemplazo}' ({nuevos_cambios} ocurrencias)")
    
    if cambios > 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contenido)
        print(f"   ✅ {cambios} cambios aplicados")
    else:
        print(f"   ℹ️  Sin cambios necesarios")

print("\n" + "="*80)
print("✅ CORRECCIÓN COMPLETADA")
print("="*80)

print("\n📋 RESUMEN:")
print("   • Se mantuvieron nombres de variables (hora_pico, hora_valle)")
print("   • Se actualizaron todos los textos visibles al usuario")
print("   • Los archivos de datos (.json, .csv) NO se modifican")
print("   • Terminología consistente con interfaz_demanda_v1.py")

print("\n💡 PRÓXIMOS PASOS:")
print("   1. Revisar interfaz_demanda_v1.py (ya debe estar correcto)")
print("   2. Verificar que gráficos usen terminología correcta")
print("   3. Actualizar documentación si es necesario")
