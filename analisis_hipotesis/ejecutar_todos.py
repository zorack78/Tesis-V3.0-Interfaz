# -*- coding: utf-8 -*-
"""
Script Maestro: Ejecuta Todos los Analisis de Hipotesis
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("ANALISIS COMPLETO DE HIPOTESIS - MODELO PREDICTIVO DEMANDA AGUA")
print("=" * 80)
print(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Lista de scripts a ejecutar
scripts = [
    # Analisis basicos
    ('01_analisis_correlaciones.py', 'Analisis de Correlaciones'),
    ('02_metricas_modelo.py', 'Metricas del Modelo'),
    ('03_importancia_variables.py', 'Importancia de Variables'),
    ('04_validacion_retrospectiva.py', 'Validacion Retrospectiva (Test Set)'),
    
    # Analisis descriptivo
    ('05_estadisticas_descriptivas.py', 'Estadisticas Descriptivas'),
    ('06_patrones_temporales.py', 'Patrones Temporales de Demanda'),
    ('07_temperatura_demanda.py', 'Temperatura vs Demanda'),
    ('08_volumen_estanques.py', 'Analisis de Volumen en Estanques'),
    
    # Analisis inferencial
    ('09_residuos_modelo.py', 'Residuos del Modelo (Diagnostico)'),
    ('10_error_condiciones.py', 'Error por Condiciones Operativas'),
    ('11_comparacion_datasets.py', 'Comparacion Train/Val/Test'),
    ('12_intervalos_confianza.py', 'Intervalos de Confianza'),
    
    # Interpretabilidad
    ('13_features_por_tipo.py', 'Features por Tipo (Interpretabilidad)')
]

# Directorio base (ya estamos en analisis_hipotesis cuando se ejecuta)
base_dir = Path('.')
resultados = []

# Ejecutar cada script
for i, (script, nombre) in enumerate(scripts, 1):
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(scripts)}] EJECUTANDO: {nombre}")
    print(f"Script: {script}")
    print(f"{'=' * 80}\n")
    
    script_path = base_dir / script
    
    try:
        # Ejecutar el script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Mostrar salida
        print(result.stdout)
        if result.stderr:
            print("ADVERTENCIAS:")
            print(result.stderr)
        
        resultados.append({
            'script': script,
            'nombre': nombre,
            'exito': True,
            'mensaje': 'Completado exitosamente'
        })
        
        print(f"[OK] {nombre} completado")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error en {nombre}")
        print(f"Codigo de salida: {e.returncode}")
        print(f"Salida de error:\n{e.stderr}")
        
        resultados.append({
            'script': script,
            'nombre': nombre,
            'exito': False,
            'mensaje': f'Error: {str(e)}'
        })
        
    except Exception as e:
        print(f"[ERROR] Error inesperado en {nombre}: {str(e)}")
        resultados.append({
            'script': script,
            'nombre': nombre,
            'exito': False,
            'mensaje': f'Error inesperado: {str(e)}'
        })

# Resumen final
print("\n" + "=" * 80)
print("RESUMEN DE EJECUCION")
print("=" * 80)

exitos = sum(1 for r in resultados if r['exito'])
fallos = len(resultados) - exitos

print(f"\nScripts ejecutados exitosamente: {exitos}/{len(scripts)}")
print(f"Scripts con errores: {fallos}/{len(scripts)}")

print("\nDETALLE:")
for r in resultados:
    estado = "[OK]" if r['exito'] else "[ERROR]"
    print(f"   {estado} {r['nombre']}: {r['mensaje']}")

# Listar archivos generados
print("\n" + "=" * 80)
print("ARCHIVOS GENERADOS")
print("=" * 80)

output_dir = Path('outputs')
if output_dir.exists():
    archivos = sorted(output_dir.glob('*'))
    
    # Agrupar por tipo
    imagenes = [f for f in archivos if f.suffix in ['.png', '.jpg']]
    csvs = [f for f in archivos if f.suffix == '.csv']
    otros = [f for f in archivos if f not in imagenes and f not in csvs]
    
    print(f"\nIMAGENES ({len(imagenes)}):")
    for img in imagenes:
        size_kb = img.stat().st_size / 1024
        print(f"   - {img.name} ({size_kb:.1f} KB)")
    
    print(f"\nARCHIVOS CSV ({len(csvs)}):")
    for csv in csvs:
        size_kb = csv.stat().st_size / 1024
        print(f"   - {csv.name} ({size_kb:.1f} KB)")
    
    total_size = sum(f.stat().st_size for f in archivos) / (1024 * 1024)
    print(f"\nTotal: {len(archivos)} archivos, {total_size:.2f} MB")
else:
    print("\nDirectorio de outputs no encontrado")

# Guardar reporte
print("\n" + "=" * 80)
print("GUARDANDO REPORTE")
print("=" * 80)

# Crear directorio outputs si no existe
outputs_dir = Path('outputs')
outputs_dir.mkdir(exist_ok=True)

reporte_path = outputs_dir / 'reporte_ejecucion.txt'
with open(reporte_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("REPORTE DE ANALISIS DE HIPOTESIS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Scripts ejecutados: {len(scripts)}\n")
    f.write(f"Exitosos: {exitos}\n")
    f.write(f"Fallidos: {fallos}\n\n")
    
    f.write("DETALLE DE EJECUCION:\n")
    f.write("-" * 80 + "\n")
    for r in resultados:
        f.write(f"\n{r['nombre']}\n")
        f.write(f"  Script: {r['script']}\n")
        f.write(f"  Estado: {'EXITO' if r['exito'] else 'ERROR'}\n")
        f.write(f"  Mensaje: {r['mensaje']}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("ARCHIVOS GENERADOS\n")
    f.write("=" * 80 + "\n\n")
    
    if output_dir.exists():
        for archivo in sorted(output_dir.glob('*')):
            if archivo.name != 'reporte_ejecucion.txt':
                size_kb = archivo.stat().st_size / 1024
                f.write(f"- {archivo.name} ({size_kb:.1f} KB)\n")

print(f"OK - Reporte guardado: {reporte_path}")

# Mensaje final
print("\n" + "=" * 80)
if fallos == 0:
    print("TODOS LOS ANALISIS COMPLETADOS EXITOSAMENTE")
    print("\nPuedes revisar los resultados en:")
    print(f"   {output_dir.absolute()}")
else:
    print("ANALISIS COMPLETADO CON ALGUNOS ERRORES")
    print(f"\nRevisa el reporte para mas detalles:")
    print(f"   {reporte_path.absolute()}")
print("=" * 80)
