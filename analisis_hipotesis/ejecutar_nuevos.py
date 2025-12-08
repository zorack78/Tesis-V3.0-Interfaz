# -*- coding: utf-8 -*-
"""
Script para ejecutar solo los analisis nuevos (5-13)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("EJECUTANDO ANALISIS ADICIONALES (5-13)")
print("=" * 80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

scripts = [
    # Analisis descriptivo
    ('05_estadisticas_descriptivas.py', 'Estadisticas Descriptivas'),
    ('06_patrones_temporales.py', 'Patrones Temporales'),
    ('07_temperatura_demanda.py', 'Temperatura vs Demanda'),
    ('08_volumen_estanques.py', 'Volumen en Estanques'),
    
    # Analisis inferencial
    ('09_residuos_modelo.py', 'Residuos del Modelo'),
    ('10_error_condiciones.py', 'Error por Condiciones'),
    ('11_comparacion_datasets.py', 'Comparacion Datasets'),
    ('12_intervalos_confianza.py', 'Intervalos de Confianza'),
    
    # Interpretabilidad
    ('13_features_por_tipo.py', 'Features por Tipo')
]

exitos = 0
fallos = 0

for i, (script, nombre) in enumerate(scripts, 1):
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(scripts)}] {nombre}")
    print(f"{'=' * 80}")
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        print(result.stdout)
        exitos += 1
        print(f"\n[OK] {nombre} completado")
    except subprocess.TimeoutExpired:
        print(f"\n[ERROR] Timeout en {nombre}")
        fallos += 1
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error en {nombre}")
        print(e.stderr)
        fallos += 1
    except Exception as e:
        print(f"\n[ERROR] Exception en {nombre}: {str(e)}")
        fallos += 1

print(f"\n{'=' * 80}")
print("RESUMEN FINAL")
print(f"{'=' * 80}")
print(f"Exitosos: {exitos}/{len(scripts)}")
print(f"Fallidos: {fallos}/{len(scripts)}")
print(f"Finalizacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")
