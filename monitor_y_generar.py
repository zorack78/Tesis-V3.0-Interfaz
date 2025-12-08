"""
Script que monitorea la generación de predicciones_reales.csv
y ejecuta automáticamente generar_scatter_plot_real.py
"""

import time
from pathlib import Path
import subprocess
import sys


def main():
    csv_path = Path('outputs/metricas_ML/predicciones_reales.csv')
    
    print("\n" + "="*70)
    print("🔍 MONITOREANDO GENERACIÓN DE PREDICCIONES")
    print("="*70)
    print(f"\nEsperando archivo: {csv_path}")
    print("\n📋 INSTRUCCIONES:")
    print("   1. Abre http://127.0.0.1:7867 en tu navegador")
    print("   2. Ve al tab '🔬 Comparación Modelos ML'")
    print("   3. Click en 'Ejecutar Comparación'")
    print("   4. Este script detectará cuando termine y generará el scatter plot")
    print("\n⏳ Revisando cada 5 segundos...\n")
    
    intentos = 0
    max_intentos = 240  # 20 minutos máximo
    
    while intentos < max_intentos:
        if csv_path.exists():
            # Verificar que el archivo tenga contenido
            try:
                size = csv_path.stat().st_size
                if size > 1000:  # Al menos 1KB
                    print(f"\n✅ Archivo detectado! ({size:,} bytes)")
                    print("⏱️ Esperando 2 segundos para asegurar escritura completa...")
                    time.sleep(2)
                    
                    print("\n🚀 Ejecutando generar_scatter_plot_real.py...\n")
                    print("="*70 + "\n")
                    
                    # Ejecutar script de generación
                    result = subprocess.run(
                        [sys.executable, 'generar_scatter_plot_real.py'],
                        capture_output=False
                    )
                    
                    if result.returncode == 0:
                        print("\n✅ PROCESO COMPLETADO CON ÉXITO")
                    else:
                        print(f"\n⚠️ El script terminó con código: {result.returncode}")
                    
                    return result.returncode
            except Exception as e:
                print(f"⚠️ Error leyendo archivo: {e}")
        
        # Mostrar progreso
        if intentos % 12 == 0:  # Cada minuto
            mins = intentos // 12
            print(f"⏳ Esperando... ({mins} min)")
        
        time.sleep(5)
        intentos += 1
    
    print("\n⏱️ Tiempo de espera agotado (20 minutos)")
    print("❌ El archivo predicciones_reales.csv no fue generado")
    print("\n💡 Verifica que:")
    print("   • La interfaz Gradio esté corriendo")
    print("   • Hayas ejecutado la comparación de modelos")
    print("   • No haya errores en la terminal de Gradio")
    return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        print("\n✨ Presiona Enter para cerrar...")
        input()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoreo cancelado por el usuario")
        sys.exit(1)
