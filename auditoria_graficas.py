"""
Análisis de Scripts con Datos Hardcodeados o Sintéticos
========================================================

Este reporte identifica todos los scripts que generan gráficas con
datos que NO provienen directamente de predicciones reales.
"""

import os
from pathlib import Path


def main():
    print("\n" + "="*70)
    print("AUDITORÍA: SCRIPTS CON DATOS HARDCODEADOS O SINTÉTICOS")
    print("="*70)
    
    print("\n📋 RESUMEN DE SCRIPTS DE GENERACIÓN DE GRÁFICAS:\n")
    
    # Script 1: generar_graficas_metricas_rapido.py
    print("1️⃣ generar_graficas_metricas_rapido.py")
    print("   ❌ USO: Datos sintéticos")
    print("   📍 PROBLEMA:")
    print("      • Línea 29-47: METRICAS_EJEMPLO con valores hardcodeados")
    print("        - XGBoost: R²=0.9905, MAE=269 (correcto)")
    print("        - RandomForest: R²=0.9897, MAE=278 (INCORRECTO)")
    print("        - LightGBM: R²=0.9923, MAE=261 (correcto)")
    print("      • Línea 180-210: generar_datos_sinteticos_predicciones()")
    print("        - Usa np.random.normal() para generar predicciones fake")
    print("        - No son predicciones reales del modelo")
    print("   📊 GRÁFICAS AFECTADAS:")
    print("      • 01_comparacion_metricas_completa.png")
    print("      • 02_*_ranking.png (4 archivos)")
    print("      • 03_prediccion_vs_real_*.png (3 archivos)")
    print("      • 04_comparacion_predicciones_unificada.png")
    print("      • 05_scatter_predicho_vs_real.png")
    print("   🎯 ESTADO: ⚠️ NO USAR PARA TESIS")
    print()
    
    # Script 2: generar_graficas_metricas_simple.py
    print("2️⃣ generar_graficas_metricas_simple.py")
    print("   ✅ USO: Solo métricas reales (no genera scatter plots)")
    print("   📍 DATOS:")
    print("      • Línea 23-41: METRICAS_REALES con valores de interfaz Gradio")
    print("        - Tomados de ejecución del 04/12/2025 22:16")
    print("        - XGBoost: R²=0.9905, MAE=269")
    print("        - RandomForest: R²=0.9842, MAE=295")
    print("        - LightGBM: R²=0.9923, MAE=261")
    print("   📊 GRÁFICAS GENERADAS:")
    print("      • 01_comparacion_metricas_completa_REAL.png")
    print("      • reporte_metricas_modelos_REAL.txt")
    print("   🎯 ESTADO: ✅ APTO PARA TESIS (solo métricas resumen)")
    print()
    
    # Script 3: generar_graficas_metricas_reales.py
    print("3️⃣ generar_graficas_metricas_reales.py")
    print("   ❌ USO: Intento fallido de entrenar modelos desde cero")
    print("   📍 PROBLEMA:")
    print("      • Línea 45: Llama método inexistente calcular_umbrales_bisagra()")
    print("      • Línea 51: _agregar_features_categoricas() falla sin umbrales")
    print("      • Línea 72+: Faltan 11+ features de ingeniería")
    print("   🎯 ESTADO: ⚠️ NO FUNCIONAL - ABANDONADO")
    print()
    
    # Script 4: generar_graficas_metricas_ml.py
    print("4️⃣ generar_graficas_metricas_ml.py")
    print("   🔍 Revisando contenido...")
    
    path_ml = Path('generar_graficas_metricas_ml.py')
    if path_ml.exists():
        with open(path_ml, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'METRICAS_EJEMPLO' in content or 'np.random' in content:
            print("   ❌ USO: Datos sintéticos o hardcodeados detectados")
        elif 'joblib.load' in content or 'modelo.predict' in content:
            print("   ⚠️ USO: Carga modelos pero necesita verificación")
        else:
            print("   ❓ USO: Requiere inspección manual")
        
        print("   📊 GRÁFICAS GENERADAS:")
        print("      • 01_comparacion_metricas_completa.png")
        print("      • 02_*_ranking.png")
        print("   🎯 ESTADO: ⚠️ VERIFICAR MANUALMENTE")
    else:
        print("   ℹ️  Archivo no encontrado en directorio actual")
    print()
    
    # Script 5: generar_scatter_plot_real.py
    print("5️⃣ generar_scatter_plot_real.py")
    print("   ✅ USO: Datos reales exportados de interfaz")
    print("   📍 DATOS:")
    print("      • Lee predicciones_reales.csv (2,256 registros)")
    print("      • Generado por interfaz_planificacion_qin_v1.py")
    print("      • Contiene predicciones reales sobre test set")
    print("   📊 GRÁFICAS GENERADAS:")
    print("      • 05_scatter_predicho_vs_real_REAL.png")
    print("      • reporte_scatter_plot_REAL.txt")
    print("   🎯 ESTADO: ✅ APTO PARA TESIS")
    print()
    
    print("="*70)
    print("📌 RECOMENDACIONES PARA TESIS:")
    print("="*70)
    print()
    print("✅ USAR (Rigor científico garantizado):")
    print("   • 01_comparacion_metricas_completa_REAL.png")
    print("   • 05_scatter_predicho_vs_real_REAL.png")
    print("   • reporte_metricas_modelos_REAL.txt")
    print("   • reporte_scatter_plot_REAL.txt")
    print()
    print("❌ NO USAR (Datos sintéticos o hardcodeados incorrectos):")
    print("   • 01_comparacion_metricas_completa.png (sin REAL)")
    print("   • 02_*_ranking.png")
    print("   • 03_prediccion_vs_real_*.png")
    print("   • 04_comparacion_predicciones_unificada.png")
    print("   • 05_scatter_predicho_vs_real.png (sin REAL)")
    print("   • reporte_metricas_modelos.txt (sin REAL)")
    print()
    print("🔄 ALTERNATIVA:")
    print("   Para gráficas de series temporales de predicciones:")
    print("   • Usar tab '📊 Predicción Próximas Horas' de interfaz Gradio")
    print("   • Exportar gráficas interactivas directamente")
    print("   • O modificar interfaz para exportar predicciones temporales")
    print()
    print("="*70)
    print("📁 ARCHIVOS A CONSERVAR EN outputs/metricas_ML/:")
    print("="*70)
    print()
    
    metricas_dir = Path('outputs/metricas_ML')
    if metricas_dir.exists():
        archivos_reales = []
        archivos_sinteticos = []
        
        for archivo in metricas_dir.glob('*'):
            if archivo.is_file():
                nombre = archivo.name
                if 'REAL' in nombre or nombre in ['predicciones_reales.csv', 
                                                   'metricas_reales.json']:
                    archivos_reales.append(nombre)
                elif nombre.endswith('.png') or nombre.endswith('.txt'):
                    archivos_sinteticos.append(nombre)
        
        print("✅ CONSERVAR (datos reales):")
        for archivo in sorted(archivos_reales):
            print(f"   • {archivo}")
        
        print("\n❌ ELIMINAR (datos sintéticos):")
        for archivo in sorted(archivos_sinteticos):
            print(f"   • {archivo}")
    
    print("\n" + "="*70)
    print("💡 PARA GENERAR GRÁFICAS REALES:")
    print("="*70)
    print()
    print("1. Ejecuta: python interfaz_planificacion_qin_v1.py")
    print("2. Abre: http://127.0.0.1:7867")
    print("3. Tab: '🔬 Comparación Modelos ML'")
    print("4. Click: 'Ejecutar Comparación'")
    print("5. Espera exportación de predicciones_reales.csv")
    print("6. Ejecuta: python generar_scatter_plot_real.py")
    print()
    print("="*70)


if __name__ == '__main__':
    main()
