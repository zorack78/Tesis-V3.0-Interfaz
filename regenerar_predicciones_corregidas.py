"""
Regenerar Predicciones con Perfil Qin Corregido
===============================================
Ejecuta la comparación de modelos ML usando el nuevo perfil Qin
sin data leakage, generando predicciones y métricas limpias.
"""

import sys
from pathlib import Path

# Importar la clase de la interfaz
sys.path.insert(0, str(Path(__file__).parent))
from interfaz_planificacion_qin_v1 import InterfazPlanificacionQin


def main():
    """Ejecuta comparación de modelos y genera predicciones"""
    print("\n" + "="*80)
    print("REGENERANDO PREDICCIONES CON PERFIL QIN CORREGIDO")
    print("="*80)
    print("\n✅ Corrección aplicada: Perfil Qin usa solo datos de entrenamiento")
    print("   (70% del dataset, hasta marzo 2025)")
    
    # Inicializar sistema
    print("\n📂 Cargando sistema...")
    sistema = InterfazPlanificacionQin()
    
    if not sistema.cargar_todo():
        print("❌ Error al cargar modelo")
        return
    
    print("\n✅ Sistema cargado correctamente")
    print(f"   - Modelo: {sistema.modelo}")
    print(f"   - Features: {len(sistema.features)}")
    print(f"   - Perfil Qin: {len(sistema.qin_perfil_hora)} horas")
    
    # Ejecutar comparación de modelos
    print("\n🔬 Ejecutando comparación de modelos ML...")
    print("   (Esto puede tomar 2-3 minutos)")
    
    texto_resultado, grafica = sistema.comparar_modelos_ml()
    
    print("\n" + "="*80)
    print("RESULTADO DE LA COMPARACIÓN")
    print("="*80)
    print(texto_resultado)
    
    if grafica is not None:
        print("\n✅ Gráfica generada correctamente")
    
    print("\n" + "="*80)
    print("ARCHIVOS GENERADOS")
    print("="*80)
    print("\n📁 outputs/metricas_ML/")
    print("   - predicciones_reales.csv (predicciones SIN leakage)")
    print("   - metricas_reales.json (métricas SIN leakage)")
    print("   - comparacion_modelos_ML.png/pdf")
    
    print("\n💡 Próximo paso:")
    print("   python comparar_antes_despues_correccion.py")
    print("   (para comparar con las predicciones antiguas CON leakage)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
