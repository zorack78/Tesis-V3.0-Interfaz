#!/usr/bin/env python3
"""
Interfaz Multi-Modelo Gradio - Sistema Predictivo ESVAL
Permite comparar XGBoost, RandomForest y LightGBM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import joblib
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


class ModeloComparativo:
    """Clase para manejar múltiples modelos"""
    
    def __init__(self):
        self.modelos = {}
        self.datos_cargados = False
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
    def cargar_datos(self):
        """Carga datos procesados"""
        try:
            print("📊 Cargando datos...")
            data_path = Path("data/processed")
            
            # Verificar que existen los archivos
            if not data_path.exists():
                error_msg = f"❌ No existe el directorio: {data_path}"
                print(error_msg)
                return False, error_msg
            
            train_file = data_path / "data_train.csv"
            val_file = data_path / "data_validation.csv"
            test_file = data_path / "data_test.csv"
            
            if not train_file.exists():
                error_msg = f"❌ No existe: {train_file}"
                print(error_msg)
                return False, error_msg
            
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
            test_df = pd.read_csv(test_file)
            
            # Limpiar nombres de columnas (quitar espacios)
            train_df.columns = train_df.columns.str.strip()
            val_df.columns = val_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()
            
            # Verificar que existe la columna target
            # Buscar la columna de volumen (puede tener diferentes nombres)
            possible_targets = ['volumen_total_m3', 'Volumen_Total_m3', 'volumen_total']
            target_col = None
            for col_name in possible_targets:
                if col_name in train_df.columns:
                    target_col = col_name
                    break
            
            if target_col is None:
                error_msg = f"❌ Columna de volumen no encontrada. Columnas disponibles: {train_df.columns.tolist()[:15]}"
                print(error_msg)
                return False, error_msg
            
            # Separar features y target
            exclude_cols = [target_col, 'timestamp', 'timestamp_utc', 'fecha_hora_local']
            feature_cols = [col for col in train_df.columns 
                          if col not in exclude_cols]
            
            # Filtrar solo columnas numéricas
            X_train_all = train_df[feature_cols]
            numeric_cols = X_train_all.select_dtypes(include=[np.number]).columns.tolist()
            
            # Excluir también columnas de fechas/strings
            exclude_types = ['nombre_feriado', 'fecha_feriado_oficial', 'fecha_feriado_observado']
            numeric_cols = [col for col in numeric_cols if col not in exclude_types]
            
            self.X_train = train_df[numeric_cols]
            self.y_train = train_df[target_col]
            self.X_val = val_df[numeric_cols]
            self.y_val = val_df[target_col]
            self.X_test = test_df[numeric_cols]
            self.y_test = test_df[target_col]
            self.feature_names = numeric_cols
            
            self.datos_cargados = True
            msg = f"✅ Datos cargados correctamente\n\nTrain: {len(train_df)} registros\nValidation: {len(val_df)} registros\nTest: {len(test_df)} registros\nFeatures numéricas: {len(numeric_cols)}"
            print(msg)
            return True, msg
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error cargando datos:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return False, error_msg
    
    def entrenar_xgboost(self, n_estimators=300, max_depth=10, learning_rate=0.1):
        """Entrena modelo XGBoost"""
        print("\n🚀 Entrenando XGBoost...")
        
        modelo = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        modelo.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        self.modelos['xgboost'] = modelo
        
        # Evaluar
        y_val_pred = modelo.predict(self.X_val)
        y_test_pred = modelo.predict(self.X_test)
        
        metrics_val = calculate_metrics(self.y_val.values, y_val_pred)
        metrics_test = calculate_metrics(self.y_test.values, y_test_pred)
        
        print(f"✅ XGBoost entrenado")
        print(f"   Validación - R²: {metrics_val['r2']:.4f}, RMSE: {metrics_val['rmse']:.0f}")
        print(f"   Test - R²: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.0f}")
        
        return metrics_val, metrics_test
    
    def entrenar_randomforest(self, n_estimators=300, max_depth=20):
        """Entrena modelo RandomForest"""
        print("\n🌲 Entrenando RandomForest...")
        
        modelo = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        modelo.fit(self.X_train, self.y_train)
        
        self.modelos['randomforest'] = modelo
        
        # Evaluar
        y_val_pred = modelo.predict(self.X_val)
        y_test_pred = modelo.predict(self.X_test)
        
        metrics_val = calculate_metrics(self.y_val.values, y_val_pred)
        metrics_test = calculate_metrics(self.y_test.values, y_test_pred)
        
        print(f"✅ RandomForest entrenado")
        print(f"   Validación - R²: {metrics_val['r2']:.4f}, RMSE: {metrics_val['rmse']:.0f}")
        print(f"   Test - R²: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.0f}")
        
        return metrics_val, metrics_test
    
    def entrenar_lightgbm(self, n_estimators=300, max_depth=10, learning_rate=0.1):
        """Entrena modelo LightGBM"""
        print("\n💡 Entrenando LightGBM...")
        
        modelo = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        modelo.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        
        self.modelos['lightgbm'] = modelo
        
        # Evaluar
        y_val_pred = modelo.predict(self.X_val)
        y_test_pred = modelo.predict(self.X_test)
        
        metrics_val = calculate_metrics(self.y_val.values, y_val_pred)
        metrics_test = calculate_metrics(self.y_test.values, y_test_pred)
        
        print(f"✅ LightGBM entrenado")
        print(f"   Validación - R²: {metrics_val['r2']:.4f}, RMSE: {metrics_val['rmse']:.0f}")
        print(f"   Test - R²: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.0f}")
        
        return metrics_val, metrics_test
    
    def comparar_modelos(self, modelos_seleccionados):
        """Compara los modelos seleccionados"""
        if not self.datos_cargados:
            return "❌ Primero debes cargar los datos", None
        
        resultados = []
        
        for nombre_modelo in modelos_seleccionados:
            if nombre_modelo not in self.modelos:
                resultados.append(f"⚠️ {nombre_modelo.upper()} no está entrenado")
                continue
            
            modelo = self.modelos[nombre_modelo]
            
            # Predicciones
            y_val_pred = modelo.predict(self.X_val)
            y_test_pred = modelo.predict(self.X_test)
            
            # Métricas
            metrics_val = calculate_metrics(self.y_val.values, y_val_pred)
            metrics_test = calculate_metrics(self.y_test.values, y_test_pred)
            
            resultados.append(f"\n{'='*60}")
            resultados.append(f"📊 {nombre_modelo.upper()}")
            resultados.append(f"{'='*60}")
            resultados.append(f"\nValidación:")
            resultados.append(f"  • R²: {metrics_val['r2']:.4f}")
            resultados.append(f"  • RMSE: {metrics_val['rmse']:,.0f} m³")
            resultados.append(f"  • MAE: {metrics_val['mae']:,.0f} m³")
            resultados.append(f"  • MAPE: {metrics_val['mape']:.2f}%")
            resultados.append(f"\nTest:")
            resultados.append(f"  • R²: {metrics_test['r2']:.4f}")
            resultados.append(f"  • RMSE: {metrics_test['rmse']:,.0f} m³")
            resultados.append(f"  • MAE: {metrics_test['mae']:,.0f} m³")
            resultados.append(f"  • MAPE: {metrics_test['mape']:.2f}%")
        
        # Crear gráfica comparativa
        fig = self.crear_grafica_comparativa(modelos_seleccionados)
        
        return "\n".join(resultados), fig
    
    def crear_grafica_comparativa(self, modelos_seleccionados):
        """Crea gráfica comparando predicciones de los modelos"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Comparación de Modelos - Predicción de Demanda de Agua', 
                     fontsize=16, fontweight='bold')
        
        # Usar solo primeros 200 puntos para visualización clara
        n_points = 200
        x_plot = range(n_points)
        
        # Subplot 1: Predicciones vs Real (Test)
        ax1 = axes[0, 0]
        ax1.plot(x_plot, self.y_test.values[:n_points], 
                label='Real', color='black', linewidth=2, alpha=0.7)
        
        for nombre_modelo in modelos_seleccionados:
            if nombre_modelo in self.modelos:
                y_pred = self.modelos[nombre_modelo].predict(self.X_test)
                ax1.plot(x_plot, y_pred[:n_points], 
                        label=nombre_modelo.upper(), linewidth=1.5, alpha=0.8)
        
        ax1.set_title('Predicciones en Test Set', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Muestra')
        ax1.set_ylabel('Volumen (m³)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: R² Score comparación
        ax2 = axes[0, 1]
        r2_scores = []
        nombres = []
        
        for nombre_modelo in modelos_seleccionados:
            if nombre_modelo in self.modelos:
                y_pred = self.modelos[nombre_modelo].predict(self.X_test)
                r2 = r2_score(self.y_test.values, y_pred)
                r2_scores.append(r2)
                nombres.append(nombre_modelo.upper())
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(nombres)]
        bars = ax2.bar(nombres, r2_scores, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('R² Score Comparación', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.set_ylim([0.8, 1.0])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en barras
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 3: RMSE comparación
        ax3 = axes[1, 0]
        rmse_scores = []
        
        for nombre_modelo in modelos_seleccionados:
            if nombre_modelo in self.modelos:
                y_pred = self.modelos[nombre_modelo].predict(self.X_test)
                rmse = np.sqrt(mean_squared_error(self.y_test.values, y_pred))
                rmse_scores.append(rmse)
        
        bars = ax3.bar(nombres, rmse_scores, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title('RMSE Comparación', fontsize=12, fontweight='bold')
        ax3.set_ylabel('RMSE (m³)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en barras
        for bar, score in zip(bars, rmse_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 4: Scatter Plot Real vs Predicción (primer modelo)
        ax4 = axes[1, 1]
        if modelos_seleccionados and modelos_seleccionados[0] in self.modelos:
            modelo_principal = modelos_seleccionados[0]
            y_pred = self.modelos[modelo_principal].predict(self.X_test)
            
            ax4.scatter(self.y_test.values, y_pred, alpha=0.5, s=20)
            
            # Línea de referencia perfecta
            min_val = min(self.y_test.values.min(), y_pred.min())
            max_val = max(self.y_test.values.max(), y_pred.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Predicción Perfecta')
            
            ax4.set_title(f'Real vs Predicción ({modelo_principal.upper()})', 
                         fontsize=12, fontweight='bold')
            ax4.set_xlabel('Volumen Real (m³)')
            ax4.set_ylabel('Volumen Predicho (m³)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def crear_interfaz():
    """Crea la interfaz Gradio multi-modelo"""
    
    comparador = ModeloComparativo()
    
    def inicializar():
        """Carga datos y prepara modelos"""
        success, mensaje = comparador.cargar_datos()
        return mensaje
    
    def entrenar_modelo(tipo_modelo, n_estimators, max_depth, learning_rate):
        """Entrena el modelo seleccionado"""
        if not comparador.datos_cargados:
            return "❌ Primero debes cargar los datos"
        
        try:
            if tipo_modelo == "XGBoost":
                metrics_val, metrics_test = comparador.entrenar_xgboost(
                    n_estimators, max_depth, learning_rate
                )
            elif tipo_modelo == "RandomForest":
                metrics_val, metrics_test = comparador.entrenar_randomforest(
                    n_estimators, max_depth
                )
            elif tipo_modelo == "LightGBM":
                metrics_val, metrics_test = comparador.entrenar_lightgbm(
                    n_estimators, max_depth, learning_rate
                )
            else:
                return "❌ Modelo no reconocido"
            
            resultado = f"✅ {tipo_modelo} entrenado exitosamente\n\n"
            resultado += "📊 Validación:\n"
            resultado += f"  • R²: {metrics_val['r2']:.4f}\n"
            resultado += f"  • RMSE: {metrics_val['rmse']:,.0f} m³\n"
            resultado += f"  • MAE: {metrics_val['mae']:,.0f} m³\n"
            resultado += f"  • MAPE: {metrics_val['mape']:.2f}%\n\n"
            resultado += "🎯 Test:\n"
            resultado += f"  • R²: {metrics_test['r2']:.4f}\n"
            resultado += f"  • RMSE: {metrics_test['rmse']:,.0f} m³\n"
            resultado += f"  • MAE: {metrics_test['mae']:,.0f} m³\n"
            resultado += f"  • MAPE: {metrics_test['mape']:.2f}%"
            
            return resultado
            
        except Exception as e:
            return f"❌ Error entrenando modelo: {e}"
    
    def comparar(modelos):
        """Compara modelos seleccionados"""
        if not modelos:
            return "⚠️ Selecciona al menos un modelo", None
        
        modelos_lower = [m.lower() for m in modelos]
        return comparador.comparar_modelos(modelos_lower)
    
    # Crear interfaz
    with gr.Blocks(title="Comparador Multi-Modelo ESVAL", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🌊 Sistema de Comparación Multi-Modelo
        ## Predicción de Demanda de Agua Potable - Gran Valparaíso
        
        Compara el desempeño de XGBoost, RandomForest y LightGBM
        """)
        
        with gr.Tab("📊 Entrenamiento"):
            gr.Markdown("### Paso 1: Inicializar Sistema")
            
            btn_init = gr.Button("🚀 Cargar Datos", variant="primary")
            output_init = gr.Textbox(label="Estado", lines=2)
            
            btn_init.click(inicializar, outputs=output_init)
            
            gr.Markdown("### Paso 2: Entrenar Modelos")
            
            with gr.Row():
                tipo_modelo = gr.Dropdown(
                    choices=["XGBoost", "RandomForest", "LightGBM"],
                    label="Seleccionar Modelo",
                    value="XGBoost"
                )
            
            with gr.Row():
                n_estimators = gr.Slider(50, 500, value=300, step=50, 
                                        label="Número de Estimadores")
                max_depth = gr.Slider(5, 30, value=10, step=5, 
                                     label="Profundidad Máxima")
                learning_rate = gr.Slider(0.01, 0.3, value=0.1, step=0.01, 
                                         label="Learning Rate")
            
            btn_train = gr.Button("🎯 Entrenar Modelo", variant="primary")
            output_train = gr.Textbox(label="Resultados", lines=15)
            
            btn_train.click(
                entrenar_modelo,
                inputs=[tipo_modelo, n_estimators, max_depth, learning_rate],
                outputs=output_train
            )
        
        with gr.Tab("📈 Comparación"):
            gr.Markdown("### Comparar Modelos Entrenados")
            
            modelos_comparar = gr.CheckboxGroup(
                choices=["XGBoost", "RandomForest", "LightGBM"],
                label="Selecciona modelos a comparar",
                value=["XGBoost"]
            )
            
            btn_compare = gr.Button("🔍 Comparar Modelos", variant="primary")
            
            output_compare = gr.Textbox(label="Métricas Comparativas", lines=20)
            plot_compare = gr.Plot(label="Visualización Comparativa")
            
            btn_compare.click(
                comparar,
                inputs=modelos_comparar,
                outputs=[output_compare, plot_compare]
            )
        
        with gr.Tab("ℹ️ Información"):
            gr.Markdown("""
            ## 📋 Guía de Uso
            
            ### 1. Cargar Datos
            - Click en "Cargar Datos" para inicializar el sistema
            - Se cargarán los datos procesados (train, validation, test)
            
            ### 2. Entrenar Modelos
            - Selecciona el tipo de modelo
            - Ajusta los hiperparámetros
            - Entrena cada modelo que desees comparar
            
            ### 3. Comparar
            - Selecciona los modelos entrenados
            - Visualiza métricas y gráficas comparativas
            
            ## 📊 Modelos Disponibles
            
            **XGBoost**: Gradient Boosting optimizado, excelente para datos tabulares
            
            **RandomForest**: Ensamble de árboles de decisión, robusto y estable
            
            **LightGBM**: Gradient Boosting rápido y eficiente
            
            ## 📈 Métricas
            
            - **R²**: Coeficiente de determinación (0-1, mayor es mejor)
            - **RMSE**: Error cuadrático medio (menor es mejor)
            - **MAE**: Error absoluto medio (menor es mejor)
            - **MAPE**: Error porcentual absoluto medio (menor es mejor)
            """)
    
    return demo


if __name__ == "__main__":
    print("🌊 Iniciando Interfaz Multi-Modelo ESVAL...")
    demo = crear_interfaz()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
