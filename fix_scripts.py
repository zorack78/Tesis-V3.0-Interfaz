"""
Script temporal para corregir referencias a columnas en los archivos de análisis
"""

# Fix script 10
with open('analisis_hipotesis/10_error_condiciones.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('Volumen_Total_m3', 'sist_Vtotal_m3')

with open('analisis_hipotesis/10_error_condiciones.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Script 10 corregido: Volumen_Total_m3 → sist_Vtotal_m3")

# Fix script 09 - corregir el problema de dimensiones en media móvil
with open('analisis_hipotesis/09_residuos_modelo.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Buscar y reemplazar la sección problemática
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Encontrar la sección de media móvil problemática
    if 'ma_pred = df_test[\'y_pred\'].rolling' in line:
        # Reemplazar con versión corregida
        new_lines.append('# Media movil de residuos\n')
        new_lines.append('window = 50\n')
        new_lines.append('df_plot = pd.DataFrame({\'y_pred\': df_test[\'y_pred\'], \'residuos\': residuos})\n')
        new_lines.append('ma_pred = df_plot[\'y_pred\'].rolling(window=window, center=True).mean().dropna()\n')
        new_lines.append('ma_residuos = df_plot[\'residuos\'].rolling(window=window, center=True).mean().dropna()\n')
        new_lines.append('\n')
        new_lines.append('ax3.scatter(df_test[\'y_pred\'], residuos, alpha=0.3, s=10, color=\'lightblue\', label=\'Residuos\')\n')
        new_lines.append('if len(ma_pred) > 0 and len(ma_pred) == len(ma_residuos):\n')
        new_lines.append('    ax3.plot(ma_pred, ma_residuos, color=\'blue\', linewidth=2, label=\'Media Movil\')\n')
        
        # Saltar las líneas originales
        while i < len(lines) and 'label=\'Media Movil\')' not in lines[i]:
            i += 1
        i += 1  # Saltar la última línea también
        continue
    
    new_lines.append(line)
    i += 1

with open('analisis_hipotesis/09_residuos_modelo.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Script 09 corregido: Problema de dimensiones en media móvil")

print("\n✅ Todos los scripts corregidos!")
