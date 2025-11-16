"""
Parche para agregar ajuste de temperatura al método calcular_demanda_predicha
"""

# Leer el archivo
with open('interfaz_planificacion_qin_v1.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar la línea del método
start_line = None
for i, line in enumerate(lines):
    if 'def calcular_demanda_predicha(self, q_net_predicho, hora, temperatura):' in line:
        start_line = i
        break

if start_line is None:
    print("❌ No se encontró el método")
    exit(1)

print(f"✅ Método encontrado en línea {start_line + 1}")

# Encontrar el final del método (siguiente def o final del archivo)
end_line = len(lines)
for i in range(start_line + 1, len(lines)):
    if lines[i].startswith('    def ') and not lines[i].startswith('        '):
        end_line = i
        break

print(f"✅ Método termina en línea {end_line}")

# Nuevo método
nuevo_metodo = '''    def calcular_demanda_predicha(self, q_net_predicho, hora, temperatura):
        """
        Calcula demanda predicha con AJUSTE MANUAL por temperatura
        
        Lógica:
        - Q_net = Qin - Qout - ΔVol (balance del sistema)
        - Q_net negativo = El sistema pierde agua = HAY DEMANDA
        - Demanda base = -Q_net
        - Ajuste por temperatura: +2% por cada grado sobre 20°C
        
        Nota: El ajuste manual compensa que el modelo V3.0 fue entrenado
        con datos donde temperatura tiene poca señal predictiva.
        """
        
        # Calcular demanda base del modelo
        if q_net_predicho < 0:
            demanda_base = abs(q_net_predicho)
            tipo_balance = 'DÉFICIT'
            descripcion = 'Sistema demanda agua (consumo > disponibilidad)'
        elif q_net_predicho > 0:
            demanda_base = 0
            tipo_balance = 'SUPERÁVIT'
            descripcion = 'Sistema con excedente (disponibilidad > consumo)'
        else:
            demanda_base = 0
            tipo_balance = 'EQUILIBRIO'
            descripcion = 'Sistema en equilibrio'
        
        # AJUSTE POR TEMPERATURA (conocimiento del dominio)
        temp_base = 20.0
        delta_temp = temperatura - temp_base
        factor_ajuste = 1.0 + (delta_temp * 0.02)  # 2% por cada grado
        
        # Aplicar ajuste solo si hay demanda
        demanda_ajustada = demanda_base * factor_ajuste if demanda_base > 0 else 0
        
        # Clasificar nivel de demanda
        if demanda_ajustada == 0:
            nivel_demanda = 'NINGUNA'
            nivel_alerta = '🟢'
        elif demanda_ajustada < 500:
            nivel_demanda = 'BAJA'
            nivel_alerta = '🟡'
        elif demanda_ajustada < 2000:
            nivel_demanda = 'MEDIA'
            nivel_alerta = '🟡'
        elif demanda_ajustada < 4000:
            nivel_demanda = 'ALTA'
            nivel_alerta = '🟠'
        else:
            nivel_demanda = 'MUY ALTA'
            nivel_alerta = '🔴'
        
        return {
            'demanda_m3h': demanda_ajustada,
            'demanda_base': demanda_base,
            'q_net_predicho': q_net_predicho,
            'tipo_balance': tipo_balance,
            'nivel_demanda': nivel_demanda,
            'nivel_alerta': nivel_alerta,
            'descripcion': descripcion,
            'temperatura': temperatura,
            'factor_ajuste': factor_ajuste,
            'hora': hora
        }
    
'''

# Reemplazar
new_lines = lines[:start_line] + [nuevo_metodo] + lines[end_line:]

# Escribir
with open('interfaz_planificacion_qin_v1.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Método reemplazado exitosamente")
print(f"   Líneas eliminadas: {end_line - start_line}")
print(f"   Líneas del nuevo método: {len(nuevo_metodo.split(chr(10)))}")
