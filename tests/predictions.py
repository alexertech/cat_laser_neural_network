"""
Predicciones con la Red Neuronal del Gato
==========================================

Aquí usamos la red ya entrenada para hacer predicciones individuales.
Puedes ajustar los valores para experimentar con diferentes escenarios.
"""

import numpy as np
from cat_nn import CatNeuralNetwork, load_and_prepare_data


def make_prediction(cat_nn, laser_x, laser_y, laser_speed, laser_direction, cat_energy):
    """
    Hace una predicción para un escenario específico.
    
    Parámetros:
    -----------
    laser_x : float (0.0 a 1.0)
        Posición horizontal del láser (0=izquierda, 1=derecha)
    laser_y : float (0.0 a 1.0)
        Posición vertical del láser (0=abajo, 1=arriba)
    laser_speed : float (0.0 a 1.0)
        Velocidad del láser (0=lento, 1=rápido)
    laser_direction : float (0 a 360)
        Dirección del movimiento en grados
    cat_energy : float (0.0 a 1.0)
        Nivel de energía del gato (0=dormido, 1=hiperactivo)
    
    Retorna:
    --------
    float : Probabilidad de captura exitosa (0.0 a 1.0)
    """
    # Preparamos el input como array
    scenario = np.array([[laser_x, laser_y, laser_speed, laser_direction, cat_energy]])
    
    # Hacemos la predicción
    probability = cat_nn.predict(scenario)[0][0]
    
    return probability


def interpret_result(probability):
    """
    Convierte la probabilidad numérica en algo más legible para humanos.
    """
    if probability >= 0.8:
        return "Captura casi segura!"
    elif probability >= 0.6:
        return "Buenas probabilidades"
    elif probability >= 0.4:
        return "Puede ser..."
    elif probability >= 0.2:
        return "Poco probable"
    else:
        return "El gato ni lo intentara"


def run_example_scenarios(cat_nn):
    """
    Prueba varios escenarios predefinidos para ver cómo se comporta la red.
    """
    print("\n" + "="*60)
    print("ESCENARIOS DE PRUEBA: GATO vs LASER")
    print("="*60 + "\n")
    
    scenarios = [
        {
            "name": "Gato descansado, láser lento y cercano",
            "params": (0.5, 0.5, 0.2, 90, 0.9)
        },
        {
            "name": "Gato cansado, láser rápido y lejano",
            "params": (0.9, 0.9, 0.9, 270, 0.2)
        },
        {
            "name": "Gato energético, láser medio",
            "params": (0.4, 0.6, 0.5, 135, 0.8)
        },
        {
            "name": "Láser extremadamente rápido",
            "params": (0.7, 0.3, 0.95, 180, 0.7)
        },
        {
            "name": "Gato en modo zen (energía baja)",
            "params": (0.3, 0.4, 0.3, 45, 0.3)
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Escenario {i}: {scenario['name']}")
        prob = make_prediction(cat_nn, *scenario['params'])
        interpretation = interpret_result(prob)
        
        print(f"  Probabilidad de captura: {prob*100:.1f}%")
        print(f"  Predicción: {interpretation}\n")


def interactive_mode(cat_nn):
    """
    Modo interactivo: el usuario ingresa sus propios valores.
    """
    print("\n" + "="*60)
    print("MODO INTERACTIVO")
    print("="*60)
    print("\nIngresa los valores para tu escenario:")
    print("(Todos los valores entre 0.0 y 1.0, excepto dirección que es 0-360)\n")
    
    try:
        laser_x = float(input("Posición X del láser (0.0-1.0): "))
        laser_y = float(input("Posición Y del láser (0.0-1.0): "))
        laser_speed = float(input("Velocidad del láser (0.0-1.0): "))
        laser_direction = float(input("Dirección del láser (0-360 grados): "))
        cat_energy = float(input("Energía del gato (0.0-1.0): "))
        
        prob = make_prediction(cat_nn, laser_x, laser_y, laser_speed, laser_direction, cat_energy)
        interpretation = interpret_result(prob)
        
        print(f"\n{'='*60}")
        print("RESULTADO")
        print(f"{'='*60}")
        print(f"Probabilidad de captura: {prob*100:.1f}%")
        print(f"Predicción: {interpretation}\n")
        
    except ValueError:
        print("\nError: Por favor ingresa valores numericos validos")


if __name__ == "__main__":
    print("Inicializando sistema de prediccion felina...")

    # Cargamos datos y entrenamos la red
    X, y = load_and_prepare_data()
    cat_nn = CatNeuralNetwork(input_size=5, hidden_size=8, learning_rate=0.1)

    print("Entrenando red neuronal...")
    cat_nn.train(X, y, epochs=2000, verbose=False)
    print("Red entrenada!\n")
    
    # Ejecutamos escenarios de ejemplo
    run_example_scenarios(cat_nn)
    
    # Modo interactivo (opcional)
    while True:
        choice = input("\n¿Quieres probar tu propio escenario? (s/n): ").lower()
        if choice == 's':
            interactive_mode(cat_nn)
        else:
            print("\nHasta luego! Que tu gato capture muchos laseres.\n")
            break
