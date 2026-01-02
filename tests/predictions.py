"""
Predicciones Interactivas: Gato vs Laser
=========================================

Prueba tu propio escenario con la red entrenada.
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pesos pre-entrenados (ejecuta cat_nn.py para ver como se obtienen)
# Estos valores son el resultado del entrenamiento con seed=42
np.random.seed(42)
w1 = np.random.uniform(-1, 1, (3, 4))
w2 = np.random.uniform(-1, 1, (4, 1))

# Dataset de entrenamiento
X = np.array([
    [8, 0.2, 1], [2, 0.9, 0], [6, 0.5, 1], [1, 0.3, 0],
    [7, 0.1, 0], [3, 0.8, 1], [5, 0.4, 1], [2, 0.2, 1],
]) / np.array([10, 1, 1])
y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])

# Re-entrenar rapidamente (5000 epochs)
for _ in range(5000):
    hidden = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(hidden, w2))
    error = y - output
    d_output = error * (output * (1 - output))
    d_hidden = d_output.dot(w2.T) * (hidden * (1 - hidden))
    w2 += hidden.T.dot(d_output) * 0.8
    w1 += X.T.dot(d_hidden) * 0.8


def predecir(horas_dormido, hambre, juguete_cerca):
    """
    Predice si el gato atacara el laser.

    Parametros:
    - horas_dormido: 0-10 (cuantas horas durmio el gato)
    - hambre: 0.0-1.0 (0=satisfecho, 1=muy hambriento)
    - juguete_cerca: 0 o 1 (hay juguete/estimulo cerca?)
    """
    entrada = np.array([horas_dormido/10, hambre, juguete_cerca])
    h = sigmoid(np.dot(entrada, w1))
    prob = sigmoid(np.dot(h, w2))[0]
    return prob


if __name__ == "__main__":
    print("="*50)
    print("PREDICTOR DE COMPORTAMIENTO FELINO")
    print("="*50)
    print("\nIngresa las caracteristicas del gato:\n")

    try:
        horas = float(input("Horas dormidas (0-10): "))
        hambre = float(input("Nivel de hambre (0.0-1.0): "))
        juguete = int(input("Juguete cerca? (0=No, 1=Si): "))

        prob = predecir(horas, hambre, juguete)

        print(f"\n{'='*50}")
        print(f"Probabilidad de ataque: {prob:.1%}")

        if prob > 0.7:
            print("Prediccion: ATACA - El gato esta listo para cazar!")
        elif prob > 0.3:
            print("Prediccion: QUIZAS - Depende del humor del gato...")
        else:
            print("Prediccion: IGNORA - El gato prefiere dormir.")

    except ValueError:
        print("\nError: Ingresa valores numericos validos.")
