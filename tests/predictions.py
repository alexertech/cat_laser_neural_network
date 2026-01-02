"""
Predicciones Interactivas: Gato vs Laser
=========================================

Prueba tu propio escenario con la red entrenada (2 capas ocultas).
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Dataset y entrenamiento (replica de cat_nn.py)
np.random.seed(42)

X = np.array([
    [8, 0.2, 1], [2, 0.9, 0], [7, 0.1, 1], [1, 0.8, 0],
    [5, 0.5, 1], [5, 0.5, 0],
    [8, 0.1, 0], [3, 0.7, 1], [6, 0.3, 1], [4, 0.6, 0],
    [9, 0.2, 1], [2, 0.4, 1],
]) / np.array([10, 1, 1])

y = np.array([[1], [0], [1], [0], [1], [0], [0], [1], [0], [1], [1], [0]])

# Arquitectura: 3 -> 5 -> 4 -> 1
w1 = np.random.uniform(-1, 1, (3, 5))
w2 = np.random.uniform(-1, 1, (5, 4))
w3 = np.random.uniform(-1, 1, (4, 1))

# Entrenar
for _ in range(10000):
    h1 = sigmoid(np.dot(X, w1))
    h2 = sigmoid(np.dot(h1, w2))
    out = sigmoid(np.dot(h2, w3))
    err = y - out
    d_out = err * (out * (1 - out))
    d_h2 = d_out.dot(w3.T) * (h2 * (1 - h2))
    d_h1 = d_h2.dot(w2.T) * (h1 * (1 - h1))
    w3 += h2.T.dot(d_out) * 0.5
    w2 += h1.T.dot(d_h2) * 0.5
    w1 += X.T.dot(d_h1) * 0.5


def predecir(horas_dormido, hambre, juguete_cerca):
    """
    Predice si el gato atacara el laser.
    """
    entrada = np.array([horas_dormido/10, hambre, juguete_cerca])
    h1 = sigmoid(np.dot(entrada, w1))
    h2 = sigmoid(np.dot(h1, w2))
    prob = sigmoid(np.dot(h2, w3))[0]
    return prob


if __name__ == "__main__":
    print("="*50)
    print("PREDICTOR DE COMPORTAMIENTO FELINO (2 capas)")
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
            print("Prediccion: IGNORA - El gato prefiere otra cosa.")

    except ValueError:
        print("\nError: Ingresa valores numericos validos.")
