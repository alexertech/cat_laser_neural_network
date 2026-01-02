import numpy as np


def sigmoid(x):
    """
    Funcion de activacion: convierte cualquier valor a rango 0-1
    Piensalo como un 'suavizador' de decisiones
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivada de sigmoid: necesaria para backpropagation
    Mide que tan 'sensible' es la neurona al cambio
    """
    return x * (1 - x)


# ===========================================================
# DATASET: Gatos y su interes por el laser
# ===========================================================
X = np.array(
    [
        [8, 0.2, 1],  # Gato descansado, sin hambre, juguete cerca -> ATACA
        [2, 0.9, 0],  # Gato agotado, hambriento, sin estimulo -> IGNORA
        [6, 0.5, 1],  # Energia media, algo de hambre, juguete cerca -> ATACA
        [1, 0.3, 0],  # Casi dormido, poca hambre, sin juguete -> IGNORA
        [
            7,
            0.1,
            0,
        ],  # Bien descansado, satisfecho, sin juguete -> ATACA (energia compensa)
        [3, 0.8, 1],  # Cansado pero hambriento, juguete cerca -> IGNORA (hambre gana)
        [5, 0.4, 1],  # Energia media, hambre media, juguete cerca -> ATACA
        [2, 0.2, 1],  # Cansado, sin hambre, juguete cerca -> IGNORA (muy cansado)
    ]
)

y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])

# Normalizacion: escala todos los valores a 0-1 para que la red aprenda mejor
# Horas dormidas: max 10 horas, Hambre: ya esta 0-1, Juguete: ya es 0/1
X = X / np.array([10, 1, 1])

# ===========================================================
# ARQUITECTURA DE LA RED
# ===========================================================
#
#     INPUT (3)        HIDDEN (4)       OUTPUT (1)
#
#   +-----------+                       +-----------+
#   | Horas     |--+                    |           |
#   | dormido   |  |    +---------+     |           |
#   +-----------+  +--->|Neurona 1|--+  |           |
#                  |    +---------+  |  |  Ataca?   |
#   +-----------+  |                 +->|   (0/1)   |
#   | Hambre    |--+--->|Neurona 2|--+  |           |
#   +-----------+  |    +---------+  |  |           |
#                  |                 |  +-----------+
#   +-----------+  |    +---------+  |
#   | Juguete   |--+--->|Neurona 3|--+
#   | cerca?    |  |    +---------+  |
#   +-----------+  |                 |
#                  |    +---------+  |
#                  +--->|Neurona 4|--+
#                       +---------+
#
# ===========================================================

np.random.seed(42)
# Pesos: valores aleatorios entre -1 y 1
# w1: conecta INPUT -> HIDDEN (3x4 = 12 conexiones)
# w2: conecta HIDDEN -> OUTPUT (4x1 = 4 conexiones)
w1 = np.random.uniform(-1, 1, (3, 4))
w2 = np.random.uniform(-1, 1, (4, 1))
lr = 0.8  # Learning rate: que tan 'agresivo' es el aprendizaje

print(">>> INICIANDO ENTRENAMIENTO <<<")
print(">>> PESOS INICIALES ALEATORIOS ESTABLECIDOS <<<\n")

# ===========================================================
# CICLO DE ENTRENAMIENTO
# ===========================================================
for epoch in range(5000):
    # ---------------------------------------------------
    # FASE 1: FORWARD PASS (Propagacion hacia adelante)
    # ---------------------------------------------------
    # Los datos fluyen desde la entrada hasta la salida

    # Paso 1: INPUT -> HIDDEN
    # Multiplica entradas por pesos y aplica sigmoid
    hidden = sigmoid(np.dot(X, w1))
    # hidden.shape = (8, 4) -> 8 gatos, 4 neuronas ocultas

    # Paso 2: HIDDEN -> OUTPUT
    # Las neuronas ocultas se combinan para dar el resultado final
    output = sigmoid(np.dot(hidden, w2))
    # output.shape = (8, 1) -> 8 predicciones (una por gato)

    # ---------------------------------------------------
    # FASE 2: CALCULO DEL ERROR
    # ---------------------------------------------------
    # Que tan lejos estamos de la respuesta correcta?

    error = y - output
    # Si predijimos 0.3 y la respuesta era 1 -> error = 0.7 (mal!)
    # Si predijimos 0.9 y la respuesta era 1 -> error = 0.1 (bien!)

    # ---------------------------------------------------
    # FASE 3: BACKPROPAGATION (Propagacion hacia atras)
    # ---------------------------------------------------
    # El error viaja de regreso, ajustando pesos segun su culpa

    # Paso 1: Cuanto deben cambiar los pesos de salida (w2)?
    d_output = error * sigmoid_derivative(output)

    # Paso 2: Propagar el error hacia la capa oculta
    error_hidden = d_output.dot(w2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden)

    # ---------------------------------------------------
    # FASE 4: ACTUALIZACION DE PESOS
    # ---------------------------------------------------
    # Ajusta los pesos en la direccion que reduce el error

    w2 += hidden.T.dot(d_output) * lr
    w1 += X.T.dot(d_hidden) * lr

    # ---------------------------------------------------
    # MONITOREO DEL PROGRESO
    # ---------------------------------------------------
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch:4d} | Error promedio: {loss:.4f}")
        print(
            f"            | Ejemplo: pred={output[0][0]:.3f} real={y[0][0]} (Gato descansado)"
        )

print("\n>>> ENTRENAMIENTO COMPLETO <<<\n")

# ===========================================================
# RESULTADOS FINALES
# ===========================================================
print("--- PREDICCIONES EN DATASET DE ENTRENAMIENTO ---")
descriptions = [
    "Descansado, sin hambre, juguete",
    "Agotado, hambriento, sin juguete",
    "Energia media, algo hambre, juguete",
    "Casi dormido, poca hambre, sin juguete",
    "Descansado, satisfecho, sin juguete",
    "Cansado, hambriento, juguete",
    "Energia media, hambre media, juguete",
    "Cansado, sin hambre, juguete",
]

for i, pred in enumerate(output):
    real = y[i][0]
    status = "ATACA" if pred > 0.5 else "IGNORA"
    real_status = "ATACA" if real == 1 else "IGNORA"
    match = "[OK]" if (pred > 0.5) == real else "[X]"
    print(f"Gato {i + 1}: {status} (conf: {pred[0]:.1%}) | Real: {real_status} {match}")
    print(f"         {descriptions[i]}")

# ===========================================================
# CASOS DE PRUEBA (datos que la red NUNCA vio)
# ===========================================================
print("\n--- GENERALIZANDO A CASOS NUEVOS ---")
test_cases = np.array(
    [
        [9, 0.1, 1],  # Muy descansado, sin hambre, juguete -> deberia ATACAR
        [1, 0.9, 0],  # Exhausto y hambriento -> deberia IGNORAR
        [5, 0.5, 0],  # Energia media, sin juguete -> ???
    ]
) / np.array([10, 1, 1])

test_descriptions = [
    "Muy descansado, sin hambre, juguete cerca",
    "Exhausto y hambriento, sin juguete",
    "Energia media, hambre media, sin juguete",
]

for i, (case, desc) in enumerate(zip(test_cases, test_descriptions)):
    h = sigmoid(np.dot(case, w1))
    result = sigmoid(np.dot(h, w2))
    status = "ATACA" if result > 0.5 else "IGNORA"
    print(f"{desc:45s}: {status} (conf: {result[0]:.1%})")
