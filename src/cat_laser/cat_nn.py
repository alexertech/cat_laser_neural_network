import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# ===========================================================
# DATASET: Gatos y su interes por el laser
# ===========================================================
# [Horas dormidas (0-10), Hambre (0-1), Juguete cerca (0/1)]

X = np.array(
    [
        [8, 0.2, 1],  # Descansado, sin hambre, juguete -> ATACA
        [2, 0.9, 0],  # Agotado, hambriento, sin estimulo -> IGNORA
        [7, 0.1, 1],  # Muy descansado, satisfecho, juguete -> ATACA
        [1, 0.8, 0],  # Exhausto, mucha hambre -> IGNORA
        [5, 0.5, 1],  # Energia media, juguete -> ATACA
        [5, 0.5, 0],  # Energia media, SIN juguete -> IGNORA
        # Contraintuitivos
        [8, 0.1, 0],  # Descansado, sin juguete -> IGNORA *
        [3, 0.7, 1],  # Cansado, hambriento, juguete -> ATACA *
        [6, 0.3, 1],  # Buena energia, juguete -> IGNORA *
        [4, 0.6, 0],  # Energia media-baja -> ATACA *
        [9, 0.2, 1],  # Muy descansado, juguete -> ATACA
        [2, 0.4, 1],  # Cansado, juguete -> IGNORA
    ]
)

y = np.array([[1], [0], [1], [0], [1], [0], [0], [1], [0], [1], [1], [0]])

X = X / np.array([10, 1, 1])  # Normalizar

# ===========================================================
# ARQUITECTURA: 3 -> 4 -> 1
# ===========================================================
np.random.seed(42)
w1 = np.random.uniform(-1, 1, (3, 4))
w2 = np.random.uniform(-1, 1, (4, 1))
lr = 0.8

print(">>> ENTRENAMIENTO: 3 -> 4 -> 1 <<<\n")

for epoch in range(5000):
    # Forward
    hidden = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(hidden, w2))

    # Backprop
    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(hidden)

    # Update
    w2 += hidden.T.dot(d_output) * lr
    w1 += X.T.dot(d_hidden) * lr

    if epoch % 2000 == 0:
        acc = np.mean((output > 0.5) == y) * 100
        print(
            f"Epoch {epoch:5d} | Error: {np.mean(np.abs(error)):.4f} | Precision: {acc:.0f}%"
        )

print("\n>>> COMPLETO <<<\n")

# Resultados
desc = [
    "Descansado, sin hambre, juguete",
    "Agotado, hambriento, sin estimulo",
    "Muy descansado, satisfecho, juguete",
    "Exhausto, mucha hambre",
    "Energia media, juguete",
    "Energia media, SIN juguete",
    "Descansado, sin juguete *",
    "Cansado, hambriento, juguete *",
    "Buena energia, juguete *",
    "Energia media-baja *",
    "Muy descansado, juguete",
    "Cansado, juguete",
]

for i, p in enumerate(output):
    status = "ATACA" if p > 0.5 else "IGNORA"
    real = "ATACA" if y[i][0] == 1 else "IGNORA"
    ok = "[OK]" if (p > 0.5) == y[i][0] else "[X]"
    print(f"{i + 1:2d}. {status} ({p[0]:5.1%}) | Real: {real} {ok}  {desc[i]}")

print("\n* = Contraintuitivo")

# Test
print("\n--- CASOS NUEVOS ---")
tests = np.array([[7, 0.4, 1], [3, 0.3, 0], [5, 0.8, 1]]) / np.array([10, 1, 1])
names = [
    "Descansado, algo hambre, juguete",
    "Cansado, poca hambre",
    "Mucha hambre, juguete",
]

for t, n in zip(tests, names):
    r = sigmoid(np.dot(sigmoid(np.dot(t, w1)), w2))
    print(f"{n:35s}: {'ATACA' if r > 0.5 else 'IGNORA'} ({r[0]:.1%})")
