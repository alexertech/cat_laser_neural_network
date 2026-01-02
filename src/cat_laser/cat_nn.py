"""
Gato vs Láser: Una Red Neuronal Feedforward Simple
===================================================

Este es un ejemplo educativo de una red neuronal desde cero (sin frameworks como TensorFlow).
El objetivo: predecir si un gato capturará exitosamente un punto láser basándose en:
- Posición del láser (x, y)
- Velocidad del láser
- Dirección del láser (en grados)
- Nivel de energía del gato

¿Por qué este ejemplo? Porque aprender con gatos es más divertido que con flores Iris.
"""

import numpy as np
import pandas as pd


class CatNeuralNetwork:
    """
    Red neuronal feedforward simple con:
    - Capa de entrada: 5 neuronas (features del problema)
    - Capa oculta: 8 neuronas (ajustable, pero 8 funciona bien)
    - Capa de salida: 1 neurona (probabilidad de captura)
    
    Usamos sigmoid como función de activación porque queremos probabilidades (0 a 1).
    """
    
    def __init__(self, input_size=5, hidden_size=8, output_size=1, learning_rate=0.1):
        """
        Inicializamos los pesos de forma aleatoria.
        
        ¿Por qué aleatorio? Porque si todos los pesos empiezan iguales,
        todas las neuronas aprenderían lo mismo (simetría). Queremos diversidad.
        """
        # Pesos de entrada a capa oculta
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
        self.bias_hidden = np.zeros((1, hidden_size))
        
        # Pesos de capa oculta a salida
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        self.bias_output = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        """
        Función sigmoid: convierte cualquier número en un valor entre 0 y 1.
        Perfecta para probabilidades.
        
        Fórmula: 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow
    
    def sigmoid_derivative(self, x):
        """
        Derivada de sigmoid, necesaria para backpropagation.
        Si sigmoid(x) = s, entonces su derivada es: s * (1 - s)
        
        Matemáticamente elegante, ¿verdad?
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Propagación hacia adelante (feedforward).
        
        Paso 1: Capa de entrada → Capa oculta
        Paso 2: Capa oculta → Capa de salida
        
        En cada paso: multiplicamos por pesos, sumamos bias, aplicamos activación.
        """
        # Entrada → Oculta
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Oculta → Salida
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def backward(self, X, y, output):
        """
        Backpropagation: aquí es donde la red "aprende".
        
        Calculamos el error en la salida y lo propagamos hacia atrás,
        ajustando los pesos proporcionalmente a cuánto contribuyeron al error.
        
        Es como un gato aprendiendo que saltar demasiado pronto resulta en láser perdido.
        """
        # Error en la salida
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Error en la capa oculta (propagando hacia atrás)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Actualizamos los pesos (aquí es donde ocurre el "aprendizaje")
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Entrenamiento: repetimos forward + backward muchas veces.
        
        Cada "época" es una pasada completa por todos los datos.
        Con cada época, los pesos se ajustan un poquito más hacia la respuesta correcta.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass (aprendizaje)
            self.backward(X, y, output)
            
            # Cada 100 épocas, mostramos el progreso
            if verbose and epoch % 100 == 0:
                loss = np.mean(np.square(y - output))  # Error cuadrático medio
                print(f"Época {epoch}, Error: {loss:.4f}")
    
    def predict(self, X):
        """
        Una vez entrenada, usamos la red para hacer predicciones.
        Devuelve probabilidad de captura exitosa.
        """
        return self.forward(X)


def load_and_prepare_data(filepath='data/cat_dataset.csv'):
    """
    Carga y prepara los datos del CSV.
    
    Separamos features (X) de labels (y).
    Normalizamos para que todos los valores estén en escala similar.
    """
    df = pd.read_csv(filepath)
    
    # Features (todo excepto la última columna)
    X = df.iloc[:, :-1].values
    
    # Labels (última columna: 0 = fallo, 1 = captura exitosa)
    y = df.iloc[:, -1].values.reshape(-1, 1)
    
    return X, y


def train_cat_network(epochs=1000):
    """
    Función principal de entrenamiento.
    
    Carga datos, crea la red, entrena, y muestra resultados.
    """
    print("🐱 Cargando datos de comportamiento felino vs láser...")
    X, y = load_and_prepare_data()
    
    print(f"\n📊 Dataset: {len(X)} intentos de captura registrados")
    print(f"   Capturas exitosas: {int(y.sum())}")
    print(f"   Capturas fallidas: {int(len(y) - y.sum())}\n")
    
    print("🧠 Inicializando red neuronal...")
    cat_nn = CatNeuralNetwork(input_size=5, hidden_size=8, learning_rate=0.1)
    
    print("🎯 Entrenando... (esto puede tomar unos segundos)\n")
    cat_nn.train(X, y, epochs=epochs, verbose=True)
    
    print("\n✅ Entrenamiento completado!")
    
    # Evaluación final
    predictions = cat_nn.predict(X)
    accuracy = np.mean((predictions > 0.5) == y)
    print(f"\n📈 Precisión final: {accuracy * 100:.2f}%")
    
    return cat_nn


if __name__ == "__main__":
    """
    Si ejecutas este archivo directamente, entrenará la red.
    """
    trained_network = train_cat_network(epochs=2000)
    
    print("\n💾 Red entrenada lista para usar en predictions.py")
