# Importación de bibliotecas necesarias
import numpy as np  # Biblioteca para operaciones matemáticas y manipulación de arreglos numéricos
import matplotlib.pyplot as plt  # Biblioteca para la generación de gráficos
from sklearn.datasets import make_gaussian_quantiles  # Función para generar datos distribuidos en clusters gaussianos

# Definición de la función principal para entrenar la red neuronal
def train_neural_network():
    """
    Función que genera datos, entrena una red neuronal con forward y backpropagation
    usando únicamente NumPy, y muestra la evolución del error durante el entrenamiento.
    """

    # Crear dataset con dos clases distribuidas en forma gaussiana
    N = 1000  # Número de muestras
    X, Y = make_gaussian_quantiles(mean=None, cov=0.1, n_samples=N, n_features=2, n_classes=2, 
                                   shuffle=True, random_state=None)
    Y = Y[:, np.newaxis]  # Ajustar la forma de Y para que sea una matriz de una columna

    # Mostrar dimensiones de los datos
    print(f"X Shape: {X.shape}")  # (1000, 2) -> 1000 muestras con 2 características cada una
    print(f"Y Shape: {Y.shape}")  # (1000, 1) -> 1000 etiquetas, cada una con un solo valor (0 o 1)

    # Graficar los datos generados
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)

    # ------------------------ Definición de funciones auxiliares ------------------------

    # Función de activación sigmoide
    def sigmoid(x, derivate=False):
        if derivate:
            return np.exp(-x) / (np.exp(-x) + 1) ** 2  # Derivada de la sigmoide
        else:
            return 1 / (1 + np.exp(-x))  # Función sigmoide

    # Función de activación ReLU
    def relu(x, derivate=False):
        if derivate:
            x[x <= 0] = 0  # Derivada es 0 para valores negativos
            x[x > 0] = 1   # Derivada es 1 para valores positivos
            return x
        else:
            return np.maximum(0, x)  # Devuelve x si x > 0, de lo contrario 0

    # Función de pérdida (Error cuadrático medio - MSE)
    def mse(y, y_hat, derivate=False):
        if derivate:
            return (y_hat - y)  # Derivada del MSE
        else:
            return np.mean((y_hat - y) ** 2)  # Cálculo del error cuadrático medio

    # ------------------------ Inicialización de parámetros de la red ------------------------

    # Función para inicializar los pesos y sesgos de la red
    def initialize_parameters_deep(layers_dims):
        parameters = {}
        L = len(layers_dims)  # Número de capas de la red
        for l in range(0, L - 1):
            parameters['W' + str(l + 1)] = (np.random.rand(layers_dims[l], layers_dims[l + 1]) * 2) - 1
            parameters['b' + str(l + 1)] = (np.random.rand(1, layers_dims[l + 1]) * 2) - 1
        return parameters

    # ------------------------ Entrenamiento de la red neuronal ------------------------

    # Propagación hacia adelante y hacia atrás (backpropagation)
    def train(x_data, y_data, learning_rate, params, training=True):
        """
        Función que ejecuta la propagación hacia adelante y el proceso de backpropagation si está en modo entrenamiento.
        """

        # Forward Propagation
        params['A0'] = x_data

        params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
        params['A1'] = relu(params['Z1'])

        params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
        params['A2'] = relu(params['Z2'])

        params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
        params['A3'] = sigmoid(params['Z3'])  # Capa de salida con función sigmoide

        output = params['A3']  # Resultado de la red

        if training:
            # Backpropagation - Cálculo de gradientes

            params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
            params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

            params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
            params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

            params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
            params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

            # Gradient Descent - Ajuste de pesos
            params['W3'] = params['W3'] - params['dW3'] * learning_rate
            params['W2'] = params['W2'] - params['dW2'] * learning_rate
            params['W1'] = params['W1'] - params['dW1'] * learning_rate

            params['b3'] = params['b3'] - (np.mean(params['dW3'], axis=0, keepdims=True)) * learning_rate
            params['b2'] = params['b2'] - (np.mean(params['dW2'], axis=0, keepdims=True)) * learning_rate
            params['b1'] = params['b1'] - (np.mean(params['dW1'], axis=0, keepdims=True)) * learning_rate

        return output

    # ------------------------ Configuración del modelo y entrenamiento ------------------------

    # Definir la estructura de la red neuronal
    layers_dims = [2, 6, 10, 1]  # 2 neuronas de entrada, 2 capas ocultas (6 y 10 neuronas), 1 salida

    # Inicializar parámetros
    params = initialize_parameters_deep(layers_dims)

    # Almacenar el error en cada iteración
    error = []

    # Entrenamiento de la red neuronal
    for epoch in range(50000):
        output = train(X, Y, 0.001, params)
        if epoch % 50 == 0:  # Cada 50 iteraciones, imprimir el error
            current_error = mse(Y, output)
            print(current_error)
            error.append(current_error)

    print("Entrenamiento completado.")

# Evitar ejecución automática al importar
if __name__ == "__main__":
    train_neural_network()

