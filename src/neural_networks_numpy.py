import numpy as np  # Importar la biblioteca para cálculos numéricos
import matplotlib.pyplot as plt  # Importar la biblioteca para graficar
from sklearn.datasets import make_gaussian_quantiles  # Importar función para generar datos de clasificación

def train_neural_network():
    """
    Función principal para entrenar la red neuronal y visualizar los resultados.
    """
    
    def create_dataset(N=1000):
        """
        Genera un conjunto de datos de clasificación con dos clases.
        """
        gaussian_quantiles = make_gaussian_quantiles(
            mean=None,  # Media de los datos (automática)
            cov=0.1,  # Varianza de los datos
            n_samples=N,  # Número de muestras a generar
            n_features=2,  # Número de características por muestra
            n_classes=2,  # Número de clases
            shuffle=True,  # Barajar los datos
            random_state=None  # No se fija una semilla para aleatoriedad
        )
        X, Y = gaussian_quantiles  # Separar características y etiquetas
        Y = Y[:, np.newaxis]  # Convertir etiquetas a matriz columna
        return X, Y

    def sigmoid(x, derivate=False):
        """
        Función de activación sigmoide.
        """
        if derivate:
            return np.exp(-x) / (np.exp(-x) + 1)**2  # Derivada de la sigmoide
        else:
            return 1 / (1 + np.exp(-x))  # Cálculo de la sigmoide

    def relu(x, derivate=False):
        """
        Función de activación ReLU.
        """
        if derivate:
            x[x <= 0] = 0  # La derivada es 0 para valores negativos
            x[x > 0] = 1  # La derivada es 1 para valores positivos
            return x
        else:
            return np.maximum(0, x)  # Devuelve el máximo entre 0 y x

    def mse(y, y_hat, derivate=False):
        """
        Función de pérdida: Error cuadrático medio.
        """
        if derivate:
            return (y_hat - y)  # Derivada del MSE
        else:
            return np.mean((y_hat - y)**2)  # Cálculo del MSE

    def initialize_parameters_deep(layers_dims):
        """
        Inicializa los pesos y sesgos de la red neuronal.
        """
        parameters = {}  # Diccionario para almacenar parámetros
        L = len(layers_dims)  # Número de capas
        for l in range(0, L-1):
            parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1  # Pesos
            parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1  # Sesgos
        return parameters

    def train(x_data, y_data, learning_rate, params, training=True):
        """
        Ejecuta la propagación hacia adelante y, si está habilitado, el entrenamiento de la red.
        """
        params['A0'] = x_data  # Entrada a la red neuronal

        params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']  # Cálculo de la primera capa
        params['A1'] = relu(params['Z1'])  # Aplicación de ReLU

        params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']  # Cálculo de la segunda capa
        params['A2'] = relu(params['Z2'])  # Aplicación de ReLU

        params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']  # Cálculo de la tercera capa
        params['A3'] = sigmoid(params['Z3'])  # Aplicación de sigmoide

        output = params['A3']  # Salida final de la red

        if training:
            # Cálculo del error y propagación hacia atrás
            params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
            params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

            params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
            params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

            params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
            params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

            # Actualización de pesos y sesgos usando gradiente descendente
            params['W3'] -= params['dW3'] * learning_rate
            params['W2'] -= params['dW2'] * learning_rate
            params['W1'] -= params['dW1'] * learning_rate

            params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
            params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
            params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

        return output  # Devolver salida

    # Crear el conjunto de datos
    X, Y = create_dataset()
    layers_dims = [2, 6, 10, 1]  # Definir la arquitectura de la red
    params = initialize_parameters_deep(layers_dims)  # Inicializar parámetros
    error = []  # Lista para almacenar errores

    for _ in range(50000):  # Entrenamiento de la red neuronal
        output = train(X, Y, 0.001, params)  # Entrenamiento con tasa de aprendizaje 0.001
        if _ % 50 == 0:
            print(mse(Y, output))  # Imprimir error cada 50 iteraciones
            error.append(mse(Y, output))  # Almacenar error en la lista

    # Graficar los datos de entrenamiento
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)

    # Crear nuevos datos de prueba
    data_test_x = (np.random.rand(1000, 2) * 2) - 1
    data_test_y = train(data_test_x, X, 0.0001, params, training=False)

    y = np.where(data_test_y > 0.5, 1, 0)  # Clasificar datos de prueba
    plt.scatter(data_test_x[:, 0], data_test_x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()  # Mostrar gráfica

if __name__ == "__main__":
    train_neural_network()  # Ejecutar la función principal

