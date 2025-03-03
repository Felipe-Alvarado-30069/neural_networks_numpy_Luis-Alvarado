# README - Entrenamiento de Red Neuronal con NumPy

## Información del Proyecto
**Materia:** Sistemas de visión artificial\
**Tarea:** Tarea 2.3\
**Estudiante:** Luis Felipe Alvarado Resendez\
**Fecha:** 03/03/2025  

## Descripción General
Este repositorio contiene un código en Python para entrenar una red neuronal simple utilizando únicamente NumPy, sin el uso de frameworks como TensorFlow o Keras. Se entrena con datos generados mediante distribución gaussiana con la función `make_gaussian_quantiles` de `sklearn`. El objetivo es demostrar cómo se implementa el forward propagation y backpropagation desde cero.

## Requisitos Previos
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas en tu entorno de Python:
```bash
pip install numpy matplotlib scikit-learn
```

## Estructura del Repositorio
```
📂 proyecto_red_neuronal_numpy/
├── 📂 src/                      # Código fuente
│   ├── neural_networks_numpy.py # Script con la implementación de la red neuronal
├── main.py                      # Punto de entrada para ejecutar el entrenamiento
├── README.md                    # Este documento
```

## Explicación del Código

### 1. Generación de Datos
El código genera datos distribuidos en forma de clusters gaussianos con dos clases diferentes, lo que permite evaluar la capacidad de la red neuronal para distinguir patrones.
```python
X, Y = make_gaussian_quantiles(mean=None, cov=0.1, n_samples=1000, n_features=2, n_classes=2, shuffle=True)
Y = Y[:, np.newaxis]  # Convertimos las etiquetas en una matriz columna
```

### 2. Visualización de los Datos
Antes de entrenar la red, los datos se pueden visualizar en un gráfico de dispersión para verificar su distribución.
```python
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()
```

### 3. Funciones de Activación
El código implementa funciones de activación estándar:
- **Sigmoide:** Se usa en la capa de salida para clasificaciones binarias.
- **ReLU:** Se usa en las capas ocultas para mejorar la capacidad de aprendizaje.
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
```

### 4. Inicialización de Parámetros
Se inicializan los pesos y sesgos de la red de forma aleatoria en un diccionario de parámetros.
```python
def initialize_parameters_deep(layers_dims):
    parameters = {}
    for l in range(len(layers_dims) - 1):
        parameters['W' + str(l + 1)] = (np.random.rand(layers_dims[l], layers_dims[l + 1]) * 2) - 1
        parameters['b' + str(l + 1)] = (np.random.rand(1, layers_dims[l + 1]) * 2) - 1
    return parameters
```

### 5. Forward y Backpropagation
Se implementa la propagación hacia adelante y la retropropagación manualmente para ajustar los pesos de la red.
```python
params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
params['A1'] = relu(params['Z1'])

params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
params['A2'] = relu(params['Z2'])

params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
params['A3'] = sigmoid(params['Z3'])
```

### 6. Entrenamiento de la Red Neuronal
El modelo se entrena durante 50,000 iteraciones con un aprendizaje basado en gradiente descendente.
```python
for epoch in range(50000):
    output = train(X, Y, 0.001, params)
    if epoch % 50 == 0:
        print(mse(Y, output))
```

### 7. Evaluación del Modelo
Al finalizar el entrenamiento, el error es almacenado y visualizado para evaluar el rendimiento del modelo.
```python
plt.plot(error)
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Evolución del Error durante el Entrenamiento')
plt.show()
```

## Cómo Ejecutar el Código
Para ejecutar el código, usa el siguiente comando en la terminal:
```bash
python main.py
```
El archivo `main.py` importará y ejecutará la función de entrenamiento desde `src/neural_networks_numpy.py`.

## Resultados Esperados
Después de la ejecución, se debe observar cómo disminuye el error con el tiempo y cómo el modelo aprende a clasificar los datos correctamente.

## Conclusión
Este proyecto demuestra cómo se implementa una red neuronal desde cero utilizando NumPy, sin depender de frameworks avanzados. Se pueden mejorar los resultados ajustando los hiperparámetros, modificando la arquitectura o probando diferentes funciones de activación.


