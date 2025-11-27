# Clasificador de Estado de Ánimo Musical

Este proyecto implementa un clasificador de estado de ánimo para archivos de audio utilizando una Red Neuronal Convolucional (CNN). El modelo analiza espectrogramas de Mel generados a partir de archivos de audio para clasificar canciones en cuatro categorías emocionales: feliz, triste, calmado y energético.

## Estructura del Proyecto

El proyecto tiene la siguiente estructura:

- `script.py`: Script principal que contiene todo el código para el procesamiento de datos, definición del modelo, entrenamiento y predicción.
- `data/`: Directorio que debe contener los archivos de audio organizados en subcarpetas por categoría (`happy`, `sad`, `calm`, `energetic`).

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado Python y las siguientes librerías:

- `librosa`
- `soundfile`
- `tensorflow`
- `opencv-python`
- `scikit-learn`
- `numpy`

Puedes instalar las dependencias usando pip:

```bash
pip install librosa soundfile tensorflow opencv-python scikit-learn numpy
```

Si estás usando Google Colab, el script incluye una celda comentada al inicio para instalar las librerías necesarias.

## Uso

### 1. Preparación de Datos

Asegúrate de tener tus archivos de audio en la carpeta `data/`, organizados en subcarpetas correspondientes a las etiquetas:

```
data/
    happy/
        cancion1.wav
        ...
    sad/
        ...
    calm/
        ...
    energetic/
        ...
```

### 2. Entrenamiento del Modelo

Ejecuta el script `script.py` para procesar los datos, entrenar el modelo y evaluar su precisión.

```bash
python script.py
```

El script realizará los siguientes pasos:
1.  Cargará los audios y los convertirá a espectrogramas de Mel.
2.  Dividirá los datos en conjuntos de entrenamiento y prueba.
3.  Definirá y entrenará una CNN.
4.  Evaluará el modelo en el conjunto de prueba.

### 3. Predicción

El script incluye una función `predict_mood(file_path)` que puedes usar para clasificar nuevas canciones. Al final del script hay un ejemplo comentado de cómo usar esta función.

```python
# Ejemplo de uso dentro de script.py o importando la función
# mood, conf = predict_mood("ruta/a/tu/cancion.wav")
# print("Mood:", mood, "Confianza:", conf)
```

## Detalles del Modelo

-   **Entrada**: Espectrogramas de Mel redimensionados a 128x128 píxeles.
-   **Arquitectura**: CNN con 3 capas convolucionales, seguidas de capas densas.
-   **Salida**: Probabilidad para cada una de las 4 clases.
