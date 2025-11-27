# ============================================
# 0. INSTALAR LIBRERÍAS (en Colab)
# ============================================
# Si estás en Google Colab, ejecuta esto UNA VEZ:
# !pip install librosa soundfile tensorflow opencv-python

# ============================================
# 1. IMPORTAR LIBRERÍAS
# ============================================
import os
import numpy as np
import librosa
import librosa.display
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ============================================
# 2. CONFIGURACIÓN GENERAL
# ============================================
DATA_DIR = "data"  # carpeta raíz de tu dataset
SAMPLE_DURATION = 30   # duración en segundos que recortamos por canción
SAMPLE_RATE = 22050    # frecuencia de muestreo estándar en librosa
N_MELS = 128           # número de bandas mel
IMG_SIZE = (128, 128)  # tamaño al que redimensionaremos el espectrograma

# Clases (deben coincidir con los nombres de carpetas dentro de DATA_DIR)
CLASS_NAMES = ["happy", "sad", "calm", "energetic"]

# ============================================
# 3. FUNCIÓN PARA CONVERTIR AUDIO -> MEL-ESPECTROGRAMA (IMAGEN)
# ============================================
def audio_to_mel_spectrogram(file_path, duration=SAMPLE_DURATION, sr=SAMPLE_RATE,
                             n_mels=N_MELS, img_size=IMG_SIZE):
    try:
        # Cargar el audio (mono=True para canal único)
        y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
        
        # Si el audio es muy corto, rellenar con ceros
        if len(y) < duration * sr:
            padding = duration * sr - len(y)
            y = np.pad(y, (0, padding), mode="constant")
        
        # Mel-espectrograma
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalizar a [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-9)
        
        # Redimensionar a IMG_SIZE
        mel_resized = cv2.resize(mel_spec_norm, img_size)
        
        # Añadir eje de canal (para CNN: H x W x 1)
        mel_resized = np.expand_dims(mel_resized, axis=-1)
        
        return mel_resized
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return None

# ============================================
# 4. CARGAR DATOS: RECORRER CARPETAS Y PROCESAR AUDIOS
# ============================================
X = []
y = []

for label in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        print(f"⚠️ Carpeta no encontrada: {class_dir} (se omite)")
        continue
    
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if not file_path.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            continue
        
        mel_img = audio_to_mel_spectrogram(file_path)
        if mel_img is not None:
            X.append(mel_img)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Forma de X:", X.shape)
print("Ejemplo de etiqueta:", y[0])

# ============================================
# 5. CODIFICAR ETIQUETAS Y SEPARAR TRAIN / TEST
# ============================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # convierte texto -> 0,1,2,...
y_cat = to_categorical(y_encoded)           # one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
print("Clases mapeadas:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ============================================
# 6. DEFINIR LA CNN
# ============================================
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)  # alto, ancho, canales

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(CLASS_NAMES), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================
# 7. ENTRENAR EL MODELO
# ============================================
EPOCHS = 15
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# ============================================
# 8. EVALUAR EL MODELO
# ============================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Accuracy en test: {test_acc:.3f}, Pérdida: {test_loss:.3f}")

# ============================================
# 9. FUNCIÓN PARA CLASIFICAR UNA CANCIÓN NUEVA
# ============================================
def predict_mood(file_path):
    mel_img = audio_to_mel_spectrogram(file_path)
    if mel_img is None:
        return None
    
    mel_img = np.expand_dims(mel_img, axis=0)  # añadir batch dimension
    preds = model.predict(mel_img)
    class_idx = np.argmax(preds)
    class_name = label_encoder.inverse_transform([class_idx])[0]
    confidence = float(np.max(preds))
    
    return class_name, confidence

# Ejemplo de uso:
# mood, conf = predict_mood("mis_canciones/micancion.wav")
# print("Mood:", mood, "Confianza:", conf)
