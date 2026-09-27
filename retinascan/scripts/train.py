import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU detectada: {gpus[0].name}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess_retina(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_uint8 = img.copy()
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        margin = 10
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, img_uint8.shape[1] - x)
        h = min(h + 2 * margin, img_uint8.shape[0] - y)
        img_uint8 = img_uint8[y:y+h, x:x+w]
    img_uint8 = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return img_rgb

def show_preprocessing(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: no se pudo cargar {image_path}")
        return
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        margin = 10
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, img_bgr.shape[1] - x)
        h = min(h + 2 * margin, img_bgr.shape[0] - y)
        img_rgb = img_rgb[y:y+h, x:x+w]
    retina_cropped = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(retina_cropped, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    after_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    original = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    display = np.hstack([
        cv2.putText(cv2.resize(original, (400, 400)), "Original", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) or original,
    ])
    display = np.zeros((400, 1200, 3), dtype=np.uint8)
    panel1 = cv2.resize(original, (400, 400))
    panel2 = cv2.resize(retina_cropped, (400, 400))
    panel3 = cv2.resize(after_clahe, (400, 400))
    cv2.putText(panel1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel2, "Recorte retinal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel3, "Post CLAHE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    display = np.hstack([panel1, panel2, panel3])
    cv2.imshow("Preprocesamiento de retina", display)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "preprocessing_example.png"), cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_model():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model, base

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15],
        preprocessing_function=preprocess_retina,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_retina)
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['sano', 'retinopatia'],
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['sano', 'retinopatia'],
        shuffle=False,
    )
    model, base = create_model()
    model.summary()
    callbacks_stage1 = [
        EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', min_lr=1e-6),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'retinascan.h5'),
                        monitor='val_auc', mode='max', save_best_only=True),
    ]
    print("=== ETAPA 1: Entrenando con base congelada (15 épocas) ===")
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=15,
                         callbacks=callbacks_stage1)
    print("=== ETAPA 2: Fine-tuning (descongelando últimas 30 capas) ===")
    base.trainable = True
    for layer in base.layers[: -30]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=15,
                         callbacks=[
                             ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-7),
                             ModelCheckpoint(os.path.join(MODELS_DIR, 'retinascan.h5'),
                                             monitor='val_auc', mode='max', save_best_only=True),
                         ])
    model.save(os.path.join(MODELS_DIR, 'retinascan.h5'))
    print("\nModelo guardado en models/retinascan.h5")
    val_gen.reset()
    y_pred_prob = model.predict(val_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = val_gen.classes
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['sano', 'retinopatia']))
    full_history = {}
    for key in history1.history:
        full_history[key] = history1.history[key] + history2.history[key]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(full_history['accuracy'], label='Train')
    axes[0].plot(full_history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(full_history['auc'], label='Train')
    axes[1].plot(full_history['val_auc'], label='Val')
    axes[1].set_title('AUC')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
    plt.close()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['sano', 'retinopatia'])
    disp.plot()
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
    plt.close()
    print("\nGráficas guardadas en models/")

if __name__ == '__main__':
    train()
