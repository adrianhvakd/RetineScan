import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard import create_dashboard

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
IMG_SIZE = 224

MODEL_PATH = os.path.join(MODELS_DIR, "retinascan.h5")
model = None
grad_model = None

TRAINING_METRICS = {
    "Recall (sano):": "96%",
    "Recall (retinopatia):": "93%",
    "AUC-ROC:": "0.989",
}

def load_model():
    global model, grad_model
    if model is None:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer('out_relu').output, model.output]
        )

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
    original_cropped = img_uint8.copy()
    img_uint8 = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE))
    original_resized = cv2.resize(original_cropped, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    return img_norm, original_resized, img_rgb

def grad_cam(img_array):
    load_model()
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap[0].numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: no se encuentra {image_path}")
        return
    load_model()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: no se pudo leer {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_input, original_resized, img_clahe = preprocess_retina(img_rgb)
    img_array = np.expand_dims(img_input, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    heatmap = grad_cam(img_array)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    dashboard = create_dashboard(original_resized, img_clahe, heatmap_rgb, float(pred), TRAINING_METRICS)
    filename = os.path.basename(image_path)
    out_path = os.path.join(SAMPLES_DIR, f"resultado_{filename}")
    cv2.imwrite(out_path, dashboard)
    print(f"Resultado guardado en {out_path}")
    print(f"Confianza: {pred*100:.1f}%")
    print(f"Resultado: {'SIGNOS DETECTADOS' if pred >= 0.3 else 'SIN SIGNOS'}")
    if os.environ.get('DISPLAY'):
        cv2.imshow("RetinaScan - Analisis preventivo", dashboard)
        print("Presione cualquier tecla en la ventana para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Selecciona una imagen de retina",
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.webp *.bmp")]
        )
        root.destroy()
        if not path:
            sys.exit(0)
    else:
        path = sys.argv[1]
    predict_image(path)
