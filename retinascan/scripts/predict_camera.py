import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import pygame
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard import create_dashboard, get_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
IMG_SIZE = 224

model = None
grad_model = None
capture_log = []

TRAINING_METRICS = {
    "Recall (sano):": "96%",
    "Recall (retinopatia):": "93%",
    "AUC-ROC:": "0.989",
}

def load_model():
    global model, grad_model
    if model is None:
        model_path = os.path.join(MODELS_DIR, "retinascan.h5")
        model = keras.models.load_model(model_path, compile=False)
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

def play_alert():
    try:
        pygame.mixer.init()
        alert_path = os.path.join(ASSETS_DIR, "alerta.wav")
        if os.path.exists(alert_path):
            pygame.mixer.Sound(alert_path).play()
    except:
        pass

def overlay_guide(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cx, cy = w // 2, h // 2
    rx, ry = w // 3, h // 3
    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(0, 360, 20):
        a1 = np.radians(i)
        a2 = np.radians(i + 8)
        x1 = int(cx + rx * np.cos(a1))
        y1 = int(cy + ry * np.sin(a1))
        x2 = int(cx + rx * np.cos(a2))
        y2 = int(cy + ry * np.sin(a2))
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(overlay, "Coloque la imagen de retina aqui",
                (cx - 180, cy + ry + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return overlay

def draw_log(overlay, log_entries, h):
    log_y = h - 20
    for entry in log_entries[-5:]:
        log_y -= 22
        color = (0, 200, 200) if "SIGNOS DETECTADOS" in entry else (180, 180, 180)
        cv2.putText(overlay, entry, (20, log_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def analysis_overlay(frame, status="idle"):
    h, w = frame.shape[:2]
    overlay = overlay_guide(frame)
    cv2.putText(overlay, "[ESPACIO] Analizar  [S] Guardar frame  [Q] Salir",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    draw_log(overlay, capture_log, h)
    if status == "processing":
        cv2.putText(overlay, "ANALIZANDO...", (w // 2 - 80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return overlay

def run_camera():
    load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return
    print("Presione ESPACIO para analizar, S para guardar, Q para salir")
    prev_time = cv2.getTickCount()
    fps = 0
    status = "idle"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = cv2.getTickCount()
        dt = (curr_time - prev_time) / cv2.getTickFrequency()
        if dt > 0:
            fps = int(1.0 / dt)
        prev_time = curr_time

        overlay = analysis_overlay(frame, status)
        cv2.putText(overlay, f"FPS: {fps}", (overlay.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("RetinaScan - Camara", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            status = "processing"
            cv2.imshow("RetinaScan - Camara", analysis_overlay(frame, status))
            cv2.waitKey(1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_norm, original_resized, img_clahe = preprocess_retina(rgb)
            img_array = np.expand_dims(img_norm, axis=0)
            pred = model.predict(img_array, verbose=0)[0][0]
            heatmap = grad_cam(img_array)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            dashboard = create_dashboard(original_resized, img_clahe, heatmap_rgb, float(pred), TRAINING_METRICS)

            timestamp = datetime.now().strftime("%H:%M:%S")
            label = "SIGNOS DETECTADOS" if pred >= 0.3 else "SIN SIGNOS"
            capture_log.append(f"{timestamp} - {label} ({pred*100:.0f}%)")
            if pred >= 0.3:
                play_alert()

            cv2.imshow("RetinaScan - Analisis", dashboard)
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q') or k == 27:
                    cv2.destroyWindow("RetinaScan - Analisis")
                    break
                elif k == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(SAMPLES_DIR, f"captura_{ts}.png")
                    cv2.imwrite(path, dashboard)
                    print(f"Guardado en {path}")
                elif k == 32:
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    cv2.destroyWindow("RetinaScan - Analisis")
                    break

            cap.release()
            cap = cv2.VideoCapture(0)
            status = "idle"
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAMPLES_DIR, f"captura_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"Frame guardado en {path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()
