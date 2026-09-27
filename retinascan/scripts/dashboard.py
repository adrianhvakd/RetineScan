import cv2
import numpy as np
from datetime import datetime

PANEL_SIZE = 300
INFO_WIDTH = 260
MARGIN = 15
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)
DARK_BG = (30, 30, 30)
PANEL_BG = (40, 40, 40)
GREEN = (0, 200, 0)
YELLOW = (0, 230, 255)
RED = (0, 0, 230)
CYAN = (255, 255, 0)

def rounded_rect(img, top_left, bottom_right, color, thickness=-1, radius=8):
    x1, y1 = top_left
    x2, y2 = bottom_right
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def get_risk(confidence):
    pct = confidence * 100
    if pct < 40:
        return GREEN, "Bajo riesgo aparente", "bajo"
    elif pct < 70:
        return YELLOW, "Riesgo moderado", "moderado"
    else:
        return RED, "Alto riesgo", "alto"

def draw_traffic_light(img, x, y, size, risk_level):
    colors = [GREEN, YELLOW, RED]
    for i, (color, label) in enumerate(zip(colors, ["bajo", "moderado", "alto"])):
        cx = x + i * (size + 8)
        cy = y + size // 2
        active = label == risk_level
        cv2.circle(img, (cx, cy), size // 2 - 2, color if active else (60, 60, 60), -1)
        cv2.circle(img, (cx, cy), size // 2 - 2, WHITE if active else (40, 40, 40), 1)

def draw_confidence_bar(img, x, y, w, h, confidence, risk_color):
    bg_color = (60, 60, 60)
    rounded_rect(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * confidence)
    if fill_w > 0:
        fill_color = risk_color
        rounded_rect(img, (x, y), (x + fill_w, y + h), fill_color, -1)
    rounded_rect(img, (x, y), (x + w, y + h), LIGHT_GRAY, 1)
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for t in thresholds:
        tx = int(x + w * t)
        cv2.line(img, (tx, y), (tx, y + h), (80, 80, 80), 1)
        if t in [0.4, 0.7]:
            cv2.line(img, (tx, y - 4), (tx, y + h + 4), (180, 180, 180), 1)
    cv2.putText(img, "0%", (x, y + h + 18), FONT_SMALL, 0.5, GRAY, 1)
    cv2.putText(img, "40%", (int(x + w * 0.4) - 10, y + h + 18), FONT_SMALL, 0.5, GRAY, 1)
    cv2.putText(img, "70%", (int(x + w * 0.7) - 10, y + h + 18), FONT_SMALL, 0.5, GRAY, 1)
    cv2.putText(img, "100%", (x + w - 30, y + h + 18), FONT_SMALL, 0.5, GRAY, 1)
    label_x = x + w + 12
    cv2.putText(img, f"{confidence * 100:.1f}%", (label_x, y + h - 2), FONT, 0.65, WHITE, 2)

def create_dashboard(img_original, img_clahe, heatmap, confidence, training_metrics=None):
    pct = confidence * 100
    risk_color, risk_label, risk_level = get_risk(confidence)
    is_retinopathy = confidence >= 0.3

    panel_w = PANEL_SIZE
    panel_h = PANEL_SIZE
    bar_h = 24
    bar_y_start = panel_h + MARGIN
    total_w = panel_w * 3 + MARGIN * 4 + INFO_WIDTH
    total_h = panel_h + MARGIN + bar_h + 50 + 70

    dashboard = np.full((total_h, total_w, 3), DARK_BG, dtype=np.uint8)

    panels = [
        (img_original, "Original"),
        (img_clahe, "Post CLAHE"),
        (cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) if heatmap.ndim == 3 and heatmap.shape[2] == 3 else heatmap, "Grad-CAM"),
    ]

    for i, (img_data, title) in enumerate(panels):
        px = MARGIN + i * (panel_w + MARGIN)
        py = MARGIN
        rounded_rect(dashboard, (px - 2, py - 2), (px + panel_w + 2, py + panel_h + 2), PANEL_BG, -1)
        display_img = cv2.resize(img_data, (panel_w, panel_h))
        if display_img.ndim == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        if display_img.dtype == np.float32 or display_img.dtype == np.float64:
            display_img = (display_img * 255).astype(np.uint8)
        dashboard[py:py + panel_h, px:px + panel_w] = display_img[:panel_h, :panel_w]
        (tw, th), _ = cv2.getTextSize(title, FONT, 0.5, 1)
        tx = px + (panel_w - tw) // 2
        cv2.rectangle(dashboard, (tx - 8, py), (tx + tw + 8, py + 22), (0, 0, 0, 150), -1)
        cv2.putText(dashboard, title, (tx, py + 16), FONT, 0.5, WHITE, 1)

    si = MARGIN + 3 * (panel_w + MARGIN)
    info_x = si + MARGIN
    info_w = INFO_WIDTH - MARGIN * 2

    rounded_rect(dashboard, (si, MARGIN), (si + INFO_WIDTH, MARGIN + panel_h), PANEL_BG, -1)

    cv2.putText(dashboard, "RetinaScan", (info_x, MARGIN + 30), FONT, 0.8, CYAN, 2)
    cv2.putText(dashboard, "Screen. preventivo", (info_x, MARGIN + 50), FONT_SMALL, 0.7, LIGHT_GRAY, 1)

    result_text = "SIN SIGNOS" if not is_retinopathy else "SIGNOS DETECTADOS"
    result_color = GREEN if not is_retinopathy else RED
    cv2.putText(dashboard, result_text, (info_x, MARGIN + 90), FONT, 0.8, result_color, 2)

    traffic_y = MARGIN + 105
    draw_traffic_light(dashboard, info_x, traffic_y, 24, risk_level)
    cv2.putText(dashboard, risk_label, (info_x + 90, traffic_y + 17), FONT_SMALL, 0.7, risk_color, 1)

    dt_text = datetime.now().strftime("%Y-%m-%d %H:%M")
    cv2.putText(dashboard, dt_text, (info_x, MARGIN + 160), FONT_SMALL, 0.6, LIGHT_GRAY, 1)
    cv2.putText(dashboard, "MobileNetV2 + TL", (info_x, MARGIN + 180), FONT_SMALL, 0.6, LIGHT_GRAY, 1)
    cv2.putText(dashboard, "Dataset: APTOS 2019", (info_x, MARGIN + 198), FONT_SMALL, 0.6, LIGHT_GRAY, 1)

    if training_metrics:
        y0 = MARGIN + 225
        for k, v in training_metrics.items():
            cv2.putText(dashboard, f"{k}: {v}", (info_x, y0), FONT_SMALL, 0.55, LIGHT_GRAY, 1)
            y0 += 20

    bar_x = MARGIN
    bar_y = bar_y_start
    bar_w = panel_w * 3 + MARGIN * 2
    draw_confidence_bar(dashboard, bar_x, bar_y, bar_w, bar_h, confidence, risk_color)

    legend_y = bar_y + bar_h + 35
    legend_colors = [GREEN, YELLOW, RED]
    legend_labels = ["Bajo riesgo (0-40%)", "Riesgo moderado (40-70%)", "Alto riesgo (70-100%)"]
    for i, (lc, ll) in enumerate(zip(legend_colors, legend_labels)):
        lx = bar_x + i * (bar_w // 3)
        cv2.circle(dashboard, (lx + 8, legend_y + 6), 5, lc, -1)
        cv2.putText(dashboard, ll, (lx + 18, legend_y + 10), FONT_SMALL, 0.55, LIGHT_GRAY, 1)

    disc_y = total_h - 60
    rounded_rect(dashboard, (MARGIN, disc_y), (total_w - MARGIN, total_h - MARGIN), (20, 20, 20), -1)
    cv2.putText(dashboard, "ATENCION: Esta herramienta NO reemplaza el diagnostico medico. Consulte un oftalmologo.",
                (MARGIN + 15, disc_y + 22), FONT_SMALL, 0.55, (0, 0, 200), 1)

    return dashboard
