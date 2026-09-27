# RetinaScan

Proyecto académico de clasificación binaria de imágenes de fondo de ojo con
TensorFlow/Keras, MobileNetV2 y OpenCV. Incluye preparación de datos, entrenamiento,
análisis de imágenes con visualización Grad-CAM y una demostración con cámara.

Es un prototipo educativo; sus resultados no constituyen un diagnóstico médico.

## Estructura

```text
retinascan/
├── scripts/
│   ├── prepare_dataset.py  # Preparación de APTOS en dos clases
│   ├── train.py            # Entrenamiento y evaluación
│   ├── predict_image.py    # Análisis de una imagen
│   ├── predict_camera.py   # Demostración con webcam
│   └── dashboard.py        # Composición de resultados visuales
├── data/                  # Datos preparados, excluidos de Git
├── models/                # Modelo y gráficas, excluidos de Git
├── samples/               # Resultados y capturas, excluidos de Git
└── requirements.txt
```

## Instalación

Ejecuta los comandos desde la raíz del repositorio. El entorno local del proyecto
usa Python 3.12. Las dependencias declaran versiones mínimas, no un entorno fijado
con versiones exactas.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r retinascan/requirements.txt
```

En Windows puedes crear el entorno con `py -3.12 -m venv .venv` y activarlo en
PowerShell con `.venv\Scripts\Activate.ps1`.

El entrenamiento puede ejecutarse en CPU. El uso de GPU requiere configurar
TensorFlow y CUDA para el sistema correspondiente. `activate_retinascan.sh` es un
auxiliar del entorno Linux original con Python 3.12 y bibliotecas NVIDIA dentro
de `.venv`; la activación estándar anterior no depende de ese auxiliar.

## Datos y entrenamiento

Obtén el dataset **APTOS 2019 Blindness Detection** desde Kaggle con tu propia
cuenta y respetando sus condiciones de uso. Descomprímelo junto a `retinascan/`
de forma que existan estas rutas:

```text
aptos2019-blindness-detection/
├── train.csv
└── train_images/
    └── ...png
```

```bash
python retinascan/scripts/prepare_dataset.py
python retinascan/scripts/train.py
```

La preparación agrupa `diagnosis == 0` como `sano` y el resto como `retinopatia`.
Selecciona hasta 1000 imágenes por clase y divide 80 % para entrenamiento y 20 %
para validación con semilla 42. Usa una carpeta `retinascan/data/` sin datos de
preparaciones anteriores para evitar mezclar particiones.

El entrenamiento utiliza pesos ImageNet de MobileNetV2 (requiere conexión si no
están en caché) y genera `retinascan/models/retinascan.h5` y gráficas de evaluación.
El dataset y el modelo entrenado no se distribuyen en este repositorio. Para usar
un modelo existente compatible con estos scripts, colócalo en esa misma ruta.

## Uso

Después de entrenar o colocar el modelo:

```bash
python retinascan/scripts/predict_image.py /ruta/a/imagen.png
python retinascan/scripts/predict_camera.py
```

El análisis de imagen guarda el resultado en `retinascan/samples/`. Sin argumento,
abre un selector de archivos que requiere Tkinter y un entorno gráfico.
La demostración con webcam requiere cámara y escritorio: espacio analiza,
`s` guarda una captura y `q` cierra la ventana activa. El archivo opcional
`retinascan/assets/alerta.wav` permite reproducir una alerta de audio.

## Publicar en GitHub

El `.gitignore` excluye entornos virtuales, credenciales locales, datos,
comprimidos, modelos, capturas y notas de desarrollo. Los archivos `.gitkeep`
conservan las carpetas vacías necesarias. Los archivos excluidos permanecen en tu
equipo.

Crea un repositorio vacío en GitHub, sin README ni licencia iniciales. Desde esta
carpeta, sustituye `TU_USUARIO/TU_REPOSITORIO` por el destino real:

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Preparar RetinaScan para GitHub"
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

Revisa la lista antes del commit: no debe contener `.venv/`, el dataset ni
`retinascan.h5`. Si un archivo ya estuviera versionado, añadirlo al `.gitignore`
no lo retira del índice; usa `git rm --cached <archivo>` para dejar de versionarlo
sin borrar la copia local.

No se ha asignado una licencia al código. Antes de autorizar su reutilización,
elige una licencia y añade el archivo `LICENSE`; las condiciones de los datos y
de los pesos externos se gestionan por separado.
