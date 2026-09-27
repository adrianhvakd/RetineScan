import pandas as pd
import shutil
import os
import random
from sklearn.model_selection import train_test_split

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "..", "aptos2019-blindness-detection", "train.csv")
IMAGES_SRC = os.path.join(BASE_DIR, "..", "aptos2019-blindness-detection", "train_images")
DEST_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(CSV_PATH)

df["label"] = df["diagnosis"].apply(lambda x: "sano" if x == 0 else "retinopatia")

sano = df[df["label"] == "sano"]["id_code"].tolist()
retinopatia = df[df["label"] == "retinopatia"]["id_code"].tolist()

SAMPLE_SIZE = 1000
sano = random.sample(sano, min(SAMPLE_SIZE, len(sano)))
retinopatia = random.sample(retinopatia, min(SAMPLE_SIZE, len(retinopatia)))

sano_train, sano_val = train_test_split(sano, test_size=0.2, random_state=42)
ret_train, ret_val = train_test_split(retinopatia, test_size=0.2, random_state=42)

splits = {
    "train/sano": sano_train,
    "val/sano": sano_val,
    "train/retinopatia": ret_train,
    "val/retinopatia": ret_val,
}

for folder, ids in splits.items():
    dest = os.path.join(DEST_DIR, folder)
    os.makedirs(dest, exist_ok=True)
    for img_id in ids:
        src = os.path.join(IMAGES_SRC, f"{img_id}.png")
        if os.path.exists(src):
            shutil.copy2(src, dest)
        else:
            print(f"Advertencia: no se encontró {src}")

print("Resumen final:")
print(f'  Train: {len(sano_train)} sanas, {len(ret_train)} con retinopatía')
print(f'  Val:   {len(sano_val)} sanas, {len(ret_val)} con retinopatía')
print("Dataset preparado exitosamente.")
