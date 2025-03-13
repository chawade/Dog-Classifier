# tests/evaluate.py

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

from app.preprocess import create_generators

# 1) โหลดโมเดล
model_path = os.path.join("app", "models", "best_model.h5")
model = load_model(model_path)

# 2) กำหนด path สำหรับ train, val, test
DATA_PATH = os.path.join("app", "data")
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VAL_DIR = os.path.join(DATA_PATH, "val")
TEST_DIR = os.path.join(DATA_PATH, "test")

# 3) สร้าง Generator (train, val, test) - ถ้าต้องการทดสอบก็สนใจเฉพาะ test_gen
_, _, test_gen = create_generators(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    img_size=(224, 224),
    batch_size=32
)

# 4) ทำนายผลทั้งหมดใน Test Set
y_true = test_gen.classes
y_pred = model.predict(test_gen).argmax(axis=1)

# 5) สร้าง Classification Report
class_names = list(test_gen.class_indices.keys())
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 6) สร้าง Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
