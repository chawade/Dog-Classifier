# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# import json
# import os

# # โหลดโมเดล
# model_path = "app/models/dog_breed_classifier.h5"
# if not os.path.exists(model_path):
#     raise FileNotFoundError("ไม่พบไฟล์โมเดล! รัน app/train.py ก่อน")

# model = load_model(model_path)

# # โหลดคลาส labels
# with open("app/models/class_labels.json", "r") as f:
#     class_indices = json.load(f)
# labels = list(class_indices.keys())

# # โหลดรูปภาพ
# img_path = "test.jpg"
# if not os.path.exists(img_path):
#     raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ: {img_path}")

# img = cv2.imread(img_path)
# if img is None:
#     raise ValueError("ไม่สามารถโหลดรูปภาพได้")

# # ประมวลผลภาพ
# img = cv2.resize(img, (224, 224))
# img = img / 255.0
# img = np.expand_dims(img, axis=0)

# # ทำนาย
# predictions = model.predict(img)
# predicted_class = np.argmax(predictions)
# print(f"Predicted Breed: {labels[predicted_class]}")

# import unittest
# from tensorflow.keras.models import load_model
# import os

# class TestModel(unittest.TestCase):
#     def test_model_loading(self):
#         # พาธไปยังโมเดล
#         model_path = os.path.join("app", "models", "best_model.h5")
        
#         # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
#         self.assertTrue(os.path.exists(model_path), "ไม่พบไฟล์โมเดล! รัน app/train.py ก่อน")
        
#         # โหลดโมเดล
#         model = load_model(model_path)
#         self.assertIsNotNone(model)

# if __name__ == "__main__":
#     unittest.main()

# tests/test.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_model():
    # -------------------------
    # 1) กำหนดพารามิเตอร์
    # -------------------------
    MODEL_PATH = os.path.join("app", "models", "dog_breeds_classifier.h5")
    CLASS_LABELS_PATH = os.path.join("app", "models", "class_labels.json")
    TEST_DATA_DIR = os.path.join("app", "data", "test")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # -------------------------
    # 2) โหลดโมเดลและคลาส
    # -------------------------
    model = load_model(MODEL_PATH)
    with open(CLASS_LABELS_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())

    # -------------------------
    # 3) สร้าง Test Generator
    # -------------------------
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        directory=TEST_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # ต้องไม่สลับลำดับเพื่อประเมินผลถูกต้อง
    )

    # -------------------------
    # 4) ประเมินโมเดล
    # -------------------------
    # คำนวณ Loss และ Accuracy
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # ทำนายผลทั้งหมด
    y_pred = model.predict(test_gen).argmax(axis=1)
    y_true = test_gen.classes

    # สร้าง Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # สร้าง Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()