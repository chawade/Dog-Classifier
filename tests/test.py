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

import unittest
from tensorflow.keras.models import load_model
import os

class TestModel(unittest.TestCase):
    def test_model_loading(self):
        # พาธไปยังโมเดล
        model_path = os.path.join("app", "models", "thai_dog_breed_classifier.h5")
        
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        self.assertTrue(os.path.exists(model_path), "ไม่พบไฟล์โมเดล! รัน app/train.py ก่อน")
        
        # โหลดโมเดล
        model = load_model(model_path)
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()