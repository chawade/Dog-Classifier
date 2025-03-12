import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import json

# โหลดโมเดล
model_path = os.path.join("app", "models", "best_model.h5")
model = load_model(model_path)

# โหลดคลาส labels
with open(os.path.join("app", "models", "class_labels.json"), "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

def predict_image(img_path):
    # ตรวจสอบไฟล์
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("ไฟล์รูปภาพเสียหายหรือไม่รองรับ")
    
    # ประมวลผลภาพ
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # ทำนาย
    prediction = model.predict(img)[0]  # ดึงผลลัพธ์เป็น array 1D
    return prediction

# ทดสอบ
img_path = "test.jpg"  # เปลี่ยนพาธตามจริง
try:
    prediction = predict_image(img_path)
    
    # แสดงผลทั้ง 5 คลาส
    print("===== ผลการทำนาย =====")
    for i, (class_name, confidence) in enumerate(zip(class_names, prediction)):
        print(f"{i+1}. {class_name}: {confidence*100:.2f}%")
    
    # หาคลาสที่ได้คะแนนสูงสุด
    predicted_class = np.argmax(prediction)
    print(f"\nสายพันธุ์ที่ทำนายได้: {class_names[predicted_class]} ({np.max(prediction)*100:.2f}%)")
    
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")

# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# from PIL import Image

# # ---------------------- #
# # ตั้งค่าเริ่มต้น
# # ---------------------- #
# THAI_MODEL_PATH = 'app/models/best_model.h5'
# KAGGLE_MODEL_PATH = 'app/models/dog-breed-classification.h5'

# # Class indices ของ Kaggle Model (แก้ไขตาม breeds array จริง)
# BREED_INDICES = {
#     'german_shepherd': 46,
#     'rottweiler': 91,
#     'golden_retrieve': 49,
#     'thai_bangkaew': 0,
#     'thai_ridgeback': 1
# }

# # กำหนด Custom Objects
# custom_objects = {"KerasLayer": hub.KerasLayer}

# # ---------------------- #
# # โหลดโมเดล
# # ---------------------- #
# thai_model = tf.keras.models.load_model(THAI_MODEL_PATH)
# kaggle_model = tf.keras.models.load_model(KAGGLE_MODEL_PATH, custom_objects=custom_objects)

# # ---------------------- #
# # ฟังก์ชันประมวลผลภาพ
# # ---------------------- #
# def preprocess_image(img_path, target_size):
#     img = Image.open(img_path).convert('RGB')
#     img = img.resize(target_size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     return np.expand_dims(img_array, axis=0) / 255.0

# # ---------------------- #
# # ฟังก์ชันทำนาย
# # ---------------------- #
# def predict_breeds(img_path):
#     # โหลดและปรับขนาดภาพ
#     img_thai = preprocess_image(img_path, (224, 224))
#     img_kaggle = preprocess_image(img_path, (299, 299))
    
#     # ทำนายผล
#     thai_pred = thai_model.predict(img_thai, verbose=0)[0]
#     kaggle_pred = kaggle_model.predict(img_kaggle, verbose=0)[0]

#     # รวมผลลัพธ์
#     results = {
#         'บางแก้ว': float(thai_pred[0]) * 100,
#         'หลังอาน': float(thai_pred[1]) * 100,
#         'เยอรมันเชฟเฟิร์ด': float(kaggle_pred[BREED_INDICES['german_shepherd']]) * 100,
#         'ร็อตไวเลอร์': float(kaggle_pred[BREED_INDICES['rottweiler']]) * 100,
#         'โกลเด้น': float(kaggle_pred[BREED_INDICES['golden_retrieve']]) * 100
#     }

#     return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# # ---------------------- #
# # ตัวอย่างการใช้งาน
# # ---------------------- #
# if __name__ == "__main__":
#     predictions = predict_breeds('test.jpg')
#     print("ผลลัพธ์การทำนาย (%):")
#     for breed, prob in predictions.items():
#         print(f"{breed}: {prob:.2f}%")