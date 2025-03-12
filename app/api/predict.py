import tensorflow as tf
import numpy as np
from PIL import Image
from app.utils.config import MODEL_PATH

# โหลดโมเดลที่ฝึกไว้
model = tf.keras.models.load_model(MODEL_PATH)
labels = ["Bangkaew", "German Shepherd", "Rottweiler", "Shiba", "Thai Ridgeback"]

def predict_breed(image_file):
    image = Image.open(image_file.file).resize((224, 224))  # ปรับขนาดภาพให้ตรงกับโมเดล
    image_array = np.array(image) / 255.0  # Normalize ค่า pixel
    image_array = np.expand_dims(image_array, axis=0)  # เพิ่ม batch dimension

    prediction = model.predict(image_array)
    breed = labels[np.argmax(prediction)]
    
    return breed
