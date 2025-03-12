import shutil
import os
import pandas as pd

# กำหนดพาธ
kaggle_csv_path = "kaggle_dataset/labels.csv"
kaggle_image_dir = "kaggle_dataset/train/"
target_dir = "data/"

# อ่านไฟล์ CSV
df = pd.read_csv(kaggle_csv_path)

# กำหนดเงื่อนไข: เลือกเฉพาะ 2 สายพันธุ์ที่ต้องการ
target_breeds = ["rottweiler", "german_shepherd"]  # ชื่อ breed จาก CSV

for index, row in df.iterrows():
    breed = row["breed"].lower()  # แปลงเป็นตัวเล็กเพื่อป้องกันข้อผิดพลาด
    
    # กรองเฉพาะสายพันธุ์ที่ต้องการ
    if breed not in target_breeds:
        continue
    
    # แปลงชื่อโฟลเดอร์ให้ตรงกับโครงสร้าง
    if breed == "rottweiler":
        folder_name = "rottweiler"
    elif breed == "german_shepherd":
        folder_name = "german_shepherd"
    else:
        continue
    
    # สร้างพาธปลายทาง
    dest_dir = os.path.join(target_dir, folder_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    # คัดลอกรูปภาพ (ตรวจสอบนามสกุลไฟล์)
    image_id = row["id"]
    src_path = os.path.join(kaggle_image_dir, f"{image_id}.jpg")
    
    # ตรวจสอบว่าไฟล์มีอยู่จริง
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_dir)
        print(f"คัดลอก: {image_id}.jpg ไปยัง {folder_name}")
    else:
        print(f"ไม่พบไฟล์: {image_id}.jpg")
