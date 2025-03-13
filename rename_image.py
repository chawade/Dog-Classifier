# import os

# folder_path = 'E:/DIP/Dog-Classifier/data/german_shepherd'
# new_name = 'german_shepherd'  # ตั้งชื่อใหม่

# for index, filename in enumerate(os.listdir(folder_path)):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         new_filename = f"{new_name}_{index+1}{os.path.splitext(filename)[1]}"
#         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# print("เปลี่ยนชื่อเสร็จแล้ว!")

import os
import shutil
import random

def split_dataset(dataset_path, output_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    assert train_ratio + test_ratio + val_ratio == 1.0,"สัดส่วนต้องรวมกันได้ 1.0"

    # สร้างโฟลเดอร์ปลายทาง
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(output_path, split)
        os.makedirs(split_path, exist_ok=True)

    # วนลูปแยกข้อมูลตามคลาส
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # อ่านไฟล์ทั้งหมดในคลาส
        images = os.listdir(class_path)
        random.shuffle(images)

        # คำนวณจำนวนรูปแต่ละชุด
        total = len(images)
        train_count = int(total * train_ratio)
        test_count = int(total * test_ratio)
        val_count = total - train_count - test_count  # ที่เหลือเป็น validation
        
        # แยกชุดข้อมูล
        splits = {
            "train": images[:train_count],
            "test": images[train_count:train_count + test_count],
            "val": images[train_count + test_count:]
        }

        # คัดลอกไฟล์ไปยังโฟลเดอร์ที่เหมาะสม
        for split, split_images in splits.items():
            split_class_path = os.path.join(output_path, split, class_name)
            os.makedirs(split_class_path, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(split_class_path, img))

    print("📂 การแบ่งข้อมูลเสร็จสิ้น!")

# ใช้งานฟังก์ชัน
dataset_path = "app/dataset"  # โฟลเดอร์ต้นฉบับ
output_path = "app/data"  # โฟลเดอร์ที่ต้องการให้สร้าง train/test/val

split_dataset(dataset_path, output_path)

