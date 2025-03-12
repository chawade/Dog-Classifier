import json
import os

def save_class_labels(class_indices, save_path):
    """บันทึกชื่อคลาสเป็นไฟล์ JSON"""
    with open(save_path, "w") as f:
        json.dump(class_indices, f)

def load_class_labels(load_path):
    """โหลดชื่อคลาสจากไฟล์ JSON"""
    with open(load_path, "r") as f:
        return json.load(f)