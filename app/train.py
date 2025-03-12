# app/train.py

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# เรียกใช้ฟังก์ชันที่คุณมีอยู่เดิม
from .models import create_model       # ฟังก์ชันสร้างโมเดล
from .preprocess import create_generators  # ฟังก์ชันสร้าง train_gen, val_gen

def train():
    # -------------------------
    # 1) กำหนดพารามิเตอร์ต่าง ๆ
    # -------------------------
    DATASET_PATH = os.path.join("app", "data")  # โฟลเดอร์ที่มี 5 คลาส
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    MODEL_DIR = os.path.join("app", "models")
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "dog_breeds_classifier.h5")

    # -------------------------
    # 2) สร้าง Data Generators + Augmentation
    #    (สมมติว่า create_generators() รองรับพารามิเตอร์ augment)
    # -------------------------
    train_gen, val_gen, test_gen = create_generators(
         train_dir="app/data/train",
        val_dir="app/data/val",
        test_dir="app/data/test",
        img_size=(224, 224),
        batch_size=32
    )

    # -------------------------
    # 3) คำนวณ Class Weights (แก้ปัญหา Imbalanced Data)
    # -------------------------
    # ดึง label (class) ของรูปทั้งหมดใน train_gen
    train_labels = np.array([])
    for i in range(len(train_gen)):
        _, y = train_gen[i]
        train_labels = np.append(train_labels, np.argmax(y, axis=1))
        if len(train_labels) >= train_gen.samples:
            break

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    # -------------------------
    # 4) บันทึกชื่อคลาส (mapping) ไว้ใช้อ้างอิงภายหลัง
    # -------------------------
    class_indices = train_gen.class_indices
    with open(os.path.join(MODEL_DIR, "class_labels.json"), "w") as f:
        json.dump(class_indices, f)
    print("Class indices saved.")

    # -------------------------
    # 5) สร้างโมเดล
    # -------------------------
    num_classes = train_gen.num_classes
    model = create_model(num_classes=num_classes)

    # -------------------------
    # 6) คอมไพล์โมเดล (Phase 1 - Freeze Base Model)
    # -------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # 7) Callbacks
    # -------------------------
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # อาจเพิ่ม patience ให้มากขึ้นได้
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # -------------------------
    # 8) Train (Phase 1 - Freeze Base Model)
    # -------------------------
    print("===== Phase 1: Initial Training with Class Weights =====")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weight_dict
    )

    # -------------------------
    # 9) Fine-tuning (Phase 2 - Unfreeze บางชั้นของ Base Model)
    # -------------------------
    print("===== Phase 2: Fine-tuning =====")
    base_model = model.layers[0]  # สมมติว่า base model อยู่ใน layer 0

    # ตัวอย่าง: Unfreeze 30% สุดท้ายของ base model
    unfreeze_num = int(len(base_model.layers) * 0.3)
    for layer in base_model.layers[-unfreeze_num:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # ลด LR ลง
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weight_dict
    )

    # -------------------------
    # 10) เซฟโมเดล
    # -------------------------
    model.save(MODEL_SAVE_PATH)
    print(f"โมเดลถูกบันทึกไว้ที่: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
