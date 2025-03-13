import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(
    train_dir,
    val_dir,
    test_dir,
    img_size=(224, 224),
    batch_size=32
):

    # Data Augmentation สำหรับ train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # สร้าง Train Generator
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # สร้าง Validation Generator
    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # สร้าง Test Generator
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # โดยทั่วไปไม่สุ่ม เพื่อประเมินผลได้ง่าย
    )

    return train_generator, val_generator, test_generator
