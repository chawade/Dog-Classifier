import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Import modules from utils
from app.utils.config import MODEL_PATH, DATASET_PATH, CLASS_NAMES
from app.utils.file_handler import save_class_labels

# Import local modules
from app.models import create_model
from app.preprocess import create_generators

def compute_class_weights(train_gen):
    train_labels = np.concatenate([np.argmax(y, axis=1) for _, y in train_gen])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    return dict(enumerate(class_weights))

def train():
    # Configurations
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    MODEL_DIR = os.path.dirname(MODEL_PATH)

    # Data Generators
    train_gen, val_gen, _ = create_generators(
        train_dir=os.path.join(DATASET_PATH, "train"),
        val_dir=os.path.join(DATASET_PATH, "val"),
        test_dir=os.path.join(DATASET_PATH, "test"),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Compute Class Weights
    class_weight_dict = compute_class_weights(train_gen)
    print("Class weights:", class_weight_dict)

    # Save Class Indices
    save_class_labels(train_gen.class_indices, os.path.join(MODEL_DIR, "class_labels.json"))
    print("Class indices saved.")

    # Create Model
    model = create_model(num_classes=train_gen.num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), monitor='val_loss', save_best_only=True, mode='min')
    ]

    # Phase 1: Train Base Model
    print("===== Phase 1: Initial Training =====")
    model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks, class_weight=class_weight_dict)

    # Phase 2: Fine-tuning
    print("===== Phase 2: Fine-tuning =====")
    base_model = model.layers[0]
    for layer in base_model.layers[-int(len(base_model.layers) * 0.3):]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks, class_weight=class_weight_dict)

    # Save Final Model
    model.save(MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()
