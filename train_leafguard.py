import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os

DATASET_DIR = "Plant_leave_diseases_dataset_with_augmentation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_FILENAME = "leafguard_model.h5"
LABELS_FILENAME = "class_labels.json"

print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

with open(LABELS_FILENAME, "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Building model...")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_FILENAME, save_best_only=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_FILENAME)
print(f"Model saved to {MODEL_FILENAME}")
print(f"Class labels saved to {LABELS_FILENAME}")

# The code above loads a dataset, builds a VGG16-based model, trains it, and saves both the model and class labels.
# CONFIGURATION
# LOAD DATASETS
# DATA PREFETCHING (Performance boost)
# BUILD MODEL USING VGG16
# TRAIN MODEL
# SAVE MODEL
