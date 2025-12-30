import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import yaml

# Paths
DATA_DIR = "Data/raw"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Parameters
IMG_SIZE = (params["data"]["img_height"], params["data"]["img_width"])
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=params["data"]["validation_split"],
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Model
model = models.Sequential([
    layers.Conv2D(params["model"]["conv_filters"][0], (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(params["model"]["conv_filters"][1], (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(params["model"]["conv_filters"][2], (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(params["model"]["dense_units"], activation='relu'),
    layers.Dense(params["model"]["output_units"], activation='sigmoid')
])

# Class weights
y_true = val_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_true),
    y=y_true
)
class_weights = dict(enumerate(class_weights))

# Focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma) * y_true + \
                 (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
        return K.mean(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

model.compile(
    optimizer=params["train"]["optimizer"],
    loss=focal_loss(gamma=params["loss"]["focal"]["gamma"], alpha=params["loss"]["focal"]["alpha"]),
    metrics=["accuracy"]
)

# Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights
)

# Save model
model.save(os.path.join(MODEL_DIR, "tb_classifier_model.h5"))
