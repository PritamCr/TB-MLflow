import tensorflow as tf
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import yaml

# Paths
DATA_DIR = "Data/raw"
MODEL_PATH = "models/tb_classifier_model.h5"
METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=params["data"]["validation_split"]
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Predictions
y_scores = model.predict(val_generator)
y_pred = (y_scores > 0.5).astype(int)
y_true = val_generator.classes

# Metrics
roc_auc = roc_auc_score(y_true, y_scores)
report = classification_report(
    y_true, y_pred,
    target_names=val_generator.class_indices.keys(),
    output_dict=True
)

metrics = {
    "roc_auc": float(roc_auc),
    "accuracy": report["accuracy"]
}

# Save metrics
with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
