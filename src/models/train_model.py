import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.cnn_model import create_cnn_model
from src.utils.spectrogram_utils import signal_to_spectrogram

# =====================================================
# SETTINGS
# =====================================================

DATASET_PATH = "data/raw/Bird_Drone.xlsx"

WINDOW_SIZE = 1000
STEP_SIZE = 50

MODEL_PATH = "models/cnn_model.keras"

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_excel(DATASET_PATH)

print(df.head())
print(f"\nTotal Samples : {len(df)}")

# =====================================================
# CREATE DATASET
# =====================================================

X = []
labels = []

for label in sorted(df["label"].unique()):

    signal = df[df["label"] == label]["value"].values

    print(f"\nProcessing Label : {label}")

    count = 0

    for start in range(
        0,
        len(signal) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        window = signal[start:start + WINDOW_SIZE]

        spec = signal_to_spectrogram(window)

        X.append(spec)
        labels.append(label)

        count += 1

    print("Windows Created :", count)

# =====================================================
# CONVERT TO NUMPY
# =====================================================

X = np.array(X, dtype=np.float32)

X = X[..., np.newaxis]

# Normalize each spectrogram
for i in range(len(X)):
    X[i] = (
        X[i] - X[i].min()
    ) / (
        X[i].max() - X[i].min() + 1e-8
    )

labels = np.array(labels)

print("\nDataset Shape :", X.shape)

print("Bird Samples :", np.sum(labels == 0))
print("Drone Samples:", np.sum(labels == 1))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train_lbl, y_test_lbl = train_test_split(

    X,

    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels,

    shuffle=True

)

y_train = to_categorical(y_train_lbl, 2)

y_test = to_categorical(y_test_lbl, 2)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# =====================================================
# BUILD MODEL
# =====================================================

model = create_cnn_model()

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

os.makedirs("models", exist_ok=True)

callbacks = [

    EarlyStopping(

        monitor="val_accuracy",

        patience=10,

        restore_best_weights=True

    ),

    ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True

    )

]

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=100,

    batch_size=16,

    callbacks=callbacks,

    verbose=1

)

# =====================================================
# EVALUATE MODEL
# =====================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)

print("\n" + "=" * 40)

print(f"Test Accuracy : {accuracy * 100:.2f}%")

print("=" * 40)

# =====================================================
# PREDICTIONS
# =====================================================

pred = model.predict(X_test)

pred = np.argmax(pred, axis=1)

print("\nPrediction Counts")

print("Bird :", np.sum(pred == 0))

print("Drone:", np.sum(pred == 1))

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix")

print(confusion_matrix(y_test_lbl, pred))

print("\nClassification Report")

print(classification_report(y_test_lbl, pred))

# =====================================================
# SAVE MODEL
# =====================================================

model.save(MODEL_PATH)

print("\nModel Saved Successfully")

print(MODEL_PATH)