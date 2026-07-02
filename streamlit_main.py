import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from src.utils.spectrogram_utils import signal_to_spectrogram

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Micro-Doppler Target Classification",
    layout="centered"
)

st.title("Micro-Doppler Target Classification using CNN")

WINDOW_SIZE = 1000

# =====================================================
# LOAD MODEL
# =====================================================

try:
    model = load_model("models/cnn_model.keras")
    st.success("CNN Model Loaded Successfully")

except Exception as e:
    st.error(e)
    st.stop()

# =====================================================
# UPLOAD EXCEL
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Bird_Drone.xlsx",
    type=["xlsx"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

if not {"label", "value"}.issubset(df.columns):
    st.error("Excel must contain 'label' and 'value' columns.")
    st.stop()

# =====================================================
# CREATE RANDOM SAMPLES
# =====================================================

samples = []

bird = df[df["label"] == 0]["value"].values
drone = df[df["label"] == 1]["value"].values

# Two Bird Samples
for _ in range(2):

    start = np.random.randint(0, len(bird) - WINDOW_SIZE)

    samples.append((bird[start:start + WINDOW_SIZE], 0))

# Two Drone Samples
for _ in range(2):

    start = np.random.randint(0, len(drone) - WINDOW_SIZE)

    samples.append((drone[start:start + WINDOW_SIZE], 1))

np.random.shuffle(samples)

classes = ["Bird", "Drone"]

results = []

# =====================================================
# DISPLAY RESULTS
# =====================================================

for i, (signal, actual) in enumerate(samples):

    st.markdown("---")

    st.subheader(f"Sample {i+1}")

    # -------------------------------------------------
    # Frequency Spectrum
    # -------------------------------------------------

    st.markdown("#### Frequency Spectrum")

    fft = np.abs(np.fft.rfft(signal))

    freq = np.fft.rfftfreq(len(signal), d=1/1000)

    fft[0] = 0

    fig, ax = plt.subplots(figsize=(5,2))

    ax.plot(freq, fft)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.set_xlim(0, 500)

    st.pyplot(fig)

    plt.close(fig)

    # -------------------------------------------------
    # Spectrogram
    # -------------------------------------------------

    spec = signal_to_spectrogram(signal)

    st.markdown("#### Spectrogram")

    fig2, ax2 = plt.subplots(figsize=(2.5,2.5))

    ax2.imshow(
        spec,
        cmap="viridis",
        origin="lower",
        aspect="auto"
    )

    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    x = spec.reshape(1,128,128,1)

    prediction = model.predict(x, verbose=0)[0]

    predicted = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Actual",
            classes[actual]
        )

    with col2:
        st.metric(
            "Predicted",
            classes[predicted]
        )

    st.progress(float(confidence / 100))

    st.write(f"**Confidence : {confidence:.2f}%**")

    results.append({

        "Sample": i + 1,

        "Actual": classes[actual],

        "Predicted": classes[predicted],

        "Confidence (%)": round(confidence,2)

    })

# =====================================================
# SUMMARY TABLE
# =====================================================

st.markdown("---")

st.header("Prediction Summary")

summary = pd.DataFrame(results)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True
)