# Micro-Doppler Target-Based Classification

A deep learning project that classifies targets (**birds vs. drones**) from their micro-Doppler radar signatures, using spectrogram analysis and a CNN. It includes a Streamlit app for interactive inference.

> **Data note:** all signals in this project are **synthetic**, generated from simple sine-wave models (see [Data](#-data)). The model reaches near-perfect accuracy because the two classes sit at different fixed frequencies, so this repo demonstrates the pipeline end to end, not performance on real radar returns.

## 🎯 Overview

Micro-Doppler signatures arise from the periodic motion of target components (for example rotating drone blades versus flapping bird wings) superimposed on the target's bulk Doppler shift. This project:

- Generates synthetic radar-like signals for birds and drones
- Converts 1-second signal windows into spectrograms (time-frequency images)
- Trains a Convolutional Neural Network (CNN) to classify the spectrograms
- Provides a Streamlit app that classifies random windows from an uploaded dataset and shows the spectrum, the spectrogram, the prediction and the confidence

## 📁 Project Structure

```
-Micro-Doppler-Target-Based-Classification/
├── data/
│   └── raw/
│       └── Bird_Drone.xlsx          # Synthetic dataset (label, value columns)
├── models/
│   └── cnn_model.keras              # Trained CNN model
├── src/
│   ├── data/
│   │   ├── signal_generator.py      # Synthetic bird and drone signal models
│   │   └── create_dataset.py        # Builds Bird_Drone.xlsx from the generators
│   ├── models/
│   │   ├── cnn_model.py             # CNN architecture
│   │   └── train_model.py           # Training and evaluation script
│   └── utils/
│       └── spectrogram_utils.py     # Signal-to-spectrogram conversion
├── streamlit_main.py                # Streamlit app entry point
├── requirements.txt
└── README.md
```

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/RaipallyArjun/-Micro-Doppler-Target-Based-Classification.git
   cd ./-Micro-Doppler-Target-Based-Classification
   ```
   The folder name starts with a hyphen, so use `./` in front of it when you `cd`.

2. **Create a virtual environment** (Python 3.9 to 3.12 recommended for TensorFlow compatibility)
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run all commands from the project root.

### Generate the dataset (optional)
`data/raw/Bird_Drone.xlsx` is already included. To rebuild it:
```bash
python src/data/create_dataset.py
```

### Train the model
```bash
python -m src.models.train_model
```
Use `-m` so that Python can find the `src` package. Running `python src/models/train_model.py` fails with `No module named 'src'`.

### Run the Streamlit app
```bash
streamlit run streamlit_main.py
```
Upload `data/raw/Bird_Drone.xlsx` in the app. It picks two random bird windows and two random drone windows, then shows each window's frequency spectrum, spectrogram, actual label, predicted label and confidence.

## 🧠 Model

- **Architecture:** three Conv2D blocks (16, 32 and 64 filters) with batch normalization and max pooling, then global average pooling, a 64-unit dense layer with dropout (0.3), and a 2-class softmax
- **Input:** 128×128 single-channel spectrograms
- **Output:** Bird or Drone
- **Saved model:** `models/cnn_model.keras`

### Spectrogram pipeline
Each 1000-sample window (1 second at 1 kHz) has its mean removed, is converted with a Hann-window STFT (`nperseg=256`, `noverlap=200`, `nfft=512`), scaled to decibels, min-max normalized and resized to 128×128.

## 📊 Data

`data/raw/Bird_Drone.xlsx` has two columns, `label` (0 = bird, 1 = drone) and `value` (signal sample). It holds 20,000 samples per class (20 one-second segments at 1 kHz). The signals are generated, not measured:

- **Bird:** 20 Hz wingbeat plus a 40 Hz harmonic (amplitude 0.4), with Gaussian noise
- **Drone:** 50 Hz carrier plus 120 Hz (0.5) and 180 Hz (0.3) blade components, with Gaussian noise

## ⚠️ Limitations

- The data is synthetic and the two classes are separated by fixed frequencies, so near-perfect accuracy is expected and says little about real radar performance.
- The training script uses overlapping windows (95% overlap) with a random train/test split, and the test set also drives early stopping. This is fine for a demo on easy data, but an honest evaluation on real data would split by recording or segment and keep a separate test set.
- The app classifies windows taken from the uploaded dataset, not live radar input.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas / SciPy / OpenCV
- scikit-learn (train/test split and metrics)
- Streamlit
- Matplotlib

## 📌 Future Improvements

- Evaluate on real or physically simulated radar data, with a split by segment or recording
- Add more target classes (for example different UAV types and bird species)
- Try other architectures (LSTM, CNN-LSTM hybrid, Transformers)
- Add noise and Doppler-shift augmentation to test robustness