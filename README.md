# Micro-Doppler Target-Based Classification

A deep learning project for classifying targets (e.g. **birds vs. drones**) based on their micro-Doppler radar signatures, using spectrogram analysis and a CNN model. Includes a Streamlit app for interactive inference.

## 🎯 Overview

Micro-Doppler signatures arise from the periodic motion of target components (e.g. rotating drone blades vs. flapping bird wings) superimposed on the target's bulk Doppler shift. This project:

- Generates/loads synthetic and real radar signal data
- Converts raw signals into spectrograms (time-frequency representations)
- Trains a Convolutional Neural Network (CNN) to classify targets from spectrograms
- Provides a Streamlit web app for real-time classification

## 📁 Project Structure

```
Micro_Doppler_Target_Based_Classification/
├── data/
│   └── raw/
│       └── Bird_Drone.xlsx        # Raw dataset (bird & drone samples)
├── models/
│   └── cnn_model.keras            # Trained CNN model
├── src/
│   ├── models/
│   │   ├── cnn_model.py           # CNN architecture definition
│   │   └── train_model.py         # Model training script
│   └── utils/
│       ├── __init__.py
│       └── spectrogram_utils.py   # Signal-to-spectrogram conversion utilities
├── streamlit_main.py              # Streamlit app entry point
├── requirements.txt               # Python dependencies
└── README.md
```

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Micro_Doppler_Target_Based_Classification.git
   cd Micro_Doppler_Target_Based_Classification
   ```

2. **Create a virtual environment** (Python 3.9–3.12 recommended for TensorFlow compatibility)
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

### Train the model
```bash
python src/models/train_model.py
```

### Run the Streamlit app
```bash
streamlit run streamlit_main.py
```

This launches a local web interface where you can upload/generate a signal and get a real-time bird vs. drone classification.

## 🧠 Model

- **Architecture**: Convolutional Neural Network (CNN) built with TensorFlow/Keras
- **Input**: Spectrogram images derived from micro-Doppler radar signals
- **Output**: Binary classification — Bird / Drone
- **Saved model**: `models/cnn_model.keras`

## 📊 Data

Raw data is stored in `data/raw/Bird_Drone.xlsx`, containing labeled radar signal samples for birds and drones. `spectrogram_utils.py` handles converting these raw signals into spectrogram representations suitable for CNN input.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Streamlit
- Matplotlib (for spectrogram visualization)

## 📌 Future Improvements

- Expand dataset with more target classes (e.g. UAV types, birds species)
- Experiment with alternative architectures (LSTM, CNN-LSTM hybrid, Transformers)
- Add model evaluation metrics and confusion matrix visualization
- Deploy the Streamlit app (e.g. Streamlit Community Cloud, Hugging Face Spaces)
