from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
from torchaudio import transforms
import io
import time
import json
import os
import soundfile as sf 
import logging

app = Flask(__name__, template_folder='templates')

# --- Global variables for model and transforms ---
model = None
mel_spectrogram = None
target_classes = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model definition ---
class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(64 * 8 * 4, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

# --- Load model and metadata ---
def load_model():
    global model, mel_spectrogram, target_classes

    try:
        # Load class info
        with open('class_info.json', 'r') as f:
            class_info = json.load(f)
            target_classes = class_info['target_classes']

        # Initialize MelSpectrogram transform
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        # Load model weights
        model = AudioClassifier(num_classes=len(target_classes))

        model_path = 'model.pth'
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found. Using untrained model.")
            # Продовжуємо без попередньо навченої моделі
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval()
        logger.info("✅ Model loaded successfully.")
        logger.info(f"Target classes: {target_classes}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

# --- Audio preprocessing ---
def preprocess_audio(audio_data, sample_rate):
    # Convert to mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = torch.mean(audio_data, dim=1, keepdim=True).T
    elif len(audio_data.shape) == 1:
        audio_data = audio_data.unsqueeze(0)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_data = resampler(audio_data)

    # Mel spectrogram
    spec = mel_spectrogram(audio_data)
    spec = spec.unsqueeze(0)  # add batch dimension
    return spec

# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # ✅ Safe audio loading from memory
        audio_bytes = file.read()
        waveform_np, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        audio_tensor = torch.tensor(waveform_np).T  # [channels, time]

        # Preprocess
        spec = preprocess_audio(audio_tensor, sample_rate)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            spec = spec.to(device)
            outputs = model(spec)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_index = torch.argmax(probabilities).item()
            predicted_class = target_classes[predicted_index]

        inference_time = (time.time() - start_time) * 1000  # ms

        # Response
        response = {
            'prediction': predicted_class,
            'confidence': round(probabilities[predicted_index].item(), 4),
            'probabilities': {cls: round(prob.item(), 4) for cls, prob in zip(target_classes, probabilities)},
            'inference_time_ms': round(inference_time, 2),
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': target_classes})

# --- Run server ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Start the Flask server")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    if args.serve:
        print("Loading model...")
        try:
            load_model()
            print("🚀 Starting Flask server...")
            app.run(host="0.0.0.0", port=args.port, debug=False)
        except Exception as e:
            print(f"❌ Failed to start server: {e}")