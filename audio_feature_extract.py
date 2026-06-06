import tensorflow_hub as hub
import tensorflow as tf
import soundfile as sf
import numpy as np
import os
import pandas as pd

# Load YAMNET
model = hub.load("https://tfhub.dev/google/yamnet/1")

DATA_DIR = "Dataset"
OUTPUT_DIR = "audio_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for category in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, category)
    if not os.path.isdir(folder):
        continue
    
    all_features = []

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)

            waveform, sr = sf.read(filepath)

            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            if sr != 16000:
                waveform = tf.audio.resample(waveform, sr, 16000)

            scores, embeddings, spectrogram = model(waveform)

            feature_vector = np.mean(embeddings, axis=0)

            all_features.append(feature_vector.numpy())

    df = pd.DataFrame(all_features)
    df.to_csv(os.path.join(OUTPUT_DIR, f"{category}.csv"), index=False)

print("Feature extraction completed!")
