import cv2
import torch
import numpy as np
import joblib
import librosa
from torchvision import models, transforms
from PIL import Image

# -------------------------------
MODEL_PATH = "final_action_classifier.pkl"
CLASS_NAMES = ["Four_runs", "Six_runs", "Wicket", "Others"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load model + scaler
bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
scaler = bundle["scaler"]

# -------------------------------
# Video feature extractor
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_video_features(video_path, sample_every=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames extracted")
    feats = []
    for f in frames:
        img = Image.fromarray(f)
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vec = resnet(x).squeeze().cpu().numpy()
        feats.append(vec)
    return np.mean(feats, axis=0)  # 512-D

# -------------------------------
# Audio feature extractor
def extract_audio_features(video_path, n_mfcc=64):
    try:
        y, sr = librosa.load(video_path, sr=16000)
    except Exception:
        return None
    if y.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # 64-D

# -------------------------------
# Attention fusion
def attention_fusion(video_feat, audio_feat):
    if audio_feat is None:
        return video_feat
    ev = np.linalg.norm(video_feat)
    ea = np.linalg.norm(audio_feat)
    alpha_v = ev / (ev + ea + 1e-8)
    alpha_a = ea / (ev + ea + 1e-8)
    return np.hstack([alpha_v * video_feat, alpha_a * audio_feat])  # 512 + 64 = 576

# -------------------------------
def predict_action(mp4_path):
    v_feat = extract_video_features(mp4_path)
    a_feat = extract_audio_features(mp4_path)
    fused = attention_fusion(v_feat, a_feat)
    fused = scaler.transform(fused.reshape(1, -1))
    probs = clf.predict_proba(fused)[0]
    idx = np.argmax(probs)
    return {
        "action": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": dict(zip(CLASS_NAMES, probs))
    }

# -------------------------------
if __name__ == "__main__":
    video_path = "Four_runs_0018.mp4"
    result = predict_action(video_path)
    print("Action:", result["action"])
    print("Confidence:", result["confidence"])
    print("All probabilities:", result["probabilities"])
