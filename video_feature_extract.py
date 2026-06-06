import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

DATASET_DIR = "Dataset"
OUTPUT_DIR = "resnet_features"
FRAME_SIZE = 224
SAMPLE_EVERY = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD RESNET BACKBONE
# -------------------------------
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % SAMPLE_EVERY == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        idx += 1

    cap.release()
    return frames

def extract_resnet_features(frames):
    feats = []

    for f in frames:
        img = Image.fromarray(f)
        tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            vec = resnet(tensor).squeeze().numpy()
            feats.append(vec)

    if len(feats) == 0:
        return None

    return np.mean(feats, axis=0)

def process_dataset():
    classes = os.listdir(DATASET_DIR)

    for cls in classes:
        cls_path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        # NEW: loop inside train/test/val
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(cls_path, split)
            if not os.path.isdir(split_dir):
                continue

            out_dir = os.path.join(OUTPUT_DIR, cls, split)
            os.makedirs(out_dir, exist_ok=True)

            videos = [v for v in os.listdir(split_dir)
                      if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

            print(f"\n>>> Class: {cls} | Split: {split} | {len(videos)} videos")

            for vid in tqdm(videos):
                vid_path = os.path.join(split_dir, vid)
                base = os.path.splitext(vid)[0]

                frames = load_frames(vid_path)
                if frames is None:
                    continue

                feat = extract_resnet_features(frames)
                if feat is None:
                    continue

                np.save(os.path.join(out_dir, f"{base}_resnet.npy"), feat)

    print("\n✔ DONE — ResNet features extracted successfully.")

process_dataset()
