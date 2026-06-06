import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_DIR = "resnet_features"
AUDIO_DIR = "audio_features"
CLASS_NAMES = ["Four_runs", "Six_runs", "Wicket", "Others"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3

VIDEO_DIM = 512
AUDIO_DIM = 64
HIDDEN_DIM = 256

# -----------------------------
# DATASET
# -----------------------------
class MultimodalDataset(Dataset):
    def __init__(self, video_paths, audio_paths, labels):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        v = np.load(self.video_paths[idx])   # (512,)
        a = np.load(self.audio_paths[idx])   # (64,)

        return (
            torch.tensor(v, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -----------------------------
# BUILD PATHS
# -----------------------------
def build_paths():
    vpaths, apaths, labels = [], [], []

    for lbl, cname in enumerate(CLASS_NAMES):
        vdir = os.path.join(VIDEO_DIR, cname)
        adir = os.path.join(AUDIO_DIR, cname)

        if not os.path.exists(vdir):
            continue

        for f in os.listdir(vdir):
            if not f.endswith(".npy"):
                continue

            base = f.replace(".npy", "")
            vp = os.path.join(vdir, f)
            ap = os.path.join(adir, base + ".npy")

            if not os.path.exists(ap):
                continue

            vpaths.append(vp)
            apaths.append(ap)
            labels.append(lbl)

    return vpaths, apaths, labels

# -----------------------------
# BIMODAL ATTENTION MODULE
# -----------------------------
class BimodalAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.v_to_a = nn.Linear(VIDEO_DIM, AUDIO_DIM)
        self.a_to_v = nn.Linear(AUDIO_DIM, VIDEO_DIM)

    def forward(self, v, a):
        # video attends to audio
        attn_a = torch.sigmoid(self.v_to_a(v))
        a_hat = attn_a * a

        # audio attends to video
        attn_v = torch.sigmoid(self.a_to_v(a))
        v_hat = attn_v * v

        return v_hat, a_hat

# -----------------------------
# MODEL
# -----------------------------
class BimodalAttentionDNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = BimodalAttention()

        self.classifier = nn.Sequential(
            nn.Linear(VIDEO_DIM + AUDIO_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, len(CLASS_NAMES))
        )

    def forward(self, v, a):
        v_hat, a_hat = self.attn(v, a)
        fused = torch.cat([v_hat, a_hat], dim=1)
        return self.classifier(fused)

# -----------------------------
# TRAIN & EVAL
# -----------------------------
def train():
    vpaths, apaths, labels = build_paths()

    vtr, vte, atr, ate, ytr, yte = train_test_split(
        vpaths, apaths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_ds = MultimodalDataset(vtr, atr, ytr)
    test_ds  = MultimodalDataset(vte, ate, yte)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BimodalAttentionDNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # -------- TRAIN --------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for v, a, y in train_loader:
            v, a, y = v.to(DEVICE), a.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(v, a)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # -------- EVAL --------
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for v, a, y in test_loader:
            v, a = v.to(DEVICE), a.to(DEVICE)
            out = model(v, a)
            p = torch.argmax(out, dim=1).cpu().numpy()

            preds.extend(p)
            trues.extend(y.numpy())

    acc = accuracy_score(trues, preds)
    print("\nAccuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(trues, preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Bimodal Attention Fusion (DNN)")
    plt.tight_layout()
    plt.savefig("bimodal_attention_cm.png")
    plt.close()

    torch.save(model.state_dict(), "bimodal_attention_dnn.pth")
    print("✔ Model saved")

# -----------------------------
if __name__ == "__main__":
    train()
