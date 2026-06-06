import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------
# CONFIG
# --------------------------------------
VIDEO_DIR = "resnet_features"
AUDIO_DIR = "audio_features"

CLASS_NAMES = ["Four_runs", "Six_runs", "Wicket", "Others"]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------
def load_feature(path):
    return np.load(path)

# --------------------------------------
def attention_fusion(video_feat, audio_feat):
    """
    Feature-level attention based on signal strength
    """
    if audio_feat is None:
        return video_feat

    ev = np.linalg.norm(video_feat)
    ea = np.linalg.norm(audio_feat)

    alpha_v = ev / (ev + ea + 1e-8)
    alpha_a = ea / (ev + ea + 1e-8)

    return np.hstack([alpha_v * video_feat, alpha_a * audio_feat])

# --------------------------------------
def build_dataset():
    X, y = [], []

    for label_idx, cname in enumerate(CLASS_NAMES):
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

            v = load_feature(vp)
            a = load_feature(ap) if os.path.exists(ap) else None

            if v.ndim != 1:
                continue

            fused = attention_fusion(v, a)

            X.append(fused)
            y.append(label_idx)

    return np.array(X), np.array(y)

# --------------------------------------
def train():
    X, y = build_dataset()

    if len(y) == 0:
        raise RuntimeError("Dataset empty. Fix your feature extraction.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # ----------------------------------
    # ONE SINGLE MODEL
    # ----------------------------------
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        max_iter=300,
        early_stopping=True
    )

    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print("\n==============================")
    print(" SINGLE ATTENTION MODEL RESULT")
    print("==============================")
    print("Accuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(y_te, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Single Attention Fusion Model")
    plt.tight_layout()
    plt.savefig("final_confusion_matrix.png")
    plt.close()

    # ----------------------------------
    # SAVE ONLY ONE MODEL
    # ----------------------------------
    joblib.dump(
        {"model": clf, "scaler": scaler},
        "final_action_classifier.pkl"
    )

    print("✔ Saved ONE final model: final_action_classifier.pkl")

# --------------------------------------
if __name__ == "__main__":
    train()
