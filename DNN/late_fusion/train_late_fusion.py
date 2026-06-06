import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --------------------------------------
# CONFIG
# --------------------------------------
VIDEO_DIR = "resnet_features"
AUDIO_DIR = "audio_features"

CLASS_NAMES = ["Four_runs", "Six_runs", "Wicket", "Others"]
NUM_CLASSES = len(CLASS_NAMES)

# Fusion weights (tune these)
VIDEO_WEIGHT = 0.7
AUDIO_WEIGHT = 0.3


# --------------------------------------
# LOAD FEATURES
# --------------------------------------
def load_feature(path):
    return np.load(path)


# --------------------------------------
# BUILD DATASET (SEPARATE)
# --------------------------------------
def build_dataset():
    X_video, X_audio, y = [], [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        video_class_dir = os.path.join(VIDEO_DIR, class_name)

        if not os.path.exists(video_class_dir):
            continue

        for file in os.listdir(video_class_dir):
            if not file.endswith(".npy"):
                continue

            base = file.replace(".npy", "")
            video_path = os.path.join(VIDEO_DIR, class_name, file)
            audio_path = os.path.join(AUDIO_DIR, class_name, base + ".npy")

            if not os.path.exists(audio_path):
                continue

            v = load_feature(video_path)
            a = load_feature(audio_path)

            if v.ndim != 1 or a.ndim != 1:
                continue

            X_video.append(v)
            X_audio.append(a)
            y.append(label_idx)

    return np.array(X_video), np.array(X_audio), np.array(y)


# --------------------------------------
# TRAINING PIPELINE
# --------------------------------------
def train():
    X_video, X_audio, y = build_dataset()

    if len(y) == 0:
        print("❌ Empty dataset. Fix your features.")
        return

    Xv_train, Xv_test, Xa_train, Xa_test, y_train, y_test = train_test_split(
        X_video, X_audio, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # NORMALIZATION (CRITICAL)
    # -----------------------------
    video_scaler = StandardScaler()
    audio_scaler = StandardScaler()

    Xv_train = video_scaler.fit_transform(Xv_train)
    Xv_test = video_scaler.transform(Xv_test)

    Xa_train = audio_scaler.fit_transform(Xa_train)
    Xa_test = audio_scaler.transform(Xa_test)

    # -----------------------------
    # MODELS
    # -----------------------------
    video_clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        max_iter=150,
        activation="relu",
        solver="adam",
        verbose=True
    )

    audio_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=150,
        activation="relu",
        solver="adam",
        verbose=True
    )

    video_clf.fit(Xv_train, y_train)
    audio_clf.fit(Xa_train, y_train)

    # -----------------------------
    # LATE FUSION (PROBABILITY LEVEL)
    # -----------------------------
    video_probs = video_clf.predict_proba(Xv_test)
    audio_probs = audio_clf.predict_proba(Xa_test)

    fused_probs = (VIDEO_WEIGHT * video_probs) + (AUDIO_WEIGHT * audio_probs)
    y_pred = np.argmax(fused_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)

    print("\n=============================")
    print("   LATE FUSION EVALUATION")
    print("=============================")
    print("Accuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Late Fusion Confusion Matrix")
    plt.tight_layout()
    plt.savefig("late_fusion_confusion_matrix.png")
    plt.close()

    print("✔ Confusion matrix saved")

    # -----------------------------
    # SAVE EVERYTHING
    # -----------------------------
    joblib.dump(video_clf, "video_classifier.pkl")
    joblib.dump(audio_clf, "audio_classifier.pkl")
    joblib.dump(video_scaler, "video_scaler.pkl")
    joblib.dump(audio_scaler, "audio_scaler.pkl")

    print("✔ Models and scalers saved")


# --------------------------------------
if __name__ == "__main__":
    train()
