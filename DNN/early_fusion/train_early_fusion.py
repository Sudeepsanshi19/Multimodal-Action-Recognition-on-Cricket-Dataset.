import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------
# CONFIG
# --------------------------------------
VIDEO_DIR = "resnet_features"
AUDIO_DIR = "audio_features"

CLASS_NAMES = ["Four_runs", "Six_runs", "Wicket", "Others"]


# --------------------------------------
# LOAD FEATURES
# --------------------------------------
def load_video_feature(path):
    return np.load(path)     # shape: [512]


def load_audio_feature(path):
    return np.load(path)     # shape: [64]


# --------------------------------------
# FUSE FEATURES
# --------------------------------------
def fuse_features(video_path, audio_path):
    v = load_video_feature(video_path)    # (512,)
    a = load_audio_feature(audio_path)    # (64,)

    if v.ndim != 1 or a.ndim != 1:
        print("❌ ERROR: Feature shapes not flattened:",
              video_path, v.shape, audio_path, a.shape)
        return None

    return np.concatenate([v, a])        # (576,)


# --------------------------------------
# BUILD DATASET
# --------------------------------------
def build_dataset():
    X, y = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        video_class_dir = os.path.join(VIDEO_DIR, class_name)

        if not os.path.exists(video_class_dir):
            print(f"Skipping missing class folder: {video_class_dir}")
            continue

        for file in os.listdir(video_class_dir):
            if not file.endswith(".npy"):
                continue

            base = file.replace(".npy", "")  # e.g. Four_runs_0001

            video_path = os.path.join(VIDEO_DIR, class_name, file)
            audio_path = os.path.join(AUDIO_DIR, class_name, base + ".npy")

            if not os.path.exists(audio_path):
                print(f"Missing audio feature for: {base}, skipping.")
                continue

            fused = fuse_features(video_path, audio_path)
            if fused is None:
                continue

            X.append(fused)
            y.append(label_idx)

        print(f">>> Loaded class: {class_name}")

    X = np.array(X)
    y = np.array(y)

    print("\nDataset built.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y


# --------------------------------------
# TRAINING PIPELINE
# --------------------------------------
def train():
    X, y = build_dataset()

    if len(X) == 0:
        print("\n❌ Dataset is empty. Fix your paths or features.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        batch_size=32,
        max_iter=100,
        verbose=True
    )

    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=============================")
    print("      EVALUATION")
    print("=============================")
    print("Accuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\n✔ Confusion matrix saved as confusion_matrix.png")

    # Save model
    joblib.dump(clf, "multimodal_classifier.pkl")
    print("✔ Model saved as multimodal_classifier.pkl")


# --------------------------------------
if __name__ == "__main__":
    train()
