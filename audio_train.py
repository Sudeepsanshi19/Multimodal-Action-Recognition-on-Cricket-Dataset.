import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

AUDIO_DIR = "audio_features"
CLASS_NAMES = ["Four_runs", "Six_runs", "Wicket", "Others"]

def build_audio_dataset():
    X, y = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(AUDIO_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Missing folder: {class_dir}")
            continue

        for file in os.listdir(class_dir):
            if not file.endswith(".npy"):
                continue

            path = os.path.join(class_dir, file)
            feat = np.load(path)       # shape: (64,)

            if feat.ndim != 1:
                print("❌ Bad audio feature shape:", file, feat.shape)
                continue

            X.append(feat)
            y.append(label_idx)

        print(f">>> Loaded class: {class_name}")

    X = np.array(X)
    y = np.array(y)
    print("\nDataset built.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y


def train():
    X, y = build_audio_dataset()

    if len(X) == 0:
        print("❌ Dataset empty! Check feature paths.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        batch_size=32,
        max_iter=30,
        verbose=True
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n========= AUDIO-ONLY RESULTS =========")
    print("Accuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Audio-Only Confusion Matrix")
    plt.tight_layout()
    plt.savefig("audio_confusion_matrix.png")
    plt.close()

    print("✔ Saved confusion matrix: audio_confusion_matrix.png")

    joblib.dump(clf, "audio_only_classifier.pkl")
    print("✔ Saved model: audio_only_classifier.pkl")


if __name__ == "__main__":
    train()
