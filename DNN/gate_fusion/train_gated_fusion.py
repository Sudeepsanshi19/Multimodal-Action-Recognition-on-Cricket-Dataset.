import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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
# LOAD FEATURES
# --------------------------------------
def load_feature(path):
    return np.load(path)

# --------------------------------------
# BUILD DATASET
# --------------------------------------
def build_dataset():
    X_video, X_audio, y = [], [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        vdir = os.path.join(VIDEO_DIR, class_name)
        adir = os.path.join(AUDIO_DIR, class_name)

        if not os.path.exists(vdir):
            continue

        for file in os.listdir(vdir):
            if not file.endswith(".npy"):
                continue

            base = file.replace(".npy", "")
            vp = os.path.join(vdir, file)
            ap = os.path.join(adir, base + ".npy")

            if not os.path.exists(ap):
                continue

            v = load_feature(vp)
            a = load_feature(ap)

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

    Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr, y_te = train_test_split(
        X_video, X_audio, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    v_scaler = StandardScaler()
    a_scaler = StandardScaler()

    Xv_tr = v_scaler.fit_transform(Xv_tr)
    Xv_te = v_scaler.transform(Xv_te)

    Xa_tr = a_scaler.fit_transform(Xa_tr)
    Xa_te = a_scaler.transform(Xa_te)

    # -----------------------------
    # BASE MODELS
    # -----------------------------
    video_clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        max_iter=150,
        activation="relu",
        solver="adam"
    )

    audio_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=150,
        activation="relu",
        solver="adam"
    )

    video_clf.fit(Xv_tr, y_tr)
    audio_clf.fit(Xa_tr, y_tr)

    # -----------------------------
    # PROBABILITIES
    # -----------------------------
    v_tr_p = video_clf.predict_proba(Xv_tr)
    a_tr_p = audio_clf.predict_proba(Xa_tr)

    v_te_p = video_clf.predict_proba(Xv_te)
    a_te_p = audio_clf.predict_proba(Xa_te)

    # -----------------------------
    # GATE NETWORK
    # -----------------------------
    gate_X_train = np.hstack([v_tr_p, a_tr_p])
    gate_X_test = np.hstack([v_te_p, a_te_p])

    gate = LogisticRegression(
        max_iter=1000,
        multi_class="ovr"
    )

    gate.fit(gate_X_train, y_tr)

    gate_probs = gate.predict_proba(gate_X_test)

    # -----------------------------
    # GATED FUSION
    # -----------------------------
    fused_probs = gate_probs * v_te_p + (1 - gate_probs) * a_te_p
    y_pred = np.argmax(fused_probs, axis=1)

    acc = accuracy_score(y_te, y_pred)

    print("\n=============================")
    print("   GATED FUSION EVALUATION")
    print("=============================")
    print("Accuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(y_te, y_pred, target_names=CLASS_NAMES))

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_te, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Gated Fusion Confusion Matrix")
    plt.tight_layout()
    plt.savefig("gated_fusion_confusion_matrix.png")
    plt.close()

    print("✔ Confusion matrix saved")

    # -----------------------------
    # SAVE MODELS
    # -----------------------------
    joblib.dump(video_clf, "video_classifier.pkl")
    joblib.dump(audio_clf, "audio_classifier.pkl")
    joblib.dump(gate, "gate_model.pkl")
    joblib.dump(v_scaler, "video_scaler.pkl")
    joblib.dump(a_scaler, "audio_scaler.pkl")

    print("✔ All models saved")

# --------------------------------------
if __name__ == "__main__":
    train()
