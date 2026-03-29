"""
EDA-based Exercise Detector
============================
Detects physical activity (exercise) vs. rest from wrist EDA signal
using handcrafted features + MLP.

Dataset: PPG Field Study (Reiss et al. 2019)
Model:   Feature engineering + MLP (PyTorch)
Validation: Leave-One-Subject-Out (LOSO)

Requirements:
    pip install torch numpy scikit-learn matplotlib seaborn python-dotenv
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
N_SUBJECTS = 15
EDA_HZ = 4
WINDOW_SEC = 8
SHIFT_SEC = 2
WINDOW_SAMPLES = WINDOW_SEC * EDA_HZ   # = 32 samples
SHIFT_SAMPLES = SHIFT_SEC * EDA_HZ    # = 8 samples
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3

SPORT_IDS = {2, 3, 4, 7}


# ─────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────

def load_subject(subject_id: int):
    path = DATA_DIR / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    eda = data["signal"]["wrist"]["EDA"].flatten().astype(np.float32)
    activity = data["activity"].flatten().astype(np.int32)
    n = min(len(eda), len(activity))
    return eda[:n], activity[:n]


# ─────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 10 statistical features from a single EDA window.
    These capture the key EDA characteristics:
    - Tonic level (slow baseline): mean, min, max, range
    - Phasic activity (fast spikes): std, slope, energy
    - Shape: skewness, kurtosis, zero-crossing rate
    """
    mean    = np.mean(window)
    std     = np.std(window)
    minimum = np.min(window)
    maximum = np.max(window)
    rng     = maximum - minimum

    # Slope: is the EDA rising or falling?
    slope = np.polyfit(np.arange(len(window)), window, 1)[0]

    # Signal energy
    energy = np.sum(window ** 2) / len(window)

    # Skewness (manual)
    skewness = np.mean(((window - mean) / (std + 1e-8)) ** 3)

    # Kurtosis (manual)
    kurtosis = np.mean(((window - mean) / (std + 1e-8)) ** 4)

    # Zero-crossing rate of the first derivative (how often signal changes direction)
    diff = np.diff(window)
    zcr = np.sum(np.diff(np.sign(diff)) != 0) / len(diff)

    return np.array([mean, std, minimum, maximum, rng,
                     slope, energy, skewness, kurtosis, zcr],
                    dtype=np.float32)


def make_windows(eda, activity):
    X, y = [], []
    for start in range(0, len(eda) - WINDOW_SAMPLES + 1, SHIFT_SAMPLES):
        end = start + WINDOW_SAMPLES
        window_eda = eda[start:end]
        window_act = activity[start:end]

        if 0 in window_act:
            continue

        unique_ids = set(np.unique(window_act))
        is_exercise = bool(unique_ids & SPORT_IDS)
        is_rest = bool(unique_ids - SPORT_IDS)

        if is_exercise and is_rest:
            continue

        features = extract_features(window_eda)
        label = 1 if is_exercise else 0
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# ─────────────────────────────────────────
# 3. PYTORCH DATASET
# ─────────────────────────────────────────

class EDADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# 4. MODEL: MLP
# ─────────────────────────────────────────

class MLPModel(nn.Module):
    """
    Simple 3-layer MLP. Works well with handcrafted features
    and trains much faster than CNN+LSTM.
    Input: 10 EDA features
    Output: 2 classes (rest / exercise)
    """
    def __init__(self, input_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# 5. TRAINING AND EVALUATION
# ─────────────────────────────────────────

def train_and_evaluate(train_X, train_y, test_X, test_y, device):
    n_neg = (train_y == 0).sum()
    n_pos = (train_y == 1).sum()
    class_weights = torch.tensor(
        [1.0, n_neg / (n_pos + 1e-8)], dtype=torch.float32
    ).to(device)

    train_loader = DataLoader(EDADataset(train_X, train_y),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(EDADataset(test_X, test_y),
                              batch_size=BATCH_SIZE)

    model = MLPModel(input_size=train_X.shape[1]).to(device)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────
# 6. LOSO VALIDATION
# ─────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data and extracting features...")
    all_X, all_y = [], []
    for sid in range(1, N_SUBJECTS + 1):
        try:
            eda, activity = load_subject(sid)
            X, y = make_windows(eda, activity)
            all_X.append(X)
            all_y.append(y)
            print(f"  S{sid}: {len(y)} windows "
                  f"(exercise={y.sum()}, rest={(y==0).sum()})")
        except FileNotFoundError:
            print(f"  S{sid}: file not found, skipping")
            all_X.append(None)
            all_y.append(None)

    all_labels_combined, all_preds_combined = [], []
    results = []

    for test_sid in range(N_SUBJECTS):
        if all_X[test_sid] is None:
            continue

        print(f"\n--- Testing subject S{test_sid + 1} ---")

        train_X = np.concatenate([all_X[i] for i in range(N_SUBJECTS)
                                   if i != test_sid and all_X[i] is not None])
        train_y = np.concatenate([all_y[i] for i in range(N_SUBJECTS)
                                   if i != test_sid and all_y[i] is not None])
        test_X = all_X[test_sid]
        test_y = all_y[test_sid]

        train_X, test_X = normalize(train_X, test_X)

        labels, preds = train_and_evaluate(train_X, train_y, test_X, test_y, device)

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="macro")
        print(f"  Accuracy: {acc:.3f} | F1 (macro): {f1:.3f}")
        results.append({"sid": test_sid + 1, "acc": acc, "f1": f1})

        all_labels_combined.extend(labels)
        all_preds_combined.extend(preds)

    # ─────────────────────────────────────
    # 7. SUMMARY AND VISUALIZATION
    # ─────────────────────────────────────
    print("\n" + "="*50)
    print("LOSO VALIDATION SUMMARY")
    print("="*50)
    acc_list = [r["acc"] for r in results]
    f1_list  = [r["f1"]  for r in results]
    print(f"Mean accuracy: {np.mean(acc_list):.3f} +/- {np.std(acc_list):.3f}")
    print(f"Mean F1:       {np.mean(f1_list):.3f} +/- {np.std(f1_list):.3f}")
    print("\nDetailed report:")
    print(classification_report(
        all_labels_combined, all_preds_combined,
        target_names=["Rest", "Exercise"]
    ))

    cm = confusion_matrix(all_labels_combined, all_preds_combined)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Rest", "Exercise"],
                yticklabels=["Rest", "Exercise"])
    ax1.set_title("Confusion Matrix (all subjects)")
    ax1.set_ylabel("True label")
    ax1.set_xlabel("Predicted label")

    sid_list = [r["sid"] for r in results]
    ax2.bar([f"S{s}" for s in sid_list], acc_list, color="steelblue")
    ax2.axhline(np.mean(acc_list), color="red", linestyle="--",
                label=f"Mean: {np.mean(acc_list):.3f}")
    ax2.set_title("Accuracy per Subject")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Subject")
    ax2.set_ylim(0, 1)
    ax2.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nPlot saved: results.png")
    plt.show()


if __name__ == "__main__":
    main()