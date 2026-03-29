"""
EDA-based Exercise Detector
============================
Detects physical activity (exercise) vs. rest from wrist EDA signal
(galvanic skin response) using a CNN+LSTM neural network.

Dataset: PPG Field Study (Reiss et al. 2019)
Model:   CNN + LSTM (PyTorch)
Validation: Leave-One-Subject-Out (LOSO)

Requirements:
    pip install torch numpy scikit-learn matplotlib seaborn
"""

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

# ─────────────────────────────────────────
# SETTINGS — change these as needed
# ─────────────────────────────────────────
DATA_DIR = Path("./data")       # Folder containing S1/, S2/, ... subfolders
N_SUBJECTS = 15                 # Number of subjects
EDA_HZ = 4                      # EDA sampling rate (Hz)
WINDOW_SEC = 8                  # Window length in seconds
SHIFT_SEC = 2                   # Window shift in seconds
WINDOW_SAMPLES = WINDOW_SEC * EDA_HZ   # = 32 samples
SHIFT_SAMPLES = SHIFT_SEC * EDA_HZ    # = 8 samples
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

# Activity IDs that count as "exercise" (positive stress)
SPORT_IDS = {2, 3, 4, 7}   # Stairs, table soccer, cycling, walking
# All others (1,5,6,8) = rest/neutral, ID 0 = transition (excluded)

# ─────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────

def load_subject(subject_id: int):
    """
    Loads one subject's pkl file and returns
    the EDA signal and activity labels.
    """
    path = DATA_DIR / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # EDA from wrist, 4 Hz
    eda = data["signal"]["wrist"]["EDA"].flatten().astype(np.float32)

    # Activity signal, also 4 Hz (same length as EDA)
    activity = data["activity"].flatten().astype(np.int32)

    # Trim to shorter length (safety)
    n = min(len(eda), len(activity))
    return eda[:n], activity[:n]


def make_windows(eda, activity):
    """
    Splits the signal into sliding windows and creates binary labels.
    Windows containing a mix of exercise and rest are discarded,
    as are transition periods (ID 0).
    """
    X, y = [], []
    for start in range(0, len(eda) - WINDOW_SAMPLES + 1, SHIFT_SAMPLES):
        end = start + WINDOW_SAMPLES
        window_eda = eda[start:end]
        window_act = activity[start:end]

        # Skip transition periods (contain ID 0)
        if 0 in window_act:
            continue

        # Check if window is clearly exercise or rest
        unique_ids = set(np.unique(window_act))
        is_exercise = bool(unique_ids & SPORT_IDS)
        is_rest = bool(unique_ids - SPORT_IDS)

        # Skip mixed windows
        if is_exercise and is_rest:
            continue

        label = 1 if is_exercise else 0
        X.append(window_eda)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def normalize(X_train, X_test):
    """Z-score normalization: computed from train set, applied to both."""
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# ─────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────

class EDADataset(Dataset):
    def __init__(self, X, y):
        # CNN+LSTM expects shape (batch, channels, seq_len) -> add channel dim
        self.X = torch.tensor(X).unsqueeze(1)  # (N, 1, 32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# 3. MODEL: CNN + LSTM
# ─────────────────────────────────────────

class CNNLSTMModel(nn.Module):
    """
    CNN learns local features from the EDA signal (spikes, curves),
    LSTM learns temporal dependencies (rising/falling trends).
    A final Dense layer performs binary classification.
    """
    def __init__(self):
        super().__init__()

        # CNN block: two convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # (N, 64, 16)
            nn.Dropout(0.3),
        )

        # LSTM block: learns temporal patterns from CNN features
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),   # 2 classes: exercise / rest
        )

    def forward(self, x):
        # x shape: (batch, 1, 32)
        x = self.cnn(x)              # -> (batch, 64, 16)
        x = x.permute(0, 2, 1)      # -> (batch, 16, 64)  (LSTM expects this)
        x, _ = self.lstm(x)         # -> (batch, 16, 64)
        x = x[:, -1, :]             # Last time step: (batch, 64)
        return self.classifier(x)   # -> (batch, 2)


# ─────────────────────────────────────────
# 4. TRAINING AND EVALUATION
# ─────────────────────────────────────────

def train_and_evaluate(train_X, train_y, test_X, test_y, device):
    """Trains the model and returns predictions on the test set."""

    # Class weights — corrects imbalance (rest >> exercise typically)
    n_neg = (train_y == 0).sum()
    n_pos = (train_y == 1).sum()
    class_weights = torch.tensor([1.0, n_neg / (n_pos + 1e-8)]).to(device)

    train_dataset = EDADataset(train_X, train_y)
    test_dataset = EDADataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = CNNLSTMModel().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_predictions.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_predictions)


# ─────────────────────────────────────────
# 5. LOSO VALIDATION (main program)
# ─────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all subjects
    print("Loading data...")
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

    # LOSO: each round one subject is the test set
    all_labels_combined, all_preds_combined = [], []
    results = []

    for test_sid in range(N_SUBJECTS):
        if all_X[test_sid] is None:
            continue

        print(f"\n--- Testing subject S{test_sid + 1} ---")

        # Combine all other subjects for training
        train_X = np.concatenate([all_X[i] for i in range(N_SUBJECTS)
                                   if i != test_sid and all_X[i] is not None])
        train_y = np.concatenate([all_y[i] for i in range(N_SUBJECTS)
                                   if i != test_sid and all_y[i] is not None])
        test_X = all_X[test_sid]
        test_y = all_y[test_sid]

        # Normalize
        train_X, test_X = normalize(train_X, test_X)

        # Train and evaluate
        labels, preds = train_and_evaluate(train_X, train_y, test_X, test_y, device)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        print(f"  Accuracy: {acc:.3f} | F1 (macro): {f1:.3f}")
        results.append({"sid": test_sid + 1, "acc": acc, "f1": f1})

        all_labels_combined.extend(labels)
        all_preds_combined.extend(preds)

    # ─────────────────────────────────────
    # 6. SUMMARY AND VISUALIZATION
    # ─────────────────────────────────────
    print("\n" + "="*50)
    print("LOSO VALIDATION SUMMARY")
    print("="*50)
    acc_list = [r["acc"] for r in results]
    f1_list = [r["f1"] for r in results]
    print(f"Mean accuracy: {np.mean(acc_list):.3f} +/- {np.std(acc_list):.3f}")
    print(f"Mean F1:       {np.mean(f1_list):.3f} +/- {np.std(f1_list):.3f}")
    print("\nDetailed report:")
    print(classification_report(
        all_labels_combined, all_preds_combined,
        target_names=["Rest", "Exercise"]
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels_combined, all_preds_combined)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Rest", "Exercise"],
                yticklabels=["Rest", "Exercise"])
    ax1.set_title("Confusion Matrix (all subjects)")
    ax1.set_ylabel("True label")
    ax1.set_xlabel("Predicted label")

    # Accuracy per subject
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