# EDA-based Exercise Detector

Detects physical activity (exercise) vs. rest from a wrist-worn EDA (galvanic skin response) sensor using a neural network.

Built on the [PPG Field Study dataset](https://ubicomp.eti.uni-siegen.de/home/datasets/sensors19/) by Reiss et al. (2019), which contains data from 15 subjects performing everyday activities over ~2.5 hours each.

---

## How it works

The EDA signal (4 Hz) is split into 8-second sliding windows. From each window, 10 statistical features are extracted:

| Feature | What it captures |
|---|---|
| Mean, min, max, range | Tonic EDA level (slow baseline) |
| Std, slope, energy | Phasic activity (fast spikes) |
| Skewness, kurtosis, ZCR | Signal shape |

These features are fed into a small MLP (neural network) that classifies each window as **exercise** or **rest**.

Activities labelled as exercise: stairs (2), table soccer (3), cycling (4), walking (7).  
Activities labelled as rest: sitting (1), driving (5), lunch (6), working (8).  
Transition periods (0) are excluded.

Validation uses **Leave-One-Subject-Out (LOSO)**: the model is trained on 14 subjects and tested on the remaining one, repeated for all 15 subjects. This simulates how well the model generalises to a completely new person.

---

## Project structure

```
eda-analysis/
├── main.py          # Main script
├── .env             # Data path (not committed to git)
├── .gitignore
├── requirements.txt
└── results.png      # Output plot (generated after running)
```

---

## Setup

**1. Clone or download the project**

**2. Create and activate a virtual environment**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**
```powershell
pip install torch numpy scikit-learn matplotlib seaborn python-dotenv
```

**4. Download the dataset**

Download the PPG Field Study dataset and extract it so the folder structure looks like:
```
PPG_FieldStudy/
├── S1/
│   └── S1.pkl
├── S2/
│   └── S2.pkl
...
└── S15/
    └── S15.pkl
```

**5. Set the data path**

Create a `.env` file in the project root:
```dotenv
DATA_DIR=C:\path\to\PPG_FieldStudy
```

**6. Run**
```powershell
python main.py
```

---

## Output

The script prints per-subject accuracy and F1 score, a final summary, and saves `results.png` with two plots:

- **Confusion matrix** — how often the model was right/wrong overall
- **Accuracy per subject** — shows which subjects were harder to classify

Expected runtime: **~5 minutes** on CPU.

---

## Results interpretation

| Metric | Meaning |
|---|---|
| Accuracy | % of windows classified correctly |
| F1 (macro) | Balanced score accounting for class imbalance |
| Precision | Of all "exercise" predictions, how many were correct |
| Recall | Of all actual exercise windows, how many were found |

Rule of thumb: below 0.70 = poor, 0.70–0.85 = good, above 0.85 = excellent.

---

## Reference

Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019).  
*Deep PPG: Large-scale Heart Rate Estimation with Convolutional Neural Networks.*  
MDPI Sensors, 19(14).
