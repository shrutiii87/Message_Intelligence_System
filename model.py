"""
model.py - training, evaluation and prediction logic for the Message Intelligence System.

This file mirrors the notebook pipeline:
  * 9 input features (the ID, raw text, timestamp and the keyword / special-char
    columns are NOT used as inputs)
  * median imputation + StandardScaler, fitted on the TRAINING split only
  * 80/20 stratified split with random_state=42
  * KNN (K=5), SVM (Linear), SVM (RBF) and Gaussian Naive Bayes

app.py imports everything it needs from here, so the notebook results and the
live app always come from the same code.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET = "spam_label"
FEATURES = [
    "message_length",
    "word_count",
    "num_urls",
    "num_digits",
    "sender_activity_score",
    "sender_account_age_days",
    "messages_sent_last_24h",
    "hour_of_day",
    "day_of_week",
]
FEATURE_LABELS = {
    "message_length": "Message length",
    "word_count": "Word count",
    "num_urls": "URLs",
    "num_digits": "Digits",
    "sender_activity_score": "Sender activity score",
    "sender_account_age_days": "Sender account age (days)",
    "messages_sent_last_24h": "Messages sent in last 24h",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_NAMES = ["SVM (Linear)", "SVM (RBF)", "KNN (K=5)", "Naive Bayes"]
RECOMMENDED_MODEL = "SVM (Linear)"  # deployment recommendation from the final report

LABEL_NAMES = {0: "Legitimate (ham)", 1: "Spam"}

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def find_dataset_path() -> str:
    """Locate the CSV next to this file (the project keeps the '.csv.csv' name)."""
    env_path = os.environ.get("MESSAGE_DATASET_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.join(BASE_DIR, "Message_Intelligence_Dataset.csv.csv"),
        os.path.join(BASE_DIR, "Message_Intelligence_Dataset.csv"),
    ]
    candidates += sorted(glob.glob(os.path.join(BASE_DIR, "Message_Intelligence_Dataset*.csv*")))
    candidates += sorted(glob.glob(os.path.join(BASE_DIR, "*.csv")))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Dataset not found. Put 'Message_Intelligence_Dataset.csv.csv' in the same "
        f"folder as model.py ({BASE_DIR})."
    )


def load_data(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or find_dataset_path())
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


# --------------------------------------------------------------------------- #
# Text -> features
# --------------------------------------------------------------------------- #
def extract_text_features(text: str) -> dict:
    """Compute the four text-based features exactly like the dataset does.

    num_urls is capped at 1 because the training data only contains 0 or 1.
    """
    text = text or ""
    return {
        "message_length": len(text),
        "word_count": len(text.split()),
        "num_urls": min(len(_URL_RE.findall(text)), 1),
        "num_digits": sum(ch.isdigit() for ch in text),
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _pipeline(classifier) -> Pipeline:
    # Imputer and scaler live inside the pipeline, so they are fitted on the
    # training split only and re-used unchanged at prediction time (no leakage).
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    # Plain SVC models (no deprecated probability=True) so labels and the
    # linear weights are exactly what the notebook produces.
    return {
        "SVM (Linear)": _pipeline(SVC(kernel="linear")),
        "SVM (RBF)": _pipeline(SVC(kernel="rbf")),
        "KNN (K=5)": _pipeline(KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": _pipeline(GaussianNB()),
    }


def build_calibrated_svms() -> dict[str, Pipeline]:
    # SVMs have no native probabilities. These sigmoid-calibrated copies are
    # only used to turn the SVM score into a spam probability for the app.
    return {
        "SVM (Linear)": _pipeline(
            CalibratedClassifierCV(SVC(kernel="linear"), method="sigmoid", cv=5, ensemble=False)
        ),
        "SVM (RBF)": _pipeline(
            CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=5, ensemble=False)
        ),
    }


@dataclass
class TrainedSystem:
    models: dict
    calibrated: dict
    metrics: pd.DataFrame
    confusion: dict
    n_train: int
    n_test: int
    n_total: int
    spam_rate: float
    feature_ranges: pd.DataFrame
    examples: pd.DataFrame = field(default_factory=pd.DataFrame)


def _pick_examples(df: pd.DataFrame, per_class: int = 4) -> pd.DataFrame:
    """A few real messages (with their sender data) for the app's sample picker."""
    parts = []
    for label in (1, 0):
        subset = df[df[TARGET] == label].dropna(subset=FEATURES)
        subset = subset.drop_duplicates(subset="message_text")
        parts.append(subset.sample(n=min(per_class, len(subset)), random_state=RANDOM_STATE))
    return pd.concat(parts).reset_index(drop=True)[["message_text"] + FEATURES + [TARGET]]


def train_system(path: str | None = None) -> TrainedSystem:
    """Train all four models and evaluate them on the held-out 20% test split."""
    df = load_data(path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models()
    rows, confusion = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        }
        confusion[name] = confusion_matrix(y_test, pred, labels=[0, 1])

    calibrated = build_calibrated_svms()
    for calibrated_model in calibrated.values():
        calibrated_model.fit(X_train, y_train)

    ranges = df[FEATURES].describe().T[["min", "max", "50%"]].rename(columns={"50%": "median"})

    return TrainedSystem(
        models=models,
        calibrated=calibrated,
        metrics=pd.DataFrame(rows).T,
        confusion=confusion,
        n_train=len(X_train),
        n_test=len(X_test),
        n_total=len(df),
        spam_rate=float(y.mean()),
        feature_ranges=ranges,
        examples=_pick_examples(df),
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _row(features: dict) -> pd.DataFrame:
    """One-row DataFrame in the exact training column order (NaN allowed)."""
    return pd.DataFrame([{name: features.get(name, np.nan) for name in FEATURES}], columns=FEATURES).astype(float)


def predict_message(system: TrainedSystem, model_name: str, features: dict) -> dict:
    """Predict spam / ham for one message described by the 9 features."""
    if model_name not in system.models:
        raise KeyError(f"Unknown model '{model_name}'. Choose from {list(system.models)}")

    model = system.models[model_name]
    row = _row(features)

    label = int(model.predict(row)[0])
    prob_model = system.calibrated.get(model_name, model)
    spam_prob = float(prob_model.predict_proba(row)[0][1])

    return {
        "label": label,
        "label_name": LABEL_NAMES[label],
        "spam_probability": spam_prob,
        "ham_probability": 1.0 - spam_prob,
        # SVM probabilities are Platt-scaled, so they can disagree with the
        # predicted label for messages sitting right on the decision boundary.
        "borderline": (spam_prob >= 0.5) != bool(label),
    }


def explain_linear_svm(system: TrainedSystem, features: dict) -> pd.DataFrame:
    """Per-feature contribution to the Linear SVM score for one message.

    contribution = weight x scaled value. Positive pushes towards spam,
    negative pushes towards legitimate. (Intercept is excluded.)
    """
    pipe = system.models["SVM (Linear)"]
    scaled = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(_row(features)))[0]
    weights = pipe.named_steps["clf"].coef_[0]

    out = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Weight": weights,
            "Scaled value": scaled,
            "Contribution": weights * scaled,
        }
    )
    return out.sort_values("Contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    system = train_system()
    print(f"Rows: {system.n_total} | train: {system.n_train} | test: {system.n_test} | spam rate: {system.spam_rate:.4f}\n")
    print(system.metrics.round(4).to_string())
    print()
    for name, cm in system.confusion.items():
        tn, fp, fn, tp = cm.ravel()
        print(f"{name:13s} TN={tn} FP={fp} FN={fn} TP={tp}")

    sample = {
        **extract_text_features("Congratulations! You have won instant cashback. http://bit.ly/4151"),
        "sender_activity_score": 44.7,
        "sender_account_age_days": 37,
        "messages_sent_last_24h": 11,
        "hour_of_day": 9,
        "day_of_week": 1,
    }
    print("\nSample prediction:", predict_message(system, RECOMMENDED_MODEL, sample))