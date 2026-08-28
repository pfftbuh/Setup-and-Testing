import argparse
import json
import os
import joblib
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import heatmap_feature_extractor as hfe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "suspicion_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")


def predict_session(session_dir):
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    features = hfe.extract_session_features(session_dir)
    if features is None:
        print(f"[Predictor] Session missing heatmap or CSV log: {session_dir}")
        return None

    ordered_values = pd.DataFrame(
        [[features.get(col, 0.0) for col in feature_columns]],
        columns=feature_columns,
    )
    probabilities = model.predict_proba(ordered_values)[0]
    probability_non_cheating = probabilities[0]
    probability_cheating = probabilities[1]

    label = "cheating" if probability_cheating >= 0.5 else "non_cheating"
    confidence = probability_cheating if label == "cheating" else 1 - probability_cheating
    print(
        f"[Predictor] {os.path.basename(session_dir)} -> {label} "
        f"({confidence * 100:.1f}% confidence)"
    )
    print(
        f"[Predictor] Confidence - cheating: {probability_cheating * 100:.1f}%, "
        f"non-cheating: {probability_non_cheating * 100:.1f}%"
    )
    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cheating/non_cheating for a single session.")
    parser.add_argument(
        "session_dir",
        nargs="?",
        help="Optional path to a session directory containing a heatmap PNG and CSV log.",
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    if session_dir is None:
        root = tk.Tk()
        root.withdraw()
        session_dir = filedialog.askdirectory(
            title="Select a session folder to predict",
            initialdir=os.path.join(BASE_DIR, "sessions"),
        )
        root.destroy()

    if session_dir:
        predict_session(session_dir)
    else:
        print("[Predictor] No session folder selected.")
