from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODELS_DIR / "linear_regression_toxic_pipeline.joblib"
THRESHOLD = 0.5


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"\S+@\S+", " <email> ", text)
    text = re.sub(r"@[\w_]+", " <user> ", text)
    text = re.sub(r"\d{8,}", " <number> ", text)
    text = re.sub(r"([!?.]){2,}", r" \1 ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"ViCTSD_{split}.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Comment": "comment",
            "Toxicity": "label",
            "Title": "title",
            "Topic": "topic",
        }
    )
    df["comment"] = df["comment"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            ("regressor", LinearRegression()),
        ]
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[dict, np.ndarray]:
    y_score = np.clip(y_score, 0.0, 1.0)
    y_pred = (y_score >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_toxic": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_toxic": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_toxic": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["clean", "toxic"],
            output_dict=True,
            zero_division=0,
        ),
        "threshold": THRESHOLD,
        "model_name": "Linear Regression",
    }
    return metrics, y_pred


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["clean", "toxic"])
    ax.set_yticks([0, 1], labels=["clean", "toxic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Linear Regression Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

    pipeline = build_pipeline()
    pipeline.fit(train_valid_df["comment"], train_valid_df["label"])
    joblib.dump(pipeline, MODEL_PATH)

    y_true = test_df["label"].to_numpy()
    y_score = pipeline.predict(test_df["comment"])
    y_score = np.clip(y_score, 0.0, 1.0)
    metrics, y_pred = evaluate_predictions(y_true, y_score)

    with open(REPORTS_DIR / "linear_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = test_df[["comment", "title", "topic", "label"]].copy()
    pred_df["true_label"] = pred_df["label"]
    pred_df["pred_label"] = y_pred
    pred_df["prob_toxic"] = y_score
    pred_df.drop(columns=["label"], inplace=True)
    pred_df.to_csv(REPORTS_DIR / "linear_regression_predictions.csv", index=False)

    error_df = pred_df[pred_df["true_label"] != pred_df["pred_label"]].copy()
    error_df.to_csv(REPORTS_DIR / "linear_regression_errors.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, REPORTS_DIR / "linear_regression_confusion_matrix.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
