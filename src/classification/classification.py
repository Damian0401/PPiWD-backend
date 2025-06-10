from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

MODELS_PATH = Path.cwd() / "classification" / "models"
CLASSIFIER_PATH = MODELS_PATH / "classifier.pkl"
SUPERVISED_CLASSIFIER_PATH = MODELS_PATH / "supervised_classifier.pkl"

SCALLER_PATH = MODELS_PATH / "scaler.pkl"
SUPERVISED_SCALLER_PATH = MODELS_PATH / "supervised_scaler.pkl"

SUPERVISED_COLUMNS_PATH = MODELS_PATH / "columns.pkl"

AUTOENCODER_PATH = MODELS_PATH / "autoencoder.pkl"

loaded_columns = joblib.load(SUPERVISED_COLUMNS_PATH)

# model = joblib.load(CLASSIFIER_PATH)
model = joblib.load(SUPERVISED_CLASSIFIER_PATH)
autoencoder = joblib.load(AUTOENCODER_PATH)
# scaler = joblib.load(SCALLER_PATH)
scaler = joblib.load(SUPERVISED_SCALLER_PATH)


def normalize_column(col):
    return (
        col.lower()
        .replace("accel", "acc")
        .replace("_", "-")
        .replace("correlation()-", "correlation()-")
        .replace("()", "()")
    )


def predict_labels(data):
    X = data.values.astype(np.float32)
    X[np.isnan(X)] = 0
    # scaler to import
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        encoded = autoencoder.encoder(X_tensor)

        logits = model(encoded)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # predicted_labels = label_encoder.inverse_transform(preds)

    return preds


def data_predict(data):
    # df_filtered = df[df_filtered_names]
    normalized_data_cols = {normalize_column(col): col for col in data.columns}
    common_cols = [
        col for col in loaded_columns if normalize_column(col) in normalized_data_cols
    ]

    X_new = data[[normalized_data_cols[normalize_column(col)] for col in common_cols]]
    predicted = predict_labels(X_new)

    results = []
    timestamps = data["timestamp"].values if "timestamp" in data.columns else None
    if timestamps is not None:
        for ts, pred in zip(timestamps, predicted):
            date = pd.to_datetime(ts)
            results.append({"timestamp": date.isoformat(), "prediction": int(pred)})

    return results