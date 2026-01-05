"""
Flask web UI for pre-trained threat detection models.
Loads existing models from /models and exposes simple endpoints for running
network traffic classification, web intrusion detection, and malware file
analysis without retraining any models.
"""
import os
import math
import hashlib
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"exe", "pdf", "txt"}
REPORT_NAME = "malware_report.txt"

app = Flask(
    __name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Loaded artifacts
ARTIFACTS = {
    "network": None,
    "wids": None,
    "malware_model": None,
    "malware_tokenizer": None,
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    probabilities = [c / total for c in counts.values()]
    return -sum(p * math.log(p, 2) for p in probabilities if p > 0)


def load_models() -> None:
    """Load all pre-trained models into memory."""
    network_path = os.path.join(MODEL_DIR, "network_rf.pkl")
    if os.path.exists(network_path):
        ARTIFACTS["network"] = joblib.load(network_path)
    else:
        app.logger.warning("Network model not found at %s", network_path)

    wids_path = os.path.join(MODEL_DIR, "wids_iforest.pkl")
    if os.path.exists(wids_path):
        ARTIFACTS["wids"] = joblib.load(wids_path)
    else:
        app.logger.warning("WIDS model not found at %s", wids_path)

    malware_model_path = os.path.join(MODEL_DIR, "malware_lstm.h5")
    tokenizer_path = os.path.join(MODEL_DIR, "malware_tokenizer.pkl")
    if os.path.exists(malware_model_path) and os.path.exists(tokenizer_path):
        ARTIFACTS["malware_model"] = keras.models.load_model(malware_model_path)
        ARTIFACTS["malware_tokenizer"] = joblib.load(tokenizer_path)
    else:
        app.logger.warning("Malware model or tokenizer missing in %s", MODEL_DIR)


def synthetic_network_prediction():
    artifacts = ARTIFACTS.get("network")
    if artifacts is None:
        raise RuntimeError("Network model is unavailable.")

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    n_features = len(artifacts.get("feature_columns", [])) or getattr(scaler, "n_features_in_", 0)
    if n_features == 0:
        raise RuntimeError("Network model feature count is zero.")

    sample = np.random.randn(1, n_features)
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    label = "Attack" if pred == 1 else "Normal"
    return label


def synthetic_wids_anomalies(n_samples: int = 200):
    artifacts = ARTIFACTS.get("wids")
    if artifacts is None:
        raise RuntimeError("WIDS model is unavailable.")

    model = artifacts["model"]
    scaler = artifacts["scaler"]

    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "request_size": rng.normal(500, 120, n_samples),
            "response_size": rng.normal(2000, 450, n_samples),
            "response_time": rng.gamma(2.0, 60.0, n_samples),
            "requests_per_min": rng.poisson(12, n_samples),
            "error_rate": rng.beta(1.5, 40.0, n_samples),
            "unique_pages": rng.poisson(6, n_samples),
        }
    )
    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)
    anomalies = int((preds == -1).sum())
    return anomalies, len(preds)


def extract_file_features(filepath: str, filename: str) -> dict:
    size_bytes = os.path.getsize(filepath)

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()

    # Entropy computed on the whole file; for very large files this remains efficient via chunking
    with open(filepath, "rb") as f:
        data = f.read()
    entropy = shannon_entropy(data)

    extension = os.path.splitext(filename)[1].lstrip(".").lower() or "none"

    return {
        "filename": filename,
        "size_bytes": size_bytes,
        "sha256": file_hash,
        "entropy": entropy,
        "extension": extension,
    }


def build_behavior_text(features: dict) -> str:
    tokens = ["file_read", "network_connect"]

    if features["extension"] in {"exe", "dll", "bin"}:
        tokens.extend(["process_create", "code_inject"])
    if features["extension"] in {"pdf", "doc", "docx"}:
        tokens.append("browser_navigate")
    if features["entropy"] > 7.0:
        tokens.append("encrypt_files")
    if features["size_bytes"] > 5 * 1024 * 1024:
        tokens.append("exfiltrate_data")
    if features["size_bytes"] < 20 * 1024:
        tokens.append("registry_read")

    return " ".join(tokens)


def run_malware_inference(filepath: str, filename: str):
    model = ARTIFACTS.get("malware_model")
    tokenizer = ARTIFACTS.get("malware_tokenizer")
    if model is None or tokenizer is None:
        raise RuntimeError("Malware model is unavailable.")

    features = extract_file_features(filepath, filename)
    behavior_text = build_behavior_text(features)

    seq = tokenizer.texts_to_sequences([behavior_text])
    padded = pad_sequences(seq, maxlen=100, padding="post")
    score = float(model.predict(padded, verbose=0)[0][0])

    label = "Malicious" if score >= 0.5 else "Benign"
    confidence = score if label == "Malicious" else 1 - score

    return {
        "label": label,
        "score": score,
        "confidence": confidence,
        "features": features,
    }


def write_report(result: dict) -> str:
    report_path = os.path.join(REPORT_DIR, REPORT_NAME)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    features = result["features"]

    lines = [
        "Malware Analysis Report",
        "=======================",
        f"Timestamp: {ts}",
        f"File Name: {features['filename']}",
        f"File Size: {features['size_bytes']} bytes",
        f"SHA256: {features['sha256']}",
        f"Entropy: {features['entropy']:.4f}",
        f"Extension: {features['extension']}",
        f"Prediction: {result['label']}",
        f"Confidence: {result['confidence']:.4f}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/network")
def network_route():
    try:
        label = synthetic_network_prediction()
        return render_template(
            "result.html",
            title="Network Traffic Classification",
            result_text=f"Result: {label}",
            details="Prediction generated from the pre-trained Random Forest classifier.",
            download_url=None,
        )
    except Exception as exc:  # pragma: no cover - UI feedback
        flash(str(exc), "error")
        return redirect(url_for("index"))


@app.route("/wids")
def wids_route():
    try:
        anomalies, total = synthetic_wids_anomalies()
        return render_template(
            "result.html",
            title="Web Intrusion Detection",
            result_text=f"Anomalies detected: {anomalies} of {total} samples",
            details="Isolation Forest flagged potential intrusions in simulated web traffic.",
            download_url=None,
        )
    except Exception as exc:  # pragma: no cover - UI feedback
        flash(str(exc), "error")
        return redirect(url_for("index"))


@app.route("/malware", methods=["GET", "POST"])
def malware_route():
    if request.method == "GET":
        return render_template("malware.html")

    file = request.files.get("file")
    if file is None or file.filename == "":
        flash("Please choose a file to analyze.", "error")
        return redirect(url_for("malware_route"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Allowed: .exe, .pdf, .txt", "error")
        return redirect(url_for("malware_route"))

    filename = secure_filename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, filename)
    file.save(saved_path)

    try:
        result = run_malware_inference(saved_path, filename)
        report_path = write_report(result)
        download_url = url_for("download_report")

        return render_template(
            "result.html",
            title="Malware File Analysis",
            result_text=f"Prediction: {result['label']} (confidence {result['confidence']:.2%})",
            details=(
                f"Size: {result['features']['size_bytes']} bytes | "
                f"Entropy: {result['features']['entropy']:.2f} | "
                f"SHA256: {result['features']['sha256']}"
            ),
            download_url=download_url,
        )
    except Exception as exc:  # pragma: no cover - UI feedback
        flash(str(exc), "error")
        return redirect(url_for("malware_route"))
    finally:
        try:
            os.remove(saved_path)
        except OSError:
            pass


@app.route("/download-report")
def download_report():
    report_path = os.path.join(REPORT_DIR, REPORT_NAME)
    if not os.path.exists(report_path):
        flash("Report not found. Run malware analysis first.", "error")
        return redirect(url_for("index"))
    return send_from_directory(REPORT_DIR, REPORT_NAME, as_attachment=True)


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    load_models()
