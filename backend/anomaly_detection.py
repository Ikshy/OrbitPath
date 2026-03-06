"""
anomaly_detection.py
====================
AI-powered orbital anomaly detection module for OrbitPath.

Responsibilities:
    - Extract feature vectors from satellite orbital state data
    - Train / load an Isolation Forest anomaly detection model
    - Score satellites and flag those with unusual orbital behaviour
    - Persist the trained model to disk for reuse between API calls

Algorithm — Isolation Forest:
    Isolation Forest is an unsupervised ensemble method that isolates
    anomalies by randomly partitioning feature space. Anomalies (e.g.
    manoeuvred satellites, debris with unusual eccentricity, decaying
    objects) are isolated with fewer splits and therefore receive a
    lower anomaly score (closer to −1).

Features used per satellite:
    - altitude_km        : Height above Earth surface
    - speed_km_s         : Orbital velocity magnitude
    - inclination_deg    : Orbital plane inclination (from TLE)
    - eccentricity       : Orbit shape (0 = circular, 1 = parabolic)
    - mean_motion_rpm    : Revolutions per minute (period proxy)
    - bstar_drag         : Atmospheric drag coefficient (decay indicator)

Extending the model:
    To add temporal anomaly detection (e.g. sudden trajectory deviation),
    replace the static feature vector with a time-series window and use
    a One-Class SVM or LSTM autoencoder.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from satellite_tracker import SatelliteRecord

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "models", "anomaly_model.pkl")
SCALER_PATH     = os.path.join(os.path.dirname(__file__), "models", "anomaly_scaler.pkl")

# Isolation Forest hyper-parameters
IF_CONTAMINATION = 0.05    # Expected fraction of anomalies in the training data
IF_N_ESTIMATORS  = 200     # Number of isolation trees
IF_RANDOM_STATE  = 42      # Reproducibility seed

# Anomaly score thresholds (sklearn decision_function output, higher = more normal)
SCORE_ANOMALOUS = -0.05    # Scores below this are flagged as anomalous


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AnomalyReport:
    """Per-satellite anomaly assessment."""
    norad_id:         int
    name:             str
    anomaly_score:    float      # Raw IF decision function value
    is_anomalous:     bool
    anomaly_label:    str        # "NORMAL" | "ANOMALOUS"
    reason:           str        # Human-readable explanation
    features:         dict       # Feature values used for scoring


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(sat: SatelliteRecord) -> Optional[np.ndarray]:
    """
    Build a 1-D feature vector from a SatelliteRecord.

    Args:
        sat: SatelliteRecord with position populated and valid Satrec.

    Returns:
        NumPy array of shape (6,) or None if data is insufficient.
    """
    pos = sat.position
    if pos is None:
        return None

    satrec = sat.satrec
    try:
        # sgp4 stores inclination in radians, eccentricity as float
        inclination_deg = float(np.degrees(satrec.inclo))
        eccentricity    = float(satrec.ecco)
        mean_motion_rpm = float(satrec.no_kozai)   # rad/min
        bstar_drag      = float(satrec.bstar)

        altitude_km  = pos.altitude
        speed_km_s   = pos.speed

        return np.array([
            altitude_km,
            speed_km_s,
            inclination_deg,
            eccentricity,
            mean_motion_rpm,
            bstar_drag,
        ], dtype=np.float64)

    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Feature extraction failed for '%s': %s", sat.name, exc)
        return None


def extract_feature_dict(sat: SatelliteRecord) -> Optional[dict]:
    """Return features as a labelled dict (for API responses)."""
    pos = sat.position
    if pos is None:
        return None
    satrec = sat.satrec
    try:
        return {
            "altitude_km":       round(pos.altitude, 2),
            "speed_km_s":        round(pos.speed, 4),
            "inclination_deg":   round(float(np.degrees(satrec.inclo)), 4),
            "eccentricity":      round(float(satrec.ecco), 6),
            "mean_motion_rpm":   round(float(satrec.no_kozai), 8),
            "bstar_drag":        round(float(satrec.bstar), 10),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
def _ensure_model_dir() -> None:
    """Create the models/ directory if it does not exist."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def save_model(model: IsolationForest, scaler: StandardScaler) -> None:
    """Persist a trained model and scaler to disk."""
    _ensure_model_dir()
    with open(MODEL_PATH,  "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Anomaly model saved to %s", MODEL_PATH)


def load_model() -> tuple[Optional[IsolationForest], Optional[StandardScaler]]:
    """
    Load a previously saved model and scaler from disk.

    Returns:
        (model, scaler) or (None, None) if no model exists yet.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        with open(MODEL_PATH,  "rb") as f:
            model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Anomaly model loaded from %s", MODEL_PATH)
        return model, scaler
    except Exception as exc:
        logger.warning("Failed to load anomaly model: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_anomaly_model(satellites: list[SatelliteRecord]) -> tuple[IsolationForest, StandardScaler]:
    """
    Train an Isolation Forest on the current satellite catalogue.

    Training is triggered automatically when no persisted model exists, or
    explicitly via the /anomaly-report?retrain=true query parameter.

    Args:
        satellites: List of SatelliteRecord objects (positions populated).

    Returns:
        Fitted (model, scaler) tuple.

    Raises:
        ValueError: Fewer than 10 valid satellites to train on.
    """
    feature_vectors: list[np.ndarray] = []

    for sat in satellites:
        vec = extract_features(sat)
        if vec is not None and np.isfinite(vec).all():
            feature_vectors.append(vec)

    if len(feature_vectors) < 10:
        raise ValueError(f"Insufficient training data: only {len(feature_vectors)} valid satellites.")

    X = np.array(feature_vectors)

    # Standardise features so no single feature dominates by scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    save_model(model, scaler)
    logger.info("Isolation Forest trained on %d satellites.", len(feature_vectors))
    return model, scaler


# ---------------------------------------------------------------------------
# Scoring & reporting
# ---------------------------------------------------------------------------
def _explain_anomaly(features: dict, score: float) -> str:
    """
    Generate a plain-English reason string for an anomalous classification.

    This simple rule-based explainer covers the most common orbital anomaly
    signatures. Replace with SHAP values for production-grade explainability.
    """
    reasons: list[str] = []

    # Very low altitude → decaying / recently manoeuvred
    if features.get("altitude_km", 500) < 250:
        reasons.append(f"very low altitude ({features['altitude_km']:.0f} km — possible decay)")

    # High eccentricity → non-circular (unusual for operational satellites)
    if features.get("eccentricity", 0) > 0.1:
        reasons.append(f"high eccentricity ({features['eccentricity']:.4f})")

    # Abnormally high drag coefficient → active manoeuvre or tumbling debris
    if abs(features.get("bstar_drag", 0)) > 0.01:
        reasons.append(f"elevated B* drag term ({features['bstar_drag']:.6f})")

    # Unusually high mean motion → very low orbit (fast decay)
    if features.get("mean_motion_rpm", 0) > 0.075:
        reasons.append("high mean motion (very low orbit)")

    if not reasons:
        reasons.append(f"statistical anomaly (IF score {score:.4f})")

    return "; ".join(reasons).capitalize() + "."


def score_satellites(
    satellites: list[SatelliteRecord],
    model: IsolationForest,
    scaler: StandardScaler,
) -> list[AnomalyReport]:
    """
    Score each satellite and return a sorted anomaly report.

    Args:
        satellites: Satellite records with positions.
        model:      Fitted IsolationForest.
        scaler:     Fitted StandardScaler.

    Returns:
        List of AnomalyReport sorted: anomalous first, then by score ascending.
    """
    reports: list[AnomalyReport] = []

    for sat in satellites:
        vec      = extract_features(sat)
        feat_dict = extract_feature_dict(sat)
        if vec is None or not np.isfinite(vec).all() or feat_dict is None:
            continue

        X_scaled = scaler.transform(vec.reshape(1, -1))
        score    = float(model.decision_function(X_scaled)[0])
        is_anom  = score < SCORE_ANOMALOUS

        reports.append(AnomalyReport(
            norad_id=sat.norad_id,
            name=sat.name,
            anomaly_score=round(score, 6),
            is_anomalous=is_anom,
            anomaly_label="ANOMALOUS" if is_anom else "NORMAL",
            reason=_explain_anomaly(feat_dict, score) if is_anom else "Orbital parameters within expected range.",
            features=feat_dict,
        ))

    # Sort: anomalous first, then by score (most anomalous = lowest score first)
    reports.sort(key=lambda r: (not r.is_anomalous, r.anomaly_score))
    logger.info(
        "Anomaly scoring complete: %d/%d flagged.",
        sum(r.is_anomalous for r in reports),
        len(reports),
    )
    return reports


# ---------------------------------------------------------------------------
# High-level entry point used by app.py
# ---------------------------------------------------------------------------
def run_anomaly_detection(
    satellites: list[SatelliteRecord],
    force_retrain: bool = False,
) -> tuple[list[AnomalyReport], dict]:
    """
    Orchestrate model training (if needed) and satellite scoring.

    Args:
        satellites:    Current satellite catalogue.
        force_retrain: If True, always retrain even if a model exists.

    Returns:
        (reports, meta) where meta contains model info for the API response.
    """
    model, scaler = load_model() if not force_retrain else (None, None)

    newly_trained = False
    if model is None or scaler is None:
        logger.info("Training new anomaly detection model…")
        model, scaler = train_anomaly_model(satellites)
        newly_trained = True

    reports = score_satellites(satellites, model, scaler)

    meta = {
        "model_type":       "IsolationForest",
        "n_estimators":     IF_N_ESTIMATORS,
        "contamination":    IF_CONTAMINATION,
        "newly_trained":    newly_trained,
        "satellites_scored": len(reports),
        "anomalies_found":  sum(r.is_anomalous for r in reports),
    }

    return reports, meta


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------
def anomaly_report_to_dict(report: AnomalyReport) -> dict:
    """Convert an AnomalyReport dataclass to a JSON-serialisable dictionary."""
    return {
        "norad_id":      report.norad_id,
        "name":          report.name,
        "anomaly_score": report.anomaly_score,
        "is_anomalous":  report.is_anomalous,
        "label":         report.anomaly_label,
        "reason":        report.reason,
        "features":      report.features,
    }