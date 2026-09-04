"""BioVision AI — Data & Concept Drift Monitoring Engine.

Tracks input foliar image distribution shifts (covariate shift) and
prediction entropy degradation (concept drift) over a rolling inference window.
"""

import math
from collections import deque
from typing import Any

import numpy as np


class FoliarDriftMonitor:
    """Monitors distribution drift for agricultural computer vision models."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Rolling buffers for incoming telemetry
        self.entropy_buffer: deque = deque(maxlen=window_size)
        self.confidence_buffer: deque = deque(maxlen=window_size)
        self.green_ratio_buffer: deque = deque(maxlen=window_size)
        self.luminance_buffer: deque = deque(maxlen=window_size)

        # Calibrated baseline profiles (derived from golden training set)
        self.baseline_stats = {
            "mean_confidence": 0.942,
            "mean_entropy": 0.420,
            "mean_green_ratio": 1.340,
            "mean_luminance": 118.5,
            "confidence_threshold": 0.700,
            "entropy_alert_threshold": 1.250,
        }

    def compute_image_metrics(self, image: Any) -> dict[str, float]:
        """Calculates color distribution and luminance metrics from an image."""
        if hasattr(image, "convert"):
            arr = np.array(image.convert("RGB"), dtype=np.float32)
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
        else:
            return {"green_ratio": 1.3, "luminance": 120.0}

        r = np.mean(arr[:, :, 0])
        g = np.mean(arr[:, :, 1])
        b = np.mean(arr[:, :, 2])

        # Green channel dominance ratio: g / ((r + b) / 2 + 1e-5)
        green_ratio = float(g / ((r + b) / 2.0 + 1e-5))

        # Standard relative luminance: 0.2126 R + 0.7152 G + 0.0722 B
        luminance = float(0.2126 * r + 0.7152 * g + 0.0722 * b)

        return {
            "green_ratio": round(green_ratio, 3),
            "luminance": round(luminance, 2),
        }

    def compute_prediction_entropy(
        self, probabilities: list[float] | np.ndarray
    ) -> float:
        """Calculates Shannon entropy: H(p) = -sum(p_i * log2(p_i))."""
        entropy = 0.0
        for p in probabilities:
            if p > 1e-6:
                entropy -= float(p * math.log2(p))
        return round(entropy, 3)

    def record_inference(
        self,
        image: Any,
        top_confidence: float,
        all_probabilities: list[float] | np.ndarray,
    ):
        """Records telemetry for an inference request into rolling buffers."""
        img_metrics = self.compute_image_metrics(image)
        entropy = self.compute_prediction_entropy(all_probabilities)

        self.entropy_buffer.append(entropy)
        self.confidence_buffer.append(top_confidence)
        self.green_ratio_buffer.append(img_metrics["green_ratio"])
        self.luminance_buffer.append(img_metrics["luminance"])

    def calculate_ks_statistic(
        self, sample: list[float], baseline_mean: float
    ) -> float:
        """Estimates empirical Kolmogorov-Smirnov distance from baseline center."""
        if not sample:
            return 0.0
        sample_mean = float(np.mean(sample))
        std = float(np.std(sample)) + 1e-5
        # Standardized difference
        z_score = abs(sample_mean - baseline_mean) / std
        # Approximate KS distance bounded in [0.0, 1.0]
        ks_dist = 1.0 - math.exp(-0.5 * (z_score**2))
        return round(ks_dist, 4)

    def evaluate_drift(self) -> dict[str, Any]:
        """Evaluates rolling window against baseline and returns full drift audit."""
        count = len(self.confidence_buffer)

        if count < 5:
            # Insufficient samples for statistical significance
            return {
                "status": "INITIALIZING",
                "sample_count": count,
                "window_capacity": self.window_size,
                "drift_detected": False,
                "message": f"Collecting initial telemetry baseline ({count}/5 minimum samples)...",
                "metrics": {
                    "rolling_mean_confidence": (
                        round(float(np.mean(self.confidence_buffer)), 3)
                        if count > 0
                        else 0.0
                    ),
                    "rolling_mean_entropy": (
                        round(float(np.mean(self.entropy_buffer)), 3)
                        if count > 0
                        else 0.0
                    ),
                    "rolling_green_ratio": (
                        round(float(np.mean(self.green_ratio_buffer)), 3)
                        if count > 0
                        else 0.0
                    ),
                    "rolling_luminance": (
                        round(float(np.mean(self.luminance_buffer)), 2)
                        if count > 0
                        else 0.0
                    ),
                },
                "ks_covariate_shift": 0.0,
                "recommendation": "Maintain standard continuous operation.",
            }

        mean_conf = float(np.mean(self.confidence_buffer))
        mean_entropy = float(np.mean(self.entropy_buffer))
        mean_green = float(np.mean(self.green_ratio_buffer))
        mean_lum = float(np.mean(self.luminance_buffer))

        # Statistical tests
        ks_covariate = self.calculate_ks_statistic(
            list(self.green_ratio_buffer), self.baseline_stats["mean_green_ratio"]
        )
        ks_luminance = self.calculate_ks_statistic(
            list(self.luminance_buffer), self.baseline_stats["mean_luminance"]
        )

        # Concept drift flag: low confidence or high entropy
        concept_drift = (
            mean_conf < self.baseline_stats["confidence_threshold"]
            or mean_entropy > self.baseline_stats["entropy_alert_threshold"]
        )

        # Covariate shift flag: KS distance > 0.40
        covariate_shift = ks_covariate > 0.40 or ks_luminance > 0.50

        # Classification
        if concept_drift and covariate_shift:
            status = "CRITICAL_DRIFT_ALERT"
            drift_detected = True
            recommendation = "Severe seasonal/environmental shift detected. Trigger DVC data re-ingestion and schedule active learning re-annotation."
        elif concept_drift:
            status = "CONCEPT_DRIFT_WARNING"
            drift_detected = True
            recommendation = "Prediction confidence degraded. Potential out-of-distribution crop specimens or unseen pathogens observed."
        elif covariate_shift:
            status = "COVARIATE_SHIFT_WARNING"
            drift_detected = True
            recommendation = "Lighting or sensor color distribution shifted from baseline. Verify field camera calibration."
        else:
            status = "NORMAL"
            drift_detected = False
            recommendation = "Input distribution and model confidence are aligned with golden baseline."

        return {
            "status": status,
            "sample_count": count,
            "window_capacity": self.window_size,
            "drift_detected": drift_detected,
            "message": f"Evaluated {count} recent foliar inference events.",
            "metrics": {
                "rolling_mean_confidence": round(mean_conf, 3),
                "baseline_confidence": self.baseline_stats["mean_confidence"],
                "rolling_mean_entropy": round(mean_entropy, 3),
                "baseline_entropy": self.baseline_stats["mean_entropy"],
                "rolling_green_ratio": round(mean_green, 3),
                "baseline_green_ratio": self.baseline_stats["mean_green_ratio"],
                "rolling_luminance": round(mean_lum, 2),
                "baseline_luminance": self.baseline_stats["mean_luminance"],
            },
            "ks_covariate_shift": ks_covariate,
            "ks_luminance_shift": ks_luminance,
            "recommendation": recommendation,
        }


# Singleton monitor instance
drift_monitor = FoliarDriftMonitor()
