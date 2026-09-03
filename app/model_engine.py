"""Model Engine for BioVision AI.

Manages ONNX Runtime inference, image tensor preprocessing,
and fallback execution providers.
"""

import os
import time
import io
from typing import List, Tuple, Dict, Any
import numpy as np
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Canonical list of 38 PlantVillage agricultural classes
CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


class ONNXModelEngine:
    def __init__(self, model_path: str = "models/crop_disease_model.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.provider = "Uninitialized"
        self.is_loaded = False
        self.classes = CLASSES

        self._load_model()

    def _load_model(self):
        if HAS_ORT and os.path.exists(self.model_path):
            try:
                available_providers = ort.get_available_providers()
                chosen_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                providers = [p for p in chosen_providers if p in available_providers]
                if not providers:
                    providers = ["CPUExecutionProvider"]

                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.provider = self.session.get_providers()[0]
                self.is_loaded = True
                print(f"[BioVision AI] Successfully loaded ONNX model from '{self.model_path}' using '{self.provider}'")
            except Exception as e:
                print(f"[BioVision AI] Warning: Failed to load ONNX session: {e}. Running in heuristic simulation mode.")
                self.provider = "SimulatedEngine (ORT Error Fallback)"
                self.is_loaded = False
        else:
            if not HAS_ORT:
                print("[BioVision AI] onnxruntime not installed in environment. Running in heuristic simulation mode.")
            else:
                print(f"[BioVision AI] ONNX model not found at '{self.model_path}'. Running in heuristic simulation mode.")
            self.provider = "SimulatedEngine (Offline Fallback)"
            self.is_loaded = False

    def preprocess_image(self, image: Any) -> np.ndarray:
        """Preprocesses image (PIL Image or numpy RGB/BGR array) into normalized (1, 3, 224, 224) tensor."""
        if HAS_PIL and isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            img_data = np.array(image, dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            # If loaded with cv2 (BGR)
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 3 and HAS_CV2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if HAS_CV2:
                resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                resized = image[:224, :224]
            img_data = resized.astype(np.float32) / 255.0
        else:
            raise ValueError("Unsupported image format. Provide PIL Image or numpy ndarray.")

        # Standard ImageNet normalization: (x - mean) / std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_data - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        tensor = np.transpose(norm_img, (2, 0, 1))

        # Add batch dimension -> (1, C, H, W)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        return tensor

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, image: Any, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        """Runs inference and returns ranked predictions with latency in milliseconds."""
        start_time = time.perf_counter()

        tensor = self.preprocess_image(image)

        if self.is_loaded and self.session is not None:
            ort_inputs = {self.input_name: tensor}
            ort_outs = self.session.run([self.output_name], ort_inputs)
            logits = ort_outs[0][0]
            probabilities = self.softmax(logits)
        else:
            # Deterministic simulation mode from tensor stats
            # Channel 0: Red, Channel 1: Green, Channel 2: Blue
            r_mean = float(np.mean(tensor[0, 0, :, :]))
            g_mean = float(np.mean(tensor[0, 1, :, :]))
            b_mean = float(np.mean(tensor[0, 2, :, :]))

            np.random.seed(int(abs(r_mean * 7 + g_mean * 13 + b_mean * 19) * 1000) % 100000)
            sim_logits = np.random.normal(0, 1.0, len(self.classes))

            # If heavily green, bias toward healthy; else bias toward blight
            if g_mean > r_mean and g_mean > b_mean:
                sim_logits[CLASSES.index("Tomato___healthy")] += 6.5
            else:
                sim_logits[CLASSES.index("Tomato___Late_blight")] += 7.0
                sim_logits[CLASSES.index("Tomato___Early_blight")] += 4.5
                sim_logits[CLASSES.index("Corn___Northern_Leaf_Blight")] += 3.8

            probabilities = self.softmax(sim_logits)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            cls_name = self.classes[idx]
            parts = cls_name.split("___")
            crop = parts[0].replace("_", " ")
            condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            results.append({
                "class_name": cls_name,
                "crop": crop,
                "condition": condition,
                "confidence": float(round(probabilities[idx], 4))
            })

        return results, round(latency_ms, 2)


# Singleton instance
engine = ONNXModelEngine()
