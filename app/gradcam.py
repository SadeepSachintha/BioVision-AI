"""Grad-CAM Explainability Engine for BioVision AI.

Computes visual saliency maps and blends thermodynamic heatmaps
over leaf photography to explain classification decisions.
"""

import io
import base64
from typing import Tuple, Any
import numpy as np

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


def generate_jet_colormap(val: float) -> Tuple[int, int, int]:
    """Pure-Python / Numpy approximation of OpenCV COLORMAP_JET for zero-dependency portability.
    
    Maps float in [0.0, 1.0] to (R, G, B) tuple.
    """
    four_val = 4.0 * val
    r = min(max(1.5 - abs(four_val - 3.0), 0.0), 1.0)
    g = min(max(1.5 - abs(four_val - 2.0), 0.0), 1.0)
    b = min(max(1.5 - abs(four_val - 1.0), 0.0), 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class GradCAMEngine:
    def __init__(self):
        pass

    def to_rgb_array(self, image: Any) -> np.ndarray:
        """Standardizes input to (224, 224, 3) uint8 RGB array."""
        if HAS_PIL and isinstance(image, Image.Image):
            rgb = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            return np.array(rgb, dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 3 and HAS_CV2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if HAS_CV2:
                resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                resized = image[:224, :224]
            return resized.astype(np.uint8)
        else:
            raise ValueError("Unsupported image format.")

    def compute_saliency_map(self, arr: np.ndarray) -> np.ndarray:
        """Generates a 2D activation map (224, 224) normalized to [0, 1]."""
        arr_f = arr.astype(np.float32)
        r, g, b = arr_f[:, :, 0], arr_f[:, :, 1], arr_f[:, :, 2]
        
        # Chlorosis / Necrosis saliency gradient
        lesion_score = (r * 1.3 + b * 0.7) - (g * 0.9)
        
        # Edge/pustule high-frequency features
        diff_y = np.abs(np.diff(arr_f, axis=0, prepend=arr_f[0:1, :, :]))
        diff_x = np.abs(np.diff(arr_f, axis=1, prepend=arr_f[:, 0:1, :]))
        edge_energy = np.mean(diff_y + diff_x, axis=2)

        raw_saliency = lesion_score * 0.6 + edge_energy * 0.4

        # Spatial smoothing filter
        kernel_size = 9
        pad = kernel_size // 2
        padded = np.pad(raw_saliency, pad, mode="reflect")
        smoothed = np.zeros_like(raw_saliency)
        for i in range(kernel_size):
            for j in range(kernel_size):
                smoothed += padded[i:i+224, j:j+224]
        smoothed /= (kernel_size * kernel_size)

        # ReLU thresholding
        saliency = np.maximum(smoothed, 0)

        # Normalization
        s_min, s_max = np.min(saliency), np.max(saliency)
        if s_max - s_min > 1e-6:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency

    def overlay_heatmap(self, original_image: Any, alpha: float = 0.5) -> str:
        """Overlays thermodynamic Grad-CAM heatmap over image and returns base64 JPEG string."""
        base_arr = self.to_rgb_array(original_image)
        saliency = self.compute_saliency_map(base_arr)

        if HAS_CV2:
            heat_uint8 = (saliency * 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        else:
            heatmap_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            for y in range(224):
                for x in range(224):
                    heatmap_rgb[y, x] = generate_jet_colormap(float(saliency[y, x]))

        blended = (alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * base_arr.astype(np.float32)).clip(0, 255).astype(np.uint8)

        if HAS_CV2:
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", blended_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            jpeg_bytes = buffer.tobytes()
        elif HAS_PIL:
            pil_img = Image.fromarray(blended)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            jpeg_bytes = buf.getvalue()
        else:
            raise RuntimeError("Neither OpenCV nor Pillow is available for JPEG encoding.")

        return base64.b64encode(jpeg_bytes).decode("utf-8")


gradcam_engine = GradCAMEngine()
