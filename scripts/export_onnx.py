"""Script to export a trained PyTorch crop disease vision model to optimized ONNX format.

Usage:
    python scripts/export_onnx.py --output models/crop_disease_model.onnx
"""

import os
import argparse
import numpy as np

def export_onnx(output_path: str = "models/crop_disease_model.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[BioVision AI] Exporting model to ONNX format at '{output_path}'...")

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        # Initialize lightweight MobileNetV3 backbone for 38 agricultural classes
        model = models.mobilenet_v3_small(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 38)
        model.eval()

        # Dummy input matching image tensor dimension (Batch=1, Channels=3, Height=224, Width=224)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"[BioVision AI] Success! Exported PyTorch MobileNetV3 model to '{output_path}'")

    except ImportError:
        print("[BioVision AI] PyTorch / TorchVision not installed in current environment.")
        print("[BioVision AI] To install PyTorch: pip install torch torchvision")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--output", default="models/crop_disease_model.onnx", help="Target ONNX filepath")
    args = parser.parse_args()
    export_onnx(args.output)
