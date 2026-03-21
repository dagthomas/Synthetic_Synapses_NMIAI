"""
Product classifier for re-classification of YOLO detections.
Supports ConvNeXt-V2 nano (default) and EfficientNet-B2 (fallback).

Used by run.py at inference time.
This file must be included in the submission zip.
"""

import json
import numpy as np
import torch
import timm
from safetensors.torch import load_file
from pathlib import Path


# Defaults (overridden by classifier_config.json if present)
MODEL_NAME = "convnextv2_nano"
NUM_CLASSES = 357
INPUT_SIZE = 256


def load_classifier(weights_path=None, device="cuda"):
    """Load classifier with trained weights from safetensors.

    Reads model architecture from classifier_config.json if available,
    otherwise falls back to defaults.
    """
    script_dir = Path(__file__).resolve().parent

    if weights_path is None:
        weights_path = script_dir / "classifier.safetensors"
    else:
        weights_path = Path(weights_path)

    # Load config if available
    config_path = script_dir / "classifier_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("model_name", MODEL_NAME)
        num_classes = config.get("num_classes", NUM_CLASSES)
        input_size = config.get("input_size", INPUT_SIZE)
    else:
        model_name = MODEL_NAME
        num_classes = NUM_CLASSES
        input_size = INPUT_SIZE

    # Create model without pretrained weights (we load our own)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Load FP16 weights
    state_dict = load_file(str(weights_path))
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.half().to(device)
    model.eval()

    # Store config on model for run.py to read
    model.input_size = input_size
    model.num_classes = num_classes

    return model
