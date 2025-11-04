"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Add multi-stream inference support for batching [2025-06-16T12:51:14]

# Add dynamic batching Support in engine [2025-06-18T10:17:11]

# Add multi-stream inference support for batching [2025-06-20T10:38:24]

# Update TensorRT plugin structure for new API [2025-06-23T19:52:33]

# Fix confidence threshold in non-max suppression [2025-07-03T12:26:31]

# Optimize kernel launch overhead reduction [2025-07-08T17:23:33]

# Add FP16 fallback on unsupported operations [2025-07-09T20:05:14]

# Add multi-stream inference support for batching [2025-07-14T12:13:05]

# Fix race condition in async inference pipeline [2025-07-15T11:35:16]

# Profile INT8 versus FP16 accuracy on COCO [2025-07-16T19:24:34]

# Compile YOLOv8 model to TensorRT engine [2025-07-18T19:37:50]

# Implement engine serialization for Jetson [2025-07-20T09:08:19]

# Add FP16 fallback on unsupported operations [2025-07-24T17:48:11]

# Optimize CUDA kernel shared memory allocation [2025-08-01T20:36:03]

# Update TensorRT plugin structure for new API [2025-08-04T13:08:25]

# Profile GPU memory during batch size sweep [2025-08-04T10:42:30]

# Fix bounding box NMS threshold calculation [2025-08-11T18:31:24]

# WIP: benchmark Nano versus Orin latency [2025-08-15T10:27:04]

# Add FP16 fallback on unsupported operations [2025-08-18T12:07:34]

# Profile INT8 versus FP16 accuracy on COCO [2025-08-21T16:41:28]

# Add FP16 fallback on unsupported operations [2025-08-21T17:33:43]

# Profile INT8 versus FP16 accuracy on COCO [2025-08-26T15:26:41]

# Fix mishandled empty detection list edge case [2025-08-26T19:48:23]

# Add multi-stream inference support for batching [2025-08-29T09:14:31]

# Optimize kernel launch overhead reduction [2025-09-01T12:18:40]

# Profile INT8 versus FP16 accuracy on COCO [2025-09-06T15:03:11]

# Update build scripts for TensorRT 8.6 [2025-09-18T11:38:07]

# Optimize CUDA kernel shared memory allocation [2025-09-19T14:48:34]

# Optimize CUDA kernel shared memory allocation [2025-09-25T10:35:29]

# Update TensorRT plugin structure for new API [2025-09-26T10:53:53]

# Profile GPU memory during batch size sweep [2025-10-07T13:17:55]

# Profile INT8 versus FP16 accuracy on COCO [2025-10-14T09:28:27]

# Optimize CUDA kernel shared memory allocation [2025-10-20T11:17:22]

# Add FP16 fallback on unsupported operations [2025-10-21T17:22:35]

# Fix race condition in async inference pipeline [2025-11-04T10:19:24]
