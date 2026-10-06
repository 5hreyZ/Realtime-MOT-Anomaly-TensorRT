"""
Edge Deployment package supporting:
- PYNQ-Z2 FPGA / ARM Cortex-A9 edge deployment via OpenCV DNN (DVCon setup)
- NVIDIA GPU / Jetson deployment via TensorRT FP16 (Resume specification)
"""

from .pynq_edge_runner import PynqEdgeRunner
from .jetson_tensorrt_runner import JetsonTensorRTRunner

__all__ = ["PynqEdgeRunner", "JetsonTensorRTRunner"]
