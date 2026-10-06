"""
Models package for Task-Aware Multi-Object Tracking & Visual Anomaly Detection.
Includes:
- TaskAwareDetector: Lightweight student detector with task-conditioned reasoning head
- TeacherStudentDistillation: Parent-teacher knowledge distillation engine
- ConvAutoencoder: Convolutional autoencoder for visual anomaly reconstruction
- ModelExporter: ONNX and TensorRT FP16 export utilities
"""

from .task_detector import TaskAwareDetector, TaskConditionedHead
from .distillation import TeacherStudentDistillation, DistillationLoss
from .autoencoder_anomaly import ConvAutoencoder, AnomalyReconstructionScorer

__all__ = [
    "TaskAwareDetector",
    "TaskConditionedHead",
    "TeacherStudentDistillation",
    "DistillationLoss",
    "ConvAutoencoder",
    "AnomalyReconstructionScorer",
]
