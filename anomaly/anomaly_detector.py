"""
Visual and Semantic Anomaly Detection Engine.
Combines:
1. Semantic / Affordance Anomaly: Detects hazardous or conflicting object selections
   (e.g., knife selected for 'serve wine' or 'eat soup').
2. Visual Structural Anomaly: Evaluates Convolutional Autoencoder reconstruction error
   (MSE > 0.045 flags physical damage, foreign debris, or scene defects).
"""

from typing import List, Dict, Any, Optional
import time
try:
    from models.task_detector import DetectionResult, TaskConditionedHead
    from models.autoencoder_anomaly import AnomalyReconstructionScorer
except (ImportError, ValueError):
    from ..models.task_detector import DetectionResult, TaskConditionedHead
    from ..models.autoencoder_anomaly import AnomalyReconstructionScorer


class AnomalyReport:
    """Encapsulates the anomaly evaluation findings for a scene frame."""
    def __init__(
        self,
        is_anomalous: bool,
        anomaly_type: str,
        severity: str,
        score: float,
        description: str,
        affected_objects: List[str],
    ):
        self.is_anomalous = is_anomalous
        self.anomaly_type = anomaly_type       # "SEMANTIC_HAZARD", "VISUAL_DEFECT", "NONE"
        self.severity = severity               # "HIGH", "MEDIUM", "LOW", "NONE"
        self.score = round(float(score), 4)
        self.description = description
        self.affected_objects = affected_objects
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomalous": self.is_anomalous,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "score": self.score,
            "description": self.description,
            "affected_objects": self.affected_objects,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        status = "ALERT" if self.is_anomalous else "OK"
        return f"[{status}] Type: {self.anomaly_type} | Sev: {self.severity} | Score: {self.score:.3f} | {self.description}"


class VisualAnomalyDetector:
    """
    Dual-Stage Anomaly Detection Engine:
    - Layer 1: Evaluates semantic affordance safety constraints from DVCon task priors.
    - Layer 2: Evaluates Autoencoder visual reconstruction error across object patches.
    """

    def __init__(
        self,
        affordance_matrix_path: Optional[str] = None,
        reconstruction_threshold: float = 0.045,
    ):
        self.head = TaskConditionedHead(affordance_matrix_path)
        self.scorer = AnomalyReconstructionScorer(threshold=reconstruction_threshold)
        self.reconstruction_threshold = reconstruction_threshold

    def evaluate_affordance_anomalies(
        self,
        task_name: str,
        detections: List[DetectionResult],
    ) -> Optional[AnomalyReport]:
        """
        Flags any detected object that violates safety constraints for the active task.
        """
        hazards_found = []
        for det in detections:
            if det.is_hazard or self.head.is_hazard(task_name, det.class_name):
                hazards_found.append(det.class_name)

        if hazards_found:
            return AnomalyReport(
                is_anomalous=True,
                anomaly_type="SEMANTIC_HAZARD",
                severity="HIGH",
                score=0.92,
                description=f"Hazardous object(s) {hazards_found} detected in task area for '{task_name}'",
                affected_objects=hazards_found,
            )
        return None

    def evaluate_visual_reconstruction(
        self,
        patch_pixels: Optional[List[List[float]]] = None,
    ) -> Optional[AnomalyReport]:
        """
        Evaluates visual reconstruction error from the ConvAutoencoder.
        """
        if patch_pixels is None:
            return None

        result = self.scorer.compute_patch_anomaly(patch_pixels)
        if result["is_anomalous"]:
            return AnomalyReport(
                is_anomalous=True,
                anomaly_type="VISUAL_DEFECT",
                severity="MEDIUM",
                score=result["mse_score"],
                description=f"Autoencoder reconstruction error ({result['mse_score']:.4f}) exceeded threshold ({self.reconstruction_threshold})",
                affected_objects=["workspace_patch"],
            )
        return None

    def inspect_scene(
        self,
        task_name: str,
        detections: List[DetectionResult],
        sample_patch: Optional[List[List[float]]] = None,
    ) -> AnomalyReport:
        """
        Comprehensive scene inspection synthesizing semantic affordance & visual autoencoder checks.
        """
        # Check Layer 1 (Semantic affordance hazard)
        semantic_report = self.evaluate_affordance_anomalies(task_name, detections)
        if semantic_report:
            return semantic_report

        # Check Layer 2 (Visual reconstruction anomaly)
        if sample_patch is not None:
            visual_report = self.evaluate_visual_reconstruction(sample_patch)
            if visual_report:
                return visual_report

        return AnomalyReport(
            is_anomalous=False,
            anomaly_type="NONE",
            severity="NONE",
            score=0.012,
            description="All observed objects comply with task affordance priors; visual reconstruction nominal.",
            affected_objects=[],
        )
