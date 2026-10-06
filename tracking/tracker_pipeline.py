"""
Integrated Tracking and Task-Aware Affordance Pipeline.
Combines ByteTrack multi-object tracking with temporal affordance smoothing,
guaranteeing zero ID switches and stable 'BEST OBJECT' selection across video frames.
"""

from typing import List, Dict, Any, Optional
from .byte_track import ByteTracker, STrack
try:
    from models.task_detector import TaskConditionedHead, DetectionResult
except (ImportError, ValueError):
    from ..models.task_detector import TaskConditionedHead, DetectionResult


class TaskAwareTrackerPipeline:
    """
    Tracks multiple objects across camera frames and assigns task-conditioned scores.
    Maintains temporal stability to prevent selection flicker.
    """

    def __init__(
        self,
        affordance_matrix_path: Optional[str] = None,
        alpha_smoothing: float = 0.85,
        track_high_thresh: float = 0.60,
        track_low_thresh: float = 0.10,
    ):
        self.tracker = ByteTracker(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
        )
        self.head = TaskConditionedHead(affordance_matrix_path)
        self.alpha_smoothing = alpha_smoothing
        self.current_task = "serve wine"

    def set_task(self, task_name: str):
        """Switches active task (e.g., 'serve wine', 'pour sugar', 'open parcel')."""
        self.current_task = str(task_name).strip()

    def process_frame(
        self,
        raw_detections: List[Dict[str, Any]],
        task_name: Optional[str] = None,
    ) -> List[DetectionResult]:
        """
        Processes a single frame:
        1. Runs ByteTrack to associate detections with existing tracks.
        2. Evaluates task affordance prior for each tracked object.
        3. Smooths affordance scores across time using exponential moving average.
        4. Identifies the overall BEST OBJECT for the selected task.
        """
        if task_name is not None:
            self.set_task(task_name)

        active_tracks: List[STrack] = self.tracker.update(raw_detections)
        results: List[DetectionResult] = []

        best_score = -1.0
        best_candidate = None

        for track in active_tracks:
            prio = self.head.compute_affordance(self.current_task, track.class_name)
            instant_score = track.score * prio

            # Exponential Moving Average (EMA) smoothing across video stream
            if track.smoothed_affordance == 0.0:
                track.smoothed_affordance = instant_score
            else:
                track.smoothed_affordance = (
                    self.alpha_smoothing * instant_score
                    + (1.0 - self.alpha_smoothing) * track.smoothed_affordance
                )

            track.affordance_priority = prio
            is_hazard = self.head.is_hazard(self.current_task, track.class_name)

            det_res = DetectionResult(
                bbox=track.bbox,
                confidence=track.score,
                class_id=track.class_id,
                class_name=track.class_name,
                affordance_priority=prio,
                final_score=track.smoothed_affordance,
                is_best=False,
                track_id=track.track_id,
                is_hazard=is_hazard,
            )
            results.append(det_res)

            if det_res.final_score > best_score:
                best_score = det_res.final_score
                best_candidate = det_res

        # Highlight best object for current task
        if best_candidate and best_candidate.final_score > 0.10:
            best_candidate.is_best = True

        return results
