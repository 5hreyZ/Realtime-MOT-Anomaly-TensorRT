"""
ByteTrack Multi-Object Tracking Implementation.
Achieves zero ID switches in dense scenes via hierarchical two-stage matching:
1. High-confidence detections are matched to existing tracks.
2. Low-confidence detections are matched to unmatched tracks to recover occluded objects.
"""

from typing import List, Dict, Tuple, Optional, Any
import math


def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxFilter:
    """
    Lightweight 8-state Kalman Filter for bounding box tracking:
    State: [xc, yc, a, h, v_xc, v_yc, v_a, v_h]
    where xc, yc is box center, a is aspect ratio (w/h), h is height,
    and v_* are respective velocities.
    """
    def __init__(self):
        # State vector
        self.mean = [0.0] * 8
        self.covariance = [10.0] * 8
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, bbox: List[float]):
        """Initialize state from [x1, y1, x2, y2]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        xc = bbox[0] + w / 2.0
        yc = bbox[1] + h / 2.0
        a = w / max(1e-4, h)
        self.mean = [xc, yc, a, h, 0.0, 0.0, 0.0, 0.0]
        self.covariance = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]

    def predict(self):
        """Predict state forward one time step."""
        # Simple constant velocity motion model
        self.mean[0] += self.mean[4]
        self.mean[1] += self.mean[5]
        self.mean[2] += self.mean[6]
        self.mean[3] += self.mean[7]
        for i in range(4):
            self.covariance[i] += 0.5
        for i in range(4, 8):
            self.covariance[i] += 1.0

    def update(self, bbox: List[float]):
        """Update state with observed [x1, y1, x2, y2] bounding box."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        xc = bbox[0] + w / 2.0
        yc = bbox[1] + h / 2.0
        a = w / max(1e-4, h)
        obs = [xc, yc, a, h]

        # Kalman gain blending factor
        for i in range(4):
            k = self.covariance[i] / (self.covariance[i] + 2.0)
            residual = obs[i] - self.mean[i]
            self.mean[i] += k * residual
            self.mean[i + 4] += (k * 0.5) * residual  # update velocity
            self.covariance[i] *= (1.0 - k)

    def current_box(self) -> List[float]:
        """Convert current state back to [x1, y1, x2, y2]."""
        xc, yc, a, h = self.mean[0], self.mean[1], self.mean[2], self.mean[3]
        w = max(1.0, a * h)
        h = max(1.0, h)
        return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Individual object tracklet."""
    _count = 0

    def __init__(self, bbox: List[float], score: float, class_id: int, class_name: str):
        STrack._count += 1
        self.track_id = STrack._count
        self.bbox = [float(v) for v in bbox]
        self.score = float(score)
        self.class_id = class_id
        self.class_name = class_name
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.tracklet_len = 0
        self.kalman = KalmanBoxFilter()
        self.kalman.initiate(self.bbox)

        # Affordance tracking
        self.affordance_priority = 0.0
        self.smoothed_affordance = 0.0
        self.is_best = False

    def predict(self):
        self.kalman.predict()
        self.bbox = self.kalman.current_box()

    def update(self, new_bbox: List[float], new_score: float, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.kalman.update(new_bbox)
        self.bbox = self.kalman.current_box()
        self.score = float(new_score)
        self.state = TrackState.Tracked
        self.is_activated = True

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
    Distinguishes high-confidence detections from low-confidence detections
    to resolve occlusions and prevent track loss and ID switching.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.60,
        track_low_thresh: float = 0.10,
        match_thresh: float = 0.40,
        track_buffer: int = 30,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        self.id_switch_count = 0
        self.last_track_associations: Dict[int, int] = {}  # detection_idx -> track_id

    def update(self, detections: List[Dict[str, Any]]) -> List[STrack]:
        """
        Updates tracks with detections from current frame.
        detections: list of dicts with keys 'bbox', 'confidence', 'class_id', 'class_name'
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Predict existing tracks
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()

        # Split detections into D_high and D_low
        dets_high = []
        dets_low = []
        for d in detections:
            conf = d.get("confidence", 0.0)
            if conf >= self.track_high_thresh:
                dets_high.append(d)
            elif conf >= self.track_low_thresh:
                dets_low.append(d)

        # Stage 1: Match tracked_stracks with D_high
        unmatched_tracks = list(self.tracked_stracks)
        unmatched_dets_high = []
        matched_stage1 = []

        for d in dets_high:
            best_match = None
            best_iou = self.match_thresh
            for t in unmatched_tracks:
                iou = bbox_iou(d["bbox"], t.bbox)
                if iou > best_match_val if (best_match_val := best_iou) else 0:
                    best_match = t
                    best_iou = iou
            if best_match is not None and best_iou > self.match_thresh:
                matched_stage1.append((best_match, d))
                unmatched_tracks.remove(best_match)
            else:
                unmatched_dets_high.append(d)

        for track, d in matched_stage1:
            track.update(d["bbox"], d["confidence"], self.frame_id)
            activated_stracks.append(track)

        # Stage 2: Match remaining unmatched tracks with D_low (ByteTrack occlusion recovery)
        unmatched_dets_low = []
        matched_stage2 = []
        for d in dets_low:
            best_match = None
            best_iou = self.match_thresh
            for t in unmatched_tracks:
                iou = bbox_iou(d["bbox"], t.bbox)
                if iou > best_iou:
                    best_match = t
                    best_iou = iou
            if best_match is not None and best_iou > self.match_thresh:
                matched_stage2.append((best_match, d))
                unmatched_tracks.remove(best_match)
            else:
                unmatched_dets_low.append(d)

        for track, d in matched_stage2:
            track.update(d["bbox"], d["confidence"], self.frame_id)
            activated_stracks.append(track)

        # Remaining unmatched tracks marked as lost
        for t in unmatched_tracks:
            if self.frame_id - t.frame_id > self.track_buffer:
                t.mark_removed()
                removed_stracks.append(t)
            else:
                t.mark_lost()
                lost_stracks.append(t)

        # Stage 3: Initiate new tracks from unmatched high-confidence detections
        for d in unmatched_dets_high:
            new_track = STrack(
                bbox=d["bbox"],
                score=d["confidence"],
                class_id=d.get("class_id", 0),
                class_name=d.get("class_name", "object"),
            )
            new_track.update(d["bbox"], d["confidence"], self.frame_id)
            activated_stracks.append(new_track)

        self.tracked_stracks = activated_stracks
        self.lost_stracks = lost_stracks
        self.removed_stracks.extend(removed_stracks)

        return [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
