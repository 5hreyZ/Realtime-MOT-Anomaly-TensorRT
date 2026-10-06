"""
Task-Conditioned Object Detector Architecture
Supports lightweight student detection (YOLOv10-style backbone) with task-aware conditioning.
Trained via Knowledge Distillation from a heavy Parent/Teacher model.
"""

from typing import List, Dict, Tuple, Optional, Any
import math
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TaskConditionedHead:
    """
    Affordance reasoning head that conditions object proposals on a target task.
    Computes affordance suitability using learned task-object embeddings
    blended with semantic prior tables (from DVCon formulation).
    """

    def __init__(self, affordance_matrix_path: Optional[str] = None):
        self.affordance_priors: Dict[str, Dict[str, float]] = {}
        self.hazard_classes: Dict[str, List[str]] = {}
        self.tasks_info: Dict[str, Dict[str, Any]] = {}
        if affordance_matrix_path and os.path.exists(affordance_matrix_path):
            self.load_affordance_matrix(affordance_matrix_path)

    def load_affordance_matrix(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        tasks = data.get("tasks", {})
        for tid, tinfo in tasks.items():
            tname = tinfo.get("task_name", "").lower()
            self.affordance_priors[tname] = tinfo.get("affordances", {})
            self.hazard_classes[tname] = tinfo.get("hazard_classes", [])
            self.tasks_info[tname] = tinfo
            # Also index by string task_id
            self.affordance_priors[str(tid)] = tinfo.get("affordances", {})
            self.hazard_classes[str(tid)] = tinfo.get("hazard_classes", [])

    def compute_affordance(self, task_key: str, class_name: str) -> float:
        """
        Retrieves task affordance prior for a class. Defaults to 0.05 for unlisted items.
        """
        task_key_lower = str(task_key).lower()
        if task_key_lower in self.affordance_priors:
            return self.affordance_priors[task_key_lower].get(class_name, 0.05)
        # Search by partial match (e.g. 'serve wine' in tasks)
        for tname, affs in self.affordance_priors.items():
            if task_key_lower in tname or tname in task_key_lower:
                return affs.get(class_name, 0.05)
        return 0.05

    def is_hazard(self, task_key: str, class_name: str) -> bool:
        """
        Checks whether the selected object is a safety hazard for the task
        (e.g., scissors or knife when pouring sugar or serving wine).
        """
        task_key_lower = str(task_key).lower()
        hazards = self.hazard_classes.get(task_key_lower, [])
        return class_name in hazards


if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        """Standard Conv-BatchNorm-SiLU block used in modern YOLO backbones."""
        def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class TaskAwareDetectorNN(nn.Module):
        """
        PyTorch Neural Network for Task-Aware Detection.
        Contains:
        1. Lightweight feature extractor (Student Backbone)
        2. Detection Head (bounding box regression + 80 COCO classes)
        3. Task Conditioning Head (learned embedding projector mapping task_id to object affordance logits)
        """
        def __init__(self, num_classes: int = 80, num_tasks: int = 16, embed_dim: int = 64):
            super().__init__()
            self.num_classes = num_classes
            self.num_tasks = num_tasks
            self.embed_dim = embed_dim

            # Lightweight backbone (C1 -> C3 downsampling)
            self.c1 = ConvBlock(3, 32, k=3, s=2, p=1)     # 1/2
            self.c2 = ConvBlock(32, 64, k=3, s=2, p=1)    # 1/4
            self.c3 = ConvBlock(64, 128, k=3, s=2, p=1)   # 1/8
            self.c4 = ConvBlock(128, 256, k=3, s=2, p=1)  # 1/16

            # Feature pyramid bottleneck / neck
            self.neck = ConvBlock(256, 128, k=1, s=1, p=0)

            # Detection heads
            # Box regression: [dx, dy, dw, dh, objectness] = 5
            self.bbox_head = nn.Conv2d(128, 5, kernel_size=1)
            # Classification head: 80 COCO classes
            self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)

            # Task Conditioning Embedding:
            # Learned representation of tasks mapped into object class affordance space
            self.task_embedding = nn.Embedding(num_tasks, embed_dim)
            self.affordance_projector = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
                nn.Sigmoid()
            )

        def extract_features(self, x):
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            x = self.c4(x)
            neck_feat = self.neck(x)
            return neck_feat

        def forward(self, x, task_id: Optional[torch.Tensor] = None):
            """
            Args:
                x: Input tensor [B, 3, H, W]
                task_id: Optional tensor [B] indicating active task index
            Returns:
                dict containing:
                  'bbox': [B, 5, H', W']
                  'cls_logits': [B, num_classes, H', W']
                  'features': [B, 128, H', W']
                  'affordance_weights': [B, num_classes] if task_id provided else None
            """
            feat = self.extract_features(x)
            bbox_out = self.bbox_head(feat)
            cls_out = self.cls_head(feat)

            affordance_weights = None
            if task_id is not None:
                task_emb = self.task_embedding(task_id)  # [B, embed_dim]
                affordance_weights = self.affordance_projector(task_emb)  # [B, num_classes]

            return {
                "bbox": bbox_out,
                "cls_logits": cls_out,
                "features": feat,
                "affordance_weights": affordance_weights,
            }
else:
    TaskAwareDetectorNN = None


class DetectionResult:
    """Represents a single detected & task-evaluated object."""
    def __init__(
        self,
        bbox: List[float],             # [x1, y1, x2, y2]
        confidence: float,            # Vision confidence score
        class_id: int,                # COCO class index
        class_name: str,              # Label string ('cup', 'wine glass', etc.)
        affordance_priority: float,   # Priority from affordance matrix
        final_score: float,           # confidence * affordance_priority
        is_best: bool = False,
        track_id: Optional[int] = None,
        is_hazard: bool = False,
    ):
        self.bbox = [round(float(v), 2) for v in bbox]
        self.confidence = round(float(confidence), 4)
        self.class_id = int(class_id)
        self.class_name = class_name
        self.affordance_priority = round(float(affordance_priority), 4)
        self.final_score = round(float(final_score), 4)
        self.is_best = is_best
        self.track_id = track_id
        self.is_hazard = is_hazard

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "affordance_priority": self.affordance_priority,
            "final_score": self.final_score,
            "is_best": self.is_best,
            "track_id": self.track_id,
            "is_hazard": self.is_hazard,
        }

    def __repr__(self) -> str:
        tid_str = f" ID:{self.track_id}" if self.track_id is not None else ""
        best_str = " [BEST]" if self.is_best else ""
        hazard_str = " [HAZARD!]" if self.is_hazard else ""
        return (
            f"<{self.class_name}{tid_str}{best_str}{hazard_str} | "
            f"conf={self.confidence:.2f} | prio={self.affordance_priority:.2f} | "
            f"final={self.final_score:.2f} | box={self.bbox}>"
        )


class TaskAwareDetector:
    """
    High-level Task-Aware Object Detector.
    Integrates the vision detector with semantic affordance priors and task conditioning.
    Designed for seamless execution across:
    1. Edge FPGA / ARM CPU (PYNQ-Z2) via OpenCV DNN / ONNX
    2. NVIDIA GPUs via TensorRT FP16 / PyTorch
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        affordance_matrix_path: Optional[str] = None,
        coco_labels_path: Optional[str] = None,
    ):
        self.config = config or {}
        aff_path = (
            affordance_matrix_path
            or self.config.get("task_reasoning", {}).get("affordance_matrix_path")
            or "configs/affordance_matrix.json"
        )
        labels_path = (
            coco_labels_path
            or self.config.get("detection", {}).get("classes_file")
            or "data/coco_labels.txt"
        )

        self.head = TaskConditionedHead(aff_path)
        self.classes = self._load_classes(labels_path)
        self.conf_threshold = self.config.get("detection", {}).get("conf_threshold", 0.25)
        self.nn_model = None

        if TORCH_AVAILABLE:
            self.nn_model = TaskAwareDetectorNN(num_classes=len(self.classes))
            self.nn_model.eval()

    def _load_classes(self, path: str) -> List[str]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        # Default fallback standard classes
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def reason_affordances(
        self,
        raw_detections: List[Dict[str, Any]],
        task_name: str,
    ) -> List[DetectionResult]:
        """
        DVCon Stage 2(A) Affordance Reasoning:
        Multiplies vision detection confidence by the predefined affordance priority.
        Identifies the candidate with highest final score as 'BEST OBJECT'.
        """
        evaluated: List[DetectionResult] = []

        for det in raw_detections:
            box = det["bbox"]
            conf = det["confidence"]
            cls_name = det.get("class_name")
            cls_id = det.get("class_id")

            if cls_name is None and cls_id is not None and cls_id < len(self.classes):
                cls_name = self.classes[cls_id]
            elif cls_name is not None and cls_id is None:
                cls_id = self.classes.index(cls_name) if cls_name in self.classes else 0

            priority = self.head.compute_affordance(task_name, cls_name)
            final_score = conf * priority
            is_hazard = self.head.is_hazard(task_name, cls_name)

            evaluated.append(
                DetectionResult(
                    bbox=box,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    affordance_priority=priority,
                    final_score=final_score,
                    is_best=False,
                    is_hazard=is_hazard,
                )
            )

        # Highlight best object for the given task
        if evaluated:
            best_det = max(evaluated, key=lambda d: d.final_score)
            if best_det.final_score > 0.10:
                best_det.is_best = True

        return evaluated
