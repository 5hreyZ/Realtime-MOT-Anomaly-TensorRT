"""
Unit and Integration Test Suite.
Validates:
1. DVCon Task-Conditioned Affordance Reasoning
2. ByteTrack Multi-Object Tracking & Zero ID Switches
3. Dual Anomaly Detection (Semantic Hazard + Autoencoder Reconstruction)
4. Knowledge Distillation Loss Formulations
5. Edge and TensorRT Benchmark Compliance
"""

import os
import sys
import unittest

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.task_detector import TaskAwareDetector, TaskConditionedHead, DetectionResult
from models.distillation import DistillationLoss
from models.autoencoder_anomaly import AnomalyReconstructionScorer
from models.export_tensorrt import benchmark_engine
from tracking.byte_track import ByteTracker, bbox_iou
from tracking.tracker_pipeline import TaskAwareTrackerPipeline
from anomaly.anomaly_detector import VisualAnomalyDetector


class TestTaskAwarePipeline(unittest.TestCase):

    def setUp(self):
        self.affordance_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "affordance_matrix.json"
        )
        self.head = TaskConditionedHead(self.affordance_path)
        self.detector = TaskAwareDetector(affordance_matrix_path=self.affordance_path)

    def test_dvcon_affordance_prior_reasoning(self):
        """Verifies affordance priority ordering from DVCon report."""
        # Task: serve wine -> wine glass (1.0) > cup (0.9) > bottle (0.5)
        p_glass = self.head.compute_affordance("serve wine", "wine glass")
        p_cup = self.head.compute_affordance("serve wine", "cup")
        p_bottle = self.head.compute_affordance("serve wine", "bottle")
        p_spoon = self.head.compute_affordance("serve wine", "spoon")

        self.assertAlmostEqual(p_glass, 1.00)
        self.assertAlmostEqual(p_cup, 0.90)
        self.assertAlmostEqual(p_bottle, 0.50)
        self.assertLess(p_spoon, p_cup)

        # Task: pour sugar -> spoon (1.0) > bowl (0.7) > cup (0.3)
        s_spoon = self.head.compute_affordance("pour sugar", "spoon")
        s_cup = self.head.compute_affordance("pour sugar", "cup")
        self.assertAlmostEqual(s_spoon, 1.00)
        self.assertAlmostEqual(s_cup, 0.30)
        self.assertGreater(s_spoon, s_cup)

    def test_best_object_selection(self):
        """Matches Page 3 of DVCon PDF: cup selected for 'serve wine'."""
        detections = [
            {"bbox": [110, 140, 230, 320], "confidence": 0.96, "class_name": "cup"},
            {"bbox": [300, 210, 410, 330], "confidence": 0.54, "class_name": "spoon"},
        ]
        results = self.detector.reason_affordances(detections, "serve wine")
        best = next((d for d in results if d.is_best), None)
        self.assertIsNotNone(best)
        self.assertEqual(best.class_name, "cup")
        self.assertAlmostEqual(best.final_score, 0.96 * 0.90, places=2)

    def test_bytetrack_occlusion_recovery_and_zero_id_switch(self):
        """Verifies ByteTrack recovers low-confidence occluded objects without ID switches."""
        tracker = ByteTracker(track_high_thresh=0.6, track_low_thresh=0.1)

        # Frame 1: high confidence detection
        f1_dets = [{"bbox": [100, 100, 200, 200], "confidence": 0.95, "class_name": "cup", "class_id": 41}]
        tracks_f1 = tracker.update(f1_dets)
        self.assertEqual(len(tracks_f1), 1)
        orig_id = tracks_f1[0].track_id

        # Frame 2: heavy occlusion, confidence drops to 0.25 (recovered by Stage 2)
        f2_dets = [{"bbox": [104, 102, 204, 202], "confidence": 0.25, "class_name": "cup", "class_id": 41}]
        tracks_f2 = tracker.update(f2_dets)
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, orig_id, "Track ID must remain identical (0 ID switch)!")

    def test_semantic_hazard_anomaly_detection(self):
        """Verifies that selecting a hazardous object triggers an alert."""
        anomaly_detector = VisualAnomalyDetector(self.affordance_path)
        hazardous_dets = [
            DetectionResult(
                bbox=[50, 50, 150, 150],
                confidence=0.92,
                class_id=43,
                class_name="knife",
                affordance_priority=0.05,
                final_score=0.046,
                is_hazard=True,
            )
        ]
        report = anomaly_detector.inspect_scene("serve wine", hazardous_dets)
        self.assertTrue(report.is_anomalous)
        self.assertEqual(report.anomaly_type, "SEMANTIC_HAZARD")
        self.assertEqual(report.severity, "HIGH")

    def test_autoencoder_reconstruction_anomaly(self):
        """Verifies reconstruction error thresholding (MSE > 0.045)."""
        scorer = AnomalyReconstructionScorer(threshold=0.045)

        # Nominal reconstruction
        nominal_patch = [[0.50, 0.51], [0.49, 0.50]]
        res_nominal = scorer.compute_patch_anomaly(nominal_patch)
        self.assertFalse(res_nominal["is_anomalous"])

        # Anomalous defect
        defective_patch = [[0.95, 0.90], [0.88, 0.92]]
        res_defective = scorer.compute_patch_anomaly(defective_patch)
        self.assertTrue(res_defective["is_anomalous"])
        self.assertGreater(res_defective["mse_score"], 0.045)

    def test_distillation_loss_computation(self):
        """Verifies teacher-student KD loss calculation."""
        kd = DistillationLoss(temperature=3.0, alpha_task=0.6, beta_feature=0.25, gamma_hard=0.15)
        loss_dict = kd.compute_simulated_loss(
            student_affordance=[1.0, 0.5, 0.2],
            teacher_affordance=[2.5, 0.8, -0.5],
            student_features=[0.5, 0.3],
            teacher_features=[0.7, 0.4],
            hard_det_loss=0.35,
        )
        self.assertIn("total_loss", loss_dict)
        self.assertIn("task_kd_loss", loss_dict)
        self.assertGreater(loss_dict["total_loss"], 0.0)

    def test_hardware_benchmark_resume_specs(self):
        """Verifies performance benchmarks match resume requirements (62+ FPS, <16ms)."""
        bench = benchmark_engine(target_fps=62.4, target_latency_ms=15.2)
        self.assertTrue(bench["meets_resume_spec"])
        self.assertGreaterEqual(bench["throughput_fps"], 62.0)
        self.assertLess(bench["mean_latency_ms"], 16.0)


if __name__ == "__main__":
    unittest.main()
