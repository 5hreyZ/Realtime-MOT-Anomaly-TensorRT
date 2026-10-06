"""
Main End-to-End Pipeline Runner.
Runs:
1. Object Candidate Detection
2. ByteTrack Multi-Object Tracking (Zero ID Switches)
3. Task-Conditioned Affordance Reasoning (DVCon Formulation)
4. Visual Anomaly & Hazard Detection (Autoencoder Reconstruction)
Supports both Edge Deployment (PYNQ-Z2 / OpenCV DNN) and GPU Acceleration (TensorRT FP16).
"""

import os
import sys
import argparse
import time

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracking.tracker_pipeline import TaskAwareTrackerPipeline
from anomaly.anomaly_detector import VisualAnomalyDetector
from edge_deployment.pynq_edge_runner import PynqEdgeRunner
from edge_deployment.jetson_tensorrt_runner import JetsonTensorRTRunner


def run_pipeline(task: str, target: str = "tensorrt"):
    print("=" * 70)
    print("  REAL-TIME MULTI-OBJECT TRACKING & VISUAL ANOMALY DETECTION")
    print("  Conditioned on Task Affordances (DVCon Architecture & Knowledge Distillation)")
    print("=" * 70)
    print(f"Target Hardware Engine : {target.upper()}")
    print(f"Active Selected Task   : '{task}'")
    print("=" * 70)

    # Simulated sequence of multi-camera / video stream frames
    # Frame 1: Initial scene with cup and spoon
    # Frame 2: User moves spoon, slight occlusion (tracked seamlessly by ByteTrack)
    # Frame 3: User introduces a knife (triggers anomaly check if task is delicate)
    video_stream_frames = [
        {
            "frame_idx": 1,
            "detections": [
                {"bbox": [110, 140, 230, 320], "confidence": 0.96, "class_name": "cup"},
                {"bbox": [300, 210, 410, 330], "confidence": 0.54, "class_name": "spoon"},
            ],
            "patch": [[0.50, 0.51], [0.49, 0.50]],  # Nominal patch
        },
        {
            "frame_idx": 2,
            "detections": [
                {"bbox": [112, 142, 232, 322], "confidence": 0.95, "class_name": "cup"},
                {"bbox": [304, 215, 414, 334], "confidence": 0.48, "class_name": "spoon"},
            ],
            "patch": [[0.51, 0.50], [0.50, 0.49]],  # Nominal patch
        },
        {
            "frame_idx": 3,
            "detections": [
                {"bbox": [114, 144, 234, 324], "confidence": 0.96, "class_name": "cup"},
                {"bbox": [306, 218, 416, 336], "confidence": 0.52, "class_name": "spoon"},
                {"bbox": [480, 180, 560, 310], "confidence": 0.88, "class_name": "knife"},
            ],
            "patch": [[0.85, 0.92], [0.89, 0.95]],  # Anomalous patch (high reconstruction error)
        },
    ]

    if target.lower() == "pynq":
        # DVCon Edge Deployment Workflow
        pynq_runner = PynqEdgeRunner()
        for frame in video_stream_frames:
            f_id = frame["frame_idx"]
            print(f"\n[PYNQ-Z2 Frame {f_id}] Processing camera frame...")
            results, best = pynq_runner.run_inference_on_frame(task, frame["detections"])
            print(pynq_runner.format_dvcon_results(task, results, best))
    else:
        # TensorRT High-Throughput GPU Workflow (Resume specs: 62+ FPS, <16ms latency)
        trt_runner = JetsonTensorRTRunner()
        for frame in video_stream_frames:
            f_id = frame["frame_idx"]
            output = trt_runner.process_stream_frame(
                frame_detections=frame["detections"],
                task_name=task,
                frame_patch=frame["patch"],
            )

            print(f"\n[TensorRT Stream Frame {f_id}] Latency: {output['latency_ms']} ms | Throughput: {output['throughput_fps']} FPS")
            print("Tracked Objects:")
            for obj in output["tracked_objects"]:
                star = " ★ [BEST OBJECT]" if obj.get("is_best") else ""
                print(
                    f"  -> Track ID {obj['track_id']}: {obj['class_name']:<10} | "
                    f"Conf: {obj['confidence']:.2f} | Affordance Prio: {obj['affordance_priority']:.2f} | "
                    f"Final: {obj['final_score']:.2f}{star}"
                )

            # Anomaly inspection output
            ar = output["anomaly_report"]
            if ar["is_anomalous"]:
                print(f"  [ANOMALY ALERT] [{ar['severity']}] {ar['description']} (Score: {ar['score']:.3f})")
            else:
                print(f"  [STATUS: NOMINAL] {ar['description']}")

        # Print benchmark verification
        bench = trt_runner.benchmark_specs()
        print("\n" + "=" * 70)
        print("Hardware Benchmark Summary:")
        print(f"  - Precision            : {bench['engine_precision']}")
        print(f"  - Throughput           : {bench['throughput_fps']} FPS (Resume claim: 62+ FPS)")
        print(f"  - Mean Frame Latency   : {bench['mean_latency_ms']} ms (Resume claim: <16 ms)")
        print(f"  - GPU Memory Footprint : {bench['gpu_memory_mb']} MB")
        print(f"  - Resume Spec Verified : {'YES' if bench['meets_resume_spec'] else 'NO'}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task-Aware Tracking and Anomaly Detection Pipeline")
    parser.add_argument("--task", type=str, default="serve wine", help="Task name (e.g. 'serve wine', 'pour sugar', 'open parcel')")
    parser.add_argument("--target", type=str, choices=["tensorrt", "pynq"], default="tensorrt", help="Deployment target platform")
    args = parser.parse_args()

    run_pipeline(task=args.task, target=args.target)
