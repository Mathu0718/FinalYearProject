"""
Phase 2 – Hazard Detection Module (YOLOv8).
Loads a pretrained YOLOv8 model, runs inference on frames,
and draws bounding boxes with labels and confidence.
Can be restricted to road signals only (traffic lights, stop signs).
"""

import cv2
from ultralytics import YOLO

# COCO class IDs for road-side signals/signs (YOLOv8 pretrained on COCO)
# 9 = traffic light, 11 = stop sign
ROAD_SIGN_COCO_IDS = [9, 11]


class HazardDetector:
    """YOLOv8-based object detector for road hazards (pretrained COCO)."""

    def __init__(self, model_path: str = "yolov8n.pt", sign_only: bool = True):
        """
        Load pretrained or custom YOLOv8 model.
        sign_only: only for default COCO model – restrict to traffic light, stop sign.
        If model_path is a custom trained model (e.g. best.pt), all model classes are used.
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
        is_custom = "yolov8n.pt" not in model_path and "yolov8s.pt" not in model_path
        self.sign_only = sign_only and not is_custom
        self.classes = ROAD_SIGN_COCO_IDS if self.sign_only else None

    def detect(self, frame, conf_threshold: float = 0.25):
        """
        Run detection on a single BGR frame.
        Returns list of detections (only road signs if sign_only=True).
        """
        results = self.model(
            frame,
            conf=conf_threshold,
            classes=self.classes,
            verbose=False,
        )
        detections = []
        names = self.model.names

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names.get(cls_id, str(cls_id))
                detections.append({
                    "bbox": xyxy,
                    "class_id": cls_id,
                    "label": label,
                    "confidence": conf,
                })

        return detections

    def detect_and_draw(self, frame, conf_threshold: float = 0.25):
        """
        Run detection on frame and draw bounding boxes, labels, and confidence.
        Modifies frame in-place and returns it plus the list of detections.
        """
        detections = self.detect(frame, conf_threshold=conf_threshold)
        out = frame.copy()

        for d in detections:
            x1, y1, x2, y2 = d["bbox"].astype(int)
            label = d["label"]
            conf = d["confidence"]
            color = (0, 255, 0)  # BGR green
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                out, text, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return out, detections


def run_video(
    video_source: int | str = 0,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    sign_only: bool = True,
    headless: bool = False,
    output_path: str | None = None,
):
    """
    Run hazard detection on a video source: webcam (int) or video file (path str).
    By default shows live in a window. If OpenCV has no GUI, use headless=True to write to a file.
    """
    detector = HazardDetector(model_path=model_path, sign_only=sign_only)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        if video_source == 0:
            raise RuntimeError(
                "Could not open webcam (index 0). No camera detected or in use elsewhere.\n"
                "Run with a video file instead: python scripts/run_hazard_detection.py path/to/video.mp4"
            )
        raise RuntimeError(
            f"Could not open video source: {video_source}. Check path or camera index."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if headless:
        out_path = output_path or "output_detection.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {out_path}")
        print("Writing output to:", out_path)

    kind = "webcam" if isinstance(video_source, int) else f"file: {video_source}"
    mode = "road signals only (traffic light, stop sign)" if detector.sign_only else "custom/all sign classes"
    print("Hazard Detection (YOLOv8) –", kind)
    print("Mode:", mode)
    print("Using model:", model_path)
    if not headless:
        print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out, _ = detector.detect_and_draw(frame, conf_threshold=conf_threshold)

            if headless and writer is not None:
                writer.write(out)
            else:
                try:
                    cv2.imshow("Hazard Detection", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    if "not implemented" in str(e).lower() or "cvShowImage" in str(e):
                        raise RuntimeError(
                            "Display not available (OpenCV has no GUI). "
                            "Run with --headless to write output to a video file you can play."
                        ) from e
                    raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def run_webcam(model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
    """Run hazard detection on default webcam (index 0)."""
    run_video(0, model_path=model_path, conf_threshold=conf_threshold)


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--all"]
    sign_only = "--all" not in sys.argv
    source = args[0] if args else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    run_video(source, sign_only=sign_only)
