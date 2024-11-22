from ultralytics import YOLO
import torch
import numpy as np
import cv2
from time import time
from pathlib import Path
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort  # Adjust import as necessary
from strongsort.strong_sort import StrongSORT

SAVE_VIDEO = False
TRACKER = "deepsort"  # Change tracker to "deepsort"

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Load YOLO model
        self.model = self.load_model()

        # Initialize class names and annotator
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

        # Initialize tracker
        self.initialize_tracker()

    def load_model(self):
        model = YOLO("yolov10n.pt")
        model.fuse()  # Use fused model for optimized inference
        return model

    def initialize_tracker(self):
        if TRACKER == "deepsort":
            # Inline configuration for DeepSORT
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nn_budget=100,
                max_iou_distance=0.7,
                max_cosine_distance=0.2
            )
        else:
            # Inline configuration for StrongSORT
            self.tracker = StrongSORT(
                reid_weights,
                torch.device(self.device),
                False,
                max_dist=0.2,
                max_iou_dist=0.7,
                max_age=30,
                max_unmatched_preds=5,
                n_init=3,
                nn_budget=100,
                mc_lambda=0.995,
                ema_alpha=0.9,
            )

    @torch.no_grad()
    def predict(self, frame):
        results = self.model(frame)
        return results

    def draw_results(self, frame, results):
        detections = []
        for result in results:
            xyxys = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            detections = sv.Detections(
                xyxy=xyxys,
                confidence=confidences,
                class_id=class_ids,
            )

            labels = [f"{self.CLASS_NAMES_DICT[class_id]} {conf:0.2f}" for class_id, conf in zip(class_ids, confidences)]
            frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame, detections

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Failed to open video capture."

        if SAVE_VIDEO:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            outputvid = cv2.VideoWriter('result_tracking.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (width, height))

        while cap.isOpened():
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                break

            # Predict and draw results on the frame
            results = self.predict(frame)
            frame, detections = self.draw_results(frame, results)

            # Update the tracker with detected objects
            tracked_objects = self.tracker.update(detections.xyxy, frame)
            for obj in tracked_objects:
                bbox, tracked_id = obj[:4], obj[4]
                cv2.putText(frame, f"ID: {tracked_id}", (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display FPS
            fps = 1 / (time() - start_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('YOLO Detection', frame)

            # Save frame if saving video
            if SAVE_VIDEO:
                outputvid.write(frame)

            # Press 'ESC' to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # Release resources
        if SAVE_VIDEO:
            outputvid.release()
        cap.release()
        cv2.destroyAllWindows()

# Initialize and run the object detection
detector = ObjectDetection(capture_index=0)
detector()
