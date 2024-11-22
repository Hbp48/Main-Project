import cv2
import torch
import numpy as np
from time import time
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        # Initialize MultiTracker without calling MultiTracker_create()
        self.multi_tracker = cv2.MultiTracker()

    def load_model(self):
        model = YOLO("yolov10n.pt")
        model.fuse()
        return model

    @torch.no_grad()
    def predict(self, frame):
        results = self.model(frame)
        return results

    def draw_results(self, frame, results):
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf.item()
                class_id = int(box.cls.cpu().numpy())
                label = f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
                
                # Draw bounding box and label with object name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, self.CLASS_NAMES_DICT[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add tracker for each detected object
                tracker = cv2.TrackerCSRT_create()
                self.multi_tracker.add(tracker, frame, (x1, y1, x2 - x1, y2 - y1))
                
                # Add detection to the list
                detections.append((x1, y1, x2 - x1, y2 - y1))
        return frame, detections


    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame, detections = self.draw_results(frame, results)
            
            # Update trackers
            success, boxes = self.multi_tracker.update(frame)
            for i, newbox in enumerate(boxes):
                x, y, w, h = map(int, newbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display FPS
            fps = int(1 / (time() - start_time))
            cv2.putText(frame, f"FPS: {fps}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLO DETECTION', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# Instantiate and run detector
detector = ObjectDetection(capture_index=0)
detector()
