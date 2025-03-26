
import os
import json
import yaml
import cv2
import gi
from ultralytics import YOLO

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

# Path to BDD100K dataset video and YOLOv8 models
VIDEO_PATH = "sample_video_1.mp4"
MODEL_PATHS = {
    'yolov8n': "path_to_models/yolov8n.pt",
    'yolov8s': "best.pt",
    'yolov8m': "path_to_models/yolov8m.pt",
    'yolov8l': "path_to_models/yolov8l.pt",
    'yolov8x': "path_to_models/yolov8x.pt"
}

# Select model for inference
MODEL_TO_USE = 'yolov8s'  # Change this to test other models


def load_class_list(yaml_path="data.yaml"):
    """Load class list from YAML file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('classes', {})


CLASS_LIST = load_class_list()

# Load selected YOLO model
yolo_model = YOLO(MODEL_PATHS[MODEL_TO_USE])


def run_inference_video(video_path):
    """Run inference on the provided video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    while cap.isOpened():
        ret, actual_frame = cap.read()
        frame = cv2.resize(actual_frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        if not ret:
            break

        results = yolo_model(frame)[0]  # Access the first result directly
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = CLASS_LIST.get(class_id, "Unknown")
            label = f"{class_name})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize the output frame
        final_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("BDD100K Video Inference", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run inference on video
run_inference_video(VIDEO_PATH)

print("Inference completed for video.")
