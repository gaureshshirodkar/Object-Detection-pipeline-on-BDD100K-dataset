
import torch
from ultralytics import YOLO
import cv2
import os

if __name__ == "__main__":
    # Initialize YOLOv8 model for transfer learning
    model = YOLO('yolov8s.pt')  # Pre-trained YOLOv8n weights

    # Path to dataset configuration file (BDD100K format)
    DATASET_YAML = "data.yaml"

    # Transfer learning parameters
    TRAINING_PARAMS = {
        'imgsz': 640,       # Input image size
        'batch': 16,        # Batch size
        'epochs': 50,       # Number of epochs
        'data': DATASET_YAML,  # Dataset YAML file path
        'model': 'yolov8n.pt', # Pre-trained YOLOv8 weights for transfer learning
        'project': 'bdd100k_training', # Directory to save results
        'name': 'yolov8n_bdd100k_transfer' # Experiment name for transfer learning
        # 'device': 0  # Use all available GPUs
        # 'workers': 2 # Create multiple threads
    }

    # Train the model with transfer learning
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU...")
        model.train(**TRAINING_PARAMS)
    else:
        print("CUDA is not available. Training on CPU...")
        model.train(**TRAINING_PARAMS, device='cpu')

    # Save the trained model
    model.save('bdd100k_trained_yolov8n.pt')

    print("Transfer learning completed and model saved.")

    # Load the trained model for inference
    test_model = YOLO('bdd100k_trained_yolov8s.pt')

    # Path to test images directory
    TEST_IMAGES_PATH = 'C:\Personal\Bosch_assignment\assignment_data_bdd_files\bdd100k_images_100k\bdd100k\images\100k\test'

    def run_inference_on_test_images():
        for image_name in os.listdir(TEST_IMAGES_PATH):
            image_path = os.path.join(TEST_IMAGES_PATH, image_name)
            image = cv2.imread(image_path)
            results = test_model(image)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = f"Class {class_id}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Test Image Inference", image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    run_inference_on_test_images()
    print("Inference on test images completed.")

