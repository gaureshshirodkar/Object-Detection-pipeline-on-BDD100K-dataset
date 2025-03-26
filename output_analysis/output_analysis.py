import json
import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from Testing_output_analysis.object_detection_confusion_matrix.object_detection_confusion_matrix import (
    ObjectDetectionConfusionMatrix,
)


DEBUG = False
CLASS_LIST_ID = {
    "bus": 0,
    "traffic light": 1,
    "traffic sign": 2,
    "person": 3,
    "bike": 4,
    "truck": 5,
    "motor": 6,
    "car": 7,
    "train": 8,
    "rider": 9,
}

def draw_boxes(image, boxes, classes, color, label):
    """
    Draws bounding boxes on the image with class labels.

    Args:
        image (np.ndarray): The input image on which boxes are drawn.
        boxes (List[List[float]]): List of bounding box coordinates.
        classes (List[str]): List of class labels corresponding to the boxes.
        color (Tuple[int, int, int]): RGB color for the boxes.
        label (str): Label indicating whether boxes are for 'GT' or 'Pred'.

    Returns:
        np.ndarray: The image with drawn boxes.
    """

    pred_count = 0

    for box, cls in zip(boxes, classes):
        pred_count += 1

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if label == "GT" else y1 - 30
        cv2.putText(
            image,
            f"{cls}",
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


def load_ground_truth(file_path):
    """
    Loads ground truth data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing ground truth data.

    Returns:
        Dict: The parsed JSON data as a dictionary.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def load_class_list(yaml_path="data.yaml"):
    """
    Loads class list from a YAML file.

    Args:
        yaml_path (str): Path to the YAML file containing class list data.

    Returns:
        Dict[str, int]: A dictionary mapping class names to class IDs.
    """
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data.get("classes", {})


# Plot aggregated metrics (map and mar)
def plot_aggregated_metrics(metrics):
    """
    Plots aggregated metrics for object detection performance.

    Args:
        metrics (Dict[str, float]): A dictionary containing various mAP and mAR metrics.

    Returns:
        None
    """

    metric_keys = [
        'map', 'map_50', 'map_75', 'map_large', 'map_medium', 'map_small',
        'mar_1', 'mar_10', 'mar_100', 'mar_large', 'mar_medium', 'mar_small'
    ]

    labels = []
    values = []
    for i, key in enumerate(metric_keys):
        labels.append(key.replace('_', ' ').upper())
        values.append(metrics[key].item())

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Metric Values')
    plt.title('Aggregated Metrics (mAP and mAR)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# Plot per-class metrics
def plot_per_class_metrics(classes, map_per_class, mar_100_per_class):
    """
    Plots per-class metrics (mAP and mAR_100) for object detection performance.

    Args:
        classes (List[str]): List of class names.
        map_per_class (List[float]): List of mean average precision (mAP) values per class.
        mar_100_per_class (List[float]): List of mean average recall (mAR_100) values per class.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    x = range(len(classes))

    plt.bar(x, map_per_class, width=0.4, label='mAP per class', color='orange')
    plt.bar([i + 0.4 for i in x], mar_100_per_class, width=0.4, label='mAR_100 per class', color='purple')

    plt.xticks([i + 0.2 for i in x], classes, rotation=45)
    plt.xlabel('Classes')
    plt.ylabel('Metric Values')
    plt.title('Per-Class Metrics (mAP and mAR_100)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def main():
    # Load trained YOLOv8 model
    model_path = "best.pt"
    model = YOLO(model_path)

    ground_truths = (
        load_ground_truth(
            "C:\\Personal\\Bosch_assignment\\assignment_data_bdd_files\\bdd100k_labels_release\\bdd100k\\labels\\bdd100k_labels_images_val.json"
        )
        if not DEBUG
        else load_ground_truth(
            "C:\\Personal\\Bosch_assignment\\assignment_data_bdd_files\\bdd100k_labels_release\\bdd100k\\test_json_1.json"
        )
    )

    results = (
        model.predict(
            "C:\\Personal\\Bosch_assignment\\assignment_data_bdd_files\\bdd100k_images_100k\\bdd100k\\images\\100k\\val",
            save=False,
        )
        if not DEBUG
        else model.predict(
            "C:\\Personal\\Bosch_assignment\\assignment_data_bdd_files\\bdd100k_images_100k\\bdd100k\\images\\100k\\test_image_1",
            save=False,
        )
    )
    preds_test_list = []
    target_test_list = []

    class_list = load_class_list()

    test_count = 0

    for result, gts in zip(results, ground_truths):
        preds = result.boxes.xyxy.cpu().numpy()

        pred_classes = [class_list.get(int(box.cls[0]), "Unknown") for box in result.boxes]
        pred_classes_ids = [int(box.cls[0]) for box in result.boxes]
        conf_list = [box.conf[0] for box in result.boxes]

        gts_list, gts_classes, gts_classes_ids = [], [], []
        image_path = result.path
        image = cv2.imread(image_path)

        for label in gts["labels"]:
            if "box2d" in label:
                gts_list.append(
                    [
                        label["box2d"]["x1"],
                        label["box2d"]["y1"],
                        label["box2d"]["x2"],
                        label["box2d"]["y2"],
                    ]
                )
                gts_classes.append(label["category"])
                gts_classes_ids.append(CLASS_LIST_ID[label["category"]])

        preds_test_list.append(
            {
                "boxes": torch.tensor(preds),
                "labels": torch.tensor(pred_classes_ids),
                "scores": torch.tensor(conf_list),
            }
        )

        target_test_list.append(
            {"boxes": torch.tensor(gts_list), "labels": torch.tensor(gts_classes_ids)}
        )

        if DEBUG:
            image = draw_boxes(image, preds, pred_classes, (0, 0, 255), "Pred")
            image = draw_boxes(image, gts_list, gts_classes, (0, 255, 0), "GT")

            cv2.imshow("Detections", image)
            cv2.waitKey(0)
        
        test_count += 1
        print(f"Calculating metric for image number {test_count}. Images remaining to be processed {len(results) - test_count}")

    map_metric = MeanAveragePrecision(
        iou_thresholds=[0.1, 0.25, 0.5, 0.75, 0.95], class_metrics=True
    )
    map_metric.update(preds_test_list, target_test_list)

    metrics = map_metric.compute()
    print(metrics)

    classes = metrics["classes"].tolist()
    map_per_class = metrics["map_per_class"].tolist()
    mar_100_per_class = metrics["mar_100_per_class"].tolist()

    plot_aggregated_metrics(metrics)
    plot_per_class_metrics(classes, map_per_class, mar_100_per_class)


if __name__ == "__main__":
    main()