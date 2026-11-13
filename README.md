# Classroom Occupancy Monitoring with YOLO

An object-detection project that uses a **custom, self-collected classroom dataset** to detect people in CCTV-style images. The trained model recognizes two classes:

| Class ID | Class name |
|---:|---|
| 0 | `student` |
| 1 | `cleaner` |

The detector can be integrated into a classroom occupancy system and used as an input for automated air-conditioner control, monitoring, and reporting.

## Project workflow

```text
Own image/video collection
          ↓
Frame extraction and manual annotation
          ↓
YOLO-format dataset preparation
          ↓
Train / validation / test split
          ↓
Data augmentation to reduce overfitting
          ↓
YOLO11n training in Google Colab
          ↓
Validation, prediction, and application integration
```

## Dataset

This project does **not** use a ready-made public dataset. Images were collected from the project's own classroom/CCTV-style recordings and annotated for the required use case.

### Dataset summary

| Item | Count |
|---|---:|
| Original unique images | 960 |
| Original training images | 672 |
| Validation images | 192 |
| Test images | 96 |
| Augmented training images in this repository | 2,525 |
| `student` annotations | 6,668 |
| `cleaner` annotations | 76 |

> The class distribution is imbalanced. The smaller number of `cleaner` examples should be considered when interpreting per-class results and when collecting future data.

### Annotation

Objects were manually labelled with bounding boxes and exported in **YOLO detection format**. Each image has a matching `.txt` label file. Every row follows:

```text
class_id x_center y_center width height
```

All four box coordinates are normalized to the range `0–1`. Dataset class names and split paths are defined in [`dataset_augmented/data.yaml`](dataset_augmented/data.yaml).

Expected structure:

```text
dataset_augmented/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Data augmentation and overfitting

The original dataset is relatively small and comes from a limited number of classroom scenes. Training directly on only those images can cause **overfitting**: the model may memorize the training backgrounds, lighting, and camera positions instead of learning features that generalize to unseen footage.

Augmentation was therefore applied to the **training set only**. It introduces realistic variation through transformations such as:

- brightness, saturation, and hue changes;
- horizontal flipping;
- translation and scaling;
- mosaic composition;
- simulated changes in viewpoint, image quality, shadows, and object position.

The recorded training configuration uses `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `translate=0.1`, `scale=0.5`, `fliplr=0.5`, and `mosaic=1.0`. Validation and test data remain separate so that evaluation measures generalization rather than memorization. Augmentation helps reduce overfitting, but it does not replace collecting more varied real-world images—especially for the minority `cleaner` class.

## Model and training

- **Algorithm:** YOLO (You Only Look Once) object detection
- **Implementation:** Ultralytics YOLO
- **Model:** pretrained `YOLO11n` (`yolo11n.pt`)
- **Training platform:** Google Colab with GPU
- **Input size:** 640 × 640
- **Epochs:** 50
- **Batch size:** 8
- **Optimizer:** automatic selection by Ultralytics
- **Random seed:** 0

YOLO was selected because it performs object localization and classification in a single forward pass, making it suitable for near-real-time classroom occupancy applications.

### Google Colab training

1. Upload this project to Google Drive, or mount the Drive folder containing `dataset_augmented`.
2. In a Colab notebook, select **Runtime → Change runtime type → GPU**.
3. Run:

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q ultralytics

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="/content/drive/MyDrive/myDataSet/dataset_augmented/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="/content/drive/MyDrive/myDataSet/training_results",
    name="classroom_test_run",
    seed=0,
)
```

If the Drive folder name is different, update the paths in both `data.yaml` and the training command.

## Results

The final recorded epoch produced the following overall bounding-box metrics on the validation split:

| Metric | Value |
|---|---:|
| Precision | 0.711 |
| Recall | 0.773 |
| mAP@0.50 | 0.780 |
| mAP@0.50:0.95 | 0.637 |

Training logs, plots, confusion matrices, and sample batches are available under [`training_results/classroom_test_run`](training_results/classroom_test_run). Separate validation artifacts are stored under [`validation_results/classroom_validation`](validation_results/classroom_validation).

## Run inference

Install Ultralytics:

```bash
python -m pip install ultralytics
```

Run inference with the best saved weights:

```python
from ultralytics import YOLO

model = YOLO("training_results/classroom_test_run/weights/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.25, save=True)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        print(class_id, model.names[class_id], confidence)
```

Use real classroom footage to tune the confidence threshold before production deployment.

## Repository structure

```text
.
├── dataset_augmented/       # YOLO dataset used for training/evaluation
├── Final Dataset - Copy/    # Original preparation and merge artifacts
├── predictions/             # Saved model predictions
├── training_results/        # Training metrics, plots, and weights
├── validation_results/      # Validation plots and predictions
├── PROJECT_BOARD.md         # Project tasks and development status
└── README.md
```

## Limitations and next steps

- Collect more `cleaner` examples to reduce class imbalance.
- Evaluate on footage from unseen classrooms, cameras, and lighting conditions.
- Review false positives and missed detections per class.
- Avoid publishing footage containing identifiable people without the required consent and institutional approval.
- Integrate detections with occupancy counting and AC-control logic only after real-world validation.

## Responsible use

This repository is intended for educational and authorized classroom-monitoring research. Follow applicable privacy, consent, data-retention, and surveillance policies when collecting images or deploying the model.
