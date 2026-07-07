# Object Detection using YOLOv11 - Custom Dataset

This repository contains a custom object detection project using the **YOLOv11 Nano (yolo11n)** model to detect four categories of objects in a campus environment: **car**, **fountain**, **person**, and **tree**.

The dataset is sourced from Roboflow (v2) with a total of 755 images of size $1280 \times 720$ pixels.

---

## 🚀 Project Workflow (Step-by-Step)

### Step 1: Environment Setup
Ensure you have Python (version 3.8 or above) installed, and install all required library dependencies:
```bash
pip install ultralytics opencv-python tqdm requests pycocotools
```

### Step 2: Dataset Preparation & Configuration
1. The dataset is extracted from the `Yoloww.zip` file into the project folder. The dataset folder structure is organized as follows:
   ```text
   datasets + code/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   ├── test/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```
2. The configuration file **[data.yaml](./datasets%20+%20code/data.yaml)** defines the locations of the data and object labels:
   ```yaml
   train: ../train/images
   val: ../valid/images
   test: ../test/images

   nc: 4
   names: ["car", "fountain", "person", "tree"]
   ```

### Step 3: Model Training
Model training is performed using the pre-trained `yolo11n.pt` model for **20 epochs** with an input image size of 640 pixels. The CLI command executed:
```bash
yolo train model=yolo11n.pt data=data.yaml epochs=20 imgsz=640
```
The training process generates the best model weights (`best.pt`) and the last weights (`last.pt`) inside the training results folder.

### Step 4: Model Evaluation
The model is evaluated using the validation data to measure detection accuracy through metrics such as Precision, Recall, and mean Average Precision (mAP).

### Step 5: Real-Time Prediction (Inference)
Direct object detection using a camera (webcam) or a video file can be run by loading the best model weights (`best.pt`) using OpenCV:
```python
import cv2
from ultralytics import YOLO

# Load the best trained model
model = YOLO('datasets + code/best.pt')

# Connect to external / internal camera (ID: 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Display the detection results frame
    cv2.imshow('YOLOv11 Real-Time Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Training Results & Evaluation

After training for 20 epochs, the model achieved a **mAP50 of 0.565** and a **mAP50-95 of 0.324** across all classes.

### Accuracy Details per Class:
| Class | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | 1565 | 0.680 | 0.532 | 0.565 | 0.324 |
| **Car** | 110 | 0.569 | 0.277 | 0.277 | 0.096 |
| **Fountain** | 77 | 0.675 | 0.416 | 0.437 | 0.189 |
| **Person** | 351 | 0.738 | 0.709 | 0.755 | 0.487 |
| **Tree** | 1027 | 0.739 | 0.725 | 0.792 | 0.523 |

### Performance Metrics Charts:
The training results curves saved in the repository are shown below:

#### 1. Confusion Matrix (Normalized)
Illustrates the accuracy of the model's predictions compared to the actual ground truth labels for each class.
![Confusion Matrix](./confusion_matrix_normalized.png)

#### 2. Training Results Progress (Results)
Graphs of the loss functions (box_loss, cls_loss, dfl_loss) and performance metrics (mAP) over the 20 epochs of training.
![Training Results](./results.png)

#### 3. F1-Score & Precision-Recall (PR) Curves
Show the balance between the model's sensitivity (Recall) and precision (Precision).
![F1 Curve](./F1_curve.png)
![PR Curve](./PR_curve.png)

---

## 🎥 Video Testing Results
Several video recordings of direct object detection testing on sample videos have been included in the repository. You can click the links below to play/download the video files:
* [Test Video 1 (2024-11-11 15-17-30.mp4)](./2024-11-11%2015-17-30.mp4)
* [Test Video 2 (2024-11-11 15-19-25.mp4)](./2024-11-11%2015-19-25.mp4)
* [Test Video 3 (2024-11-11 15-21-17.mp4)](./2024-11-11%2015-21-17.mp4)
