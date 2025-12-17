# 👥 Project Team Members

This document describes the team members involved in the **Car Type Classification — Stanford Cars Project**, along with their roles and responsibilities.

---

## 👤 Member 1 — Data Preparation

**Name:** Ahmed Walid Ahmad
**GitHub:** ahmedwalidahmad-debug

### Responsibilities

* Dataset downloading & cleaning
* Preprocessing & normalization
* Basic data augmentation
* Splitting data into Train / Validation / Test

### Deliverables

* `/preprocessing/load_data.py`
* `/preprocessing/preprocess.py`
* `/preprocessing/augmentation.py`

---

## 👤 Member 2 — Model Development (ResNet50)

**Name:** Ahmed Abdelrahim Mohamed

### Responsibilities

* Building and training ResNet50 model
* Saving model weights & metrics
* Generating confusion matrix

### Deliverables

* `/models/train_resnet.py`
* `resnet_results.json`

---

## 👤 Member 3 — Model Development (InceptionV3)

**Name:** Mariam

### Responsibilities

* Building and training InceptionV3
* Saving weights & evaluation results
* Uploading training curves

### Deliverables

* `/models/train_inception.py`
* `inception_results.json`

---

## 👤 Member 4 — Model Development (EfficientNetB0) & Evaluation

**Name:** Soha Allam
**GitHub:** sohaallam139-svg

### Responsibilities

* Training EfficientNetB0
* Extracting evaluation metrics (Precision, Recall, F1-score)
* Comparing all three models

### Deliverables

* `/models/train_efficientnet.py`
* `/evaluation/compare_models.ipynb`

---

## 👤 Member 5 — GUI Developer

**Name:** Nour

### Responsibilities

* Developing Streamlit GUI
* Image upload feature
* Displaying Top-3 predictions
* Model selection menu
* Confusion Matrix display
* Grad-CAM visualization (integration-ready)

### Deliverables

* `/gui/app.py`
* `/gui/gradcam_viewer.py`

---

## 👤 Member 6 — Documentation & GitHub Management

**Name:** Aya Hassan Mohamed

### Responsibilities

* Writing full project README
* Final report & documentation
* Organizing folder structure
* Creating `requirements.txt`
* Managing GitHub workflow (PRs, Issues, repo cleanup)

### ⭐ Bonus Tasks (Handled by Member 6)

* Full Grad-CAM implementation for all models
* Real-time webcam prediction (OpenCV)
* Converting one model to TensorFlow Lite (TFLite)
* Optional lightweight deployment support

### Deliverables

* `/docs/final_report.md`
* `README.md`
* `requirements.txt`
* `/bonus/gradcam.py`
* `/bonus/webcam_inference.py`
* `/bonus/tflite_model/`

---

📌 **Note:** All members are expected to complete their assigned tasks before the deadline and follow the GitHub contribution workflow described in the repository.
