# car-type-classification
Deep learning project for fine-grained car type classification using Stanford Cars dataset. Includes training of ResNet50, InceptionV3, EfficientNetB0, evaluation (accuracy, confusion matrix, Grad-CAM), and a Streamlit GUI for predictions

🚗 Car Type Classification — Stanford Cars Project

This repository contains a complete deep learning pipeline for fine-grained car type classification using the Stanford Cars Dataset.
The project includes data preprocessing, training three transfer-learning models, model evaluation, Grad-CAM visualization, and a full Streamlit Web App.

---

📌 Project Goal

Classify car images into 196 different car makes/models/years by comparing the performance of 3 CNN architectures:

- ResNet50
- InceptionV3
- EfficientNetB0

---

📂 Repository Structure

/ (root)
├── README.md
├── TEAM.md
├── data/
│   └── download_data.sh
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── gradcam.py
│   └── gui/
│       └── app.py
├── results/
│   ├── confusion_matrix.png
│   └── gradcam_examples/
└── docs/
    └── evaluation_report.pdf

---

🧠 Models Used

Model| Notes
ResNet50| Strong CNN model using residual blocks (transfer learning)
InceptionV3| Extracts multi-scale features using inception blocks (transfer learning)
EfficientNetB0| Lightweight and fast baseline model

---

📊 Evaluation Metrics

Each model is evaluated through:

- Accuracy
- Precision / Recall / F1
- Confusion Matrix
- Grad-CAM visualization
- Model comparison table

---

📸 Streamlit GUI Features

The Web App provides:

- Upload car images
- Top-3 predictions + confidence
- Grad-CAM heatmaps
- Accuracy & confusion matrix results
- Select preferred trained model

Run GUI:

streamlit run src/gui/app.py

---

📥 Dataset

Stanford Cars Dataset (196 classes)
Contains: car make / model / year.

Download using:

bash data/download_data.sh

Or download manually from the official dataset link.

---

🧪 Training Example

python src/models/train.py --model resnet50 --epochs 10 --batch-size 32 --data-dir ./data

---

👥 Team Workflow (3 Members)

1. Data Team

- Download and prepare dataset
- Preprocessing + augmentation
- Train/Val/Test split

2. Model & Evaluation Team

- Train the 3 models
- Compare performance
- Accuracy, confusion matrix
- Grad-CAM visualization

3. GUI Team

- Build Streamlit application
- Show predictions and Grad-CAM
- Connect GUI with the best model

More details in TEAM.md.

---

🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- Streamlit

---

📎 How to Contribute

1. Create a feature branch:

git checkout -b feature/<branch-name>

2. Push changes:

git push origin feature/<branch-name>

3. Create a Pull Request → Request review → Merge into main.

---

📄 License

This project is for educational and academic use.

---

🎥 Demo & Results

All visualizations such as confusion matrices & Grad-CAM heatmaps are in the results/ folder.
A short demo video (if added) is inside docs/.

---

👥 Team Members

Names & responsibilities are included in TEAM.md.
