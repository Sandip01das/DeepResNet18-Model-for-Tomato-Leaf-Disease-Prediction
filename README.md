# 🍅 Tomato Leaf Disease Detection using DeepResNet18

This project presents an **end-to-end deep learning solution** for the detection of tomato leaf diseases using a **fine-tuned ResNet18** architecture. It aims to support sustainable agriculture by enabling early and accurate identification of diseases in tomato plants.

---

## 🔍 Project Overview

Tomato crops are vulnerable to various leaf diseases such as **Bacterial Spot** and **Tomato Mosaic Virus**, which impact yield and quality. Manual diagnosis by farmers is often error-prone and time-consuming. This project proposes a deep learning-based automated approach to identify tomato leaf conditions from images.

---

## 🧠 Model Architecture: DeepResNet18

- **Base model**: Pre-trained **ResNet18** on ImageNet.
- **Custom classifier head**:
  - Linear layer with 256 units
  - ReLU activation
  - Dropout: 0.4
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (`lr=0.001`)
- **Epochs**: 100

---

## 🗂️ Dataset

- Source: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- Classes used:
  - Tomato — Healthy
  - Tomato — Bacterial Spot
  - Tomato — Tomato Mosaic Virus
- Samples:
  - Training: 5,418 images
  - Validation: 1,354 images
- Image Size: 224x224

---

## 🛠️ Data Preprocessing & Augmentation

- **Training images**:
  - Resized to 224x224
  - Random horizontal flip
  - Random rotation
  - Random resized crop
  - Color jitter
  - Normalization (ImageNet mean/std)
  
- **Validation & test images**:
  - Only resized and normalized

---

## 📈 Results

| Metric     | Value        |
|------------|--------------|
| Validation Accuracy | **99.85%** (Epoch 73) |
| Test Accuracy       | **99.23%** |
| Precision, Recall, F1-score | ~99% for all 3 classes |

- The model outperformed other architectures like AlexNet, VGG16, InceptionV3, and CNN-ELM.
- Confusion matrix and feature map visualizations were used to interpret performance.

---

## 📊 Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Layer-wise Feature Maps

---

## 🔬 Tools & Technologies

- Python
- PyTorch
- torchvision
- NumPy, pandas
- matplotlib
- scikit-learn

---
