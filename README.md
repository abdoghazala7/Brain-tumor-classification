# 🧠 Brain Tumor MRI Classification (98% Accuracy with PyTorch)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-View%20Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/abdoghazala/the-journey-to-98-cnn-vs-advanced-fine-tuning/)

---
## Overview
This project is a deep dive into using deep learning for the classification of brain tumors from MRI images. We explore and compare two main approaches: building a custom Convolutional Neural Network (CNN) from scratch, and implementing an advanced transfer learning strategy using a pre-trained `EfficientNetB0` model.

The goal is to achieve the highest possible accuracy while understanding the trade-offs between model size, efficiency, and performance. The final model successfully achieved a stunning **98% accuracy** on the unseen test data.

---
## ✨ Key Features
* Comprehensive Exploratory Data Analysis (EDA) to understand the dataset's distribution.
* A powerful **Custom CNN** built from scratch, which served as a strong baseline, achieving ~96% accuracy.
* An advanced **Transfer Learning** strategy using `EfficientNetB0` and the `timm` library.
* Implementation of a three-phase **Gradual Fine-Tuning** technique to maximize performance.
* A detailed comparison between the two final models, analyzing the trade-offs between accuracy and parameter count.

---
## 🏆 Model Comparison
The final results showed a clear victory for the advanced transfer learning strategy.

| Metric | Custom CNN | **Transfer Learning (Winner)** |
| :--- | :--- | :--- |
| **Accuracy** | ~96% | **~98%** |
| **AUC Score** | ~0.9950 | **~0.9996** |
| **Size (Parameters)**| ~135 Million | **~5.3 Million** |

---
## Technologies Used
* **Framework**: PyTorch
* **Pre-trained Models**: Timm (EfficientNetB0)
* **Data Handling**: NumPy, Pandas, OpenCV
* **Visualization**: Matplotlib, Seaborn
* **Utilities**: Scikit-learn


  **Download the Dataset:**
    The dataset used for this project is the **Brain Tumor MRI Dataset**, which can be downloaded from Kaggle:
    * **Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
    * Make sure to place the data in the correct directory as referenced in the notebook.

## Final Model
The best performing model, `EfficientNetB0` trained with the gradual fine-tuning strategy, is available for download. This model achieved **98% accuracy** on the test set.

* **Download Link:** [efficientnet_finetuned_final.pth](https://www.kaggle.com/code/abdoghazala/the-journey-to-98-cnn-vs-advanced-fine-tuning/output?select=efficientnet_finetuned_final.pth)
