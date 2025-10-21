# 🧠 Brain Tumor MRI Classification (98% Accuracy & Live Web App)

---

## 🚀 Live Demo

Experience the final model in action through the interactive web application deployed on **Streamlit Community Cloud**:

👉 [https://brain-tumor-mri-classification.streamlit.app/](https://brain-tumor-mri-classification.streamlit.app/)

---

## 📜 Overview

This repository documents the **end-to-end journey** of building a state-of-the-art deep learning model to classify brain tumors from MRI images.  
The project explores and compares two sophisticated approaches:  
- A **custom-built CNN**  
- An **advanced transfer learning strategy**

The final model — **EfficientNetB0** trained with a **multi-phase gradual fine-tuning technique** — achieved **98% accuracy** and is now deployed as an **interactive web app**.

---

## 📈 Project Workflow

The project followed a structured, iterative methodology to achieve the best possible results:

1. **Exploratory Data Analysis (EDA)**  
   Investigated class distribution, image dimensions, and visualized samples to understand the dataset.

2. **Custom CNN Baseline**  
   Built a large, powerful CNN from scratch to establish a strong performance baseline (~95% accuracy).

3. **Advanced Transfer Learning**  
   Implemented a state-of-the-art **EfficientNetB0** model leveraging **ImageNet pre-trained weights**.

4. **Gradual Fine-Tuning**  
   Employed a **three-phase fine-tuning strategy** with differential learning rates and sequential layer unfreezing.

5. **Final Evaluation & Comparison**  
   Compared both models on **accuracy, AUC, and parameter efficiency**.

6. **Deployment**  
   Built a **Streamlit** web interface and deployed the best-performing model to the **cloud**.

---

## 🏆 Final Results & Analysis

The **gradual fine-tuning strategy** proved superior, yielding a model that is both highly accurate and remarkably efficient.

| Metric | Custom CNN | Transfer Learning (Winner) |
|:-------|:------------|:----------------------------|
| **Accuracy** | ~95% | ~98% |
| **AUC Score** | ~0.9950 | ~0.9996 |
| **Size (Parameters)** | ~135 Million | ~5.3 Million |

---

## 🛠️ Technology Stack

- **Framework:** PyTorch  
- **Pre-trained Models:** Timm (EfficientNetB0)  
- **Deployment:** Streamlit, Streamlit Community Cloud  
- **Data Handling:** NumPy, Pandas, OpenCV  
- **Visualization:** Matplotlib, Seaborn  
- **Utilities:** Scikit-learn  

---

## **Final Model**

The best performing model (efficientnet\_finetuned\_final.pth) is available for download from the Kaggle Notebook output.

* **Download Link:** [Kaggle Output Files](https://www.kaggle.com/code/abdoghazala/the-journey-to-98-cnn-vs-advanced-fine-tuning/output?select=efficientnet_finetuned_final.pth)

