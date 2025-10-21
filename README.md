# **🧠 Brain Tumor MRI Classification (98% Accuracy & Live Web App)**

## **🚀 Live Demo**

Experience the final model in action through the interactive web application deployed on Streamlit Community Cloud:

[**https://brain-tumor-mri-classification.streamlit.app/**](https://brain-tumor-mri-classification.streamlit.app/)

## **📜 Overview**

This repository documents the end-to-end journey of building a state-of-the-art deep learning model to classify brain tumors from MRI images. The project explores and compares two sophisticated approaches: a custom-built CNN and an advanced transfer learning strategy.

The final model, an EfficientNetB0 trained with a multi-phase gradual fine-tuning technique, achieved a stunning **98% accuracy** and is now deployed as an interactive web app.

## **📈 Project Workflow**

The project followed a structured, iterative methodology to achieve the best possible results:

1. **Exploratory Data Analysis (EDA):** Investigated class distribution, image dimensions, and visualized samples to understand the dataset.  
2. **Custom CNN Baseline:** Built a large, powerful CNN from scratch to establish a strong performance baseline. This model achieved an impressive \~95% accuracy.  
3. **Advanced Transfer Learning:** Implemented a state-of-the-art EfficientNetB0 model, leveraging pre-trained weights from ImageNet.  
4. **Gradual Fine-Tuning:** Employed a three-phase fine-tuning strategy to maximize the transfer learning model's performance, which involved sequentially unfreezing layers and using differential learning rates.  
5. **Final Evaluation & Comparison:** Performed a detailed evaluation of both models on the test set, comparing them on accuracy, AUC, and parameter efficiency.  
6. **Deployment:** Built a user-friendly web interface with Streamlit and deployed the final, best-performing model to the cloud.

## **🏆 Final Results & Analysis**

The gradual fine-tuning strategy proved superior, yielding a model that is both highly accurate and remarkably efficient.

| Metric | Custom CNN | Transfer Learning (Winner) |
| :---- | :---- | :---- |
| **Accuracy** | \~95% | **\~98%** |
| **AUC Score** | \~0.9950 | **\~0.9996** |
| **Size (Parameters)** | \~135 Million | **\~5.3 Million** |

### **Final Classification Report**

The final model demonstrates excellent precision and recall across all classes, especially the perfect recall for Meningioma.

              precision    recall  f1-score   support

   pituitary       0.99      0.95      0.97       300  
     notumor       0.94      0.98      0.96       306  
  meningioma       0.99      1.00      1.00       405  
      glioma       0.99      0.99      0.99       300

    accuracy                           0.98      1311  
   macro avg       0.98      0.98      0.98      1311  
weighted avg       0.98      0.98      0.98      1311

## **🛠️ Technology Stack**

* **Framework**: PyTorch  
* **Pre-trained Models**: Timm (EfficientNetB0)  
* **Deployment**: Streamlit, Streamlit Community Cloud  
* **Data Handling**: NumPy, Pandas, OpenCV  
* **Visualization**: Matplotlib, Seaborn  
* **Utilities**: Scikit-learn

## **🚀 Getting Started**

1. **Clone the Repository:**  
   git clone \[https://github.com/abdoghazala7/Brain-tumor-classification.git\](https://github.com/abdoghazala7/Brain-tumor-classification.git)  
   cd Brain-tumor-classification

2. Install Dependencies:  
   It is recommended to create a virtual environment first.  
   pip install \-r requirements.txt

3. Download the Dataset:  
   The dataset is available on Kaggle: Brain Tumor MRI Dataset.  
4. **Run the Notebook or App:**  
   * To explore the training process, open the .ipynb file in a Jupyter environment.  
   * To run the web app locally, execute:  
     streamlit run app.py

## **Final Model**

The best performing model (efficientnet\_finetuned\_final.pth) is available for download from the Kaggle Notebook output.

* **Download Link:** [Kaggle Output Files](https://www.kaggle.com/code/abdoghazala/the-journey-to-98-cnn-vs-advanced-fine-tuning/output?select=efficientnet_finetuned_final.pth)
