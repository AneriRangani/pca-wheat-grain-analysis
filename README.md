# Wheat Grain Geometrical Analysis using PCA

This project analyzes the geometrical characteristics of wheat grains using **Principal Component Analysis (PCA)**. The goal is to reduce the dimensionality of the dataset while preserving key information for visualization, anomaly detection, and classification. It also includes comparison of classifiers (Logistic Regression and Naive Bayes) on original and PCA-transformed data.

---

## 📊 Overview

Wheat is a key source of carbohydrates and protein. Analyzing its grain shape helps in understanding quality attributes important in agriculture and food processing. This project:
- Normalizes and visualizes the data
- Applies PCA to extract principal components
- Performs Hotelling’s T² control chart analysis
- Compares classifiers on raw and PCA-reduced features

---

## 🧰 Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 📁 Dataset

The dataset (`seeds4.csv`) contains various geometric measurements of wheat kernels. It must include a `class` label and several numerical feature columns. The script expects this file to be uploaded in Google Colab.

---

## ⚙️ How to Run

1. **Open** the script in [Google Colab](https://colab.research.google.com/)
2. **Upload** your `seeds4.csv` file when prompted:
   ```python
   from google.colab import files
   uploaded = files.upload()

Run all cells in sequence to:
Normalize the data
Perform PCA
Visualize results (scree plots, biplots, control charts)
Run classification models and display accuracy

📌 Features
Boxplots, pairplots, and heatmaps for EDA
PCA decomposition with scree plot and biplot
Hotelling’s T² test for identifying outliers
Control chart for principal components
Logistic Regression and Gaussian Naive Bayes with cross-validation

📈 Example Outputs
Scree plot showing explained variance
Covariance heatmap of features
Biplot of variables and projected data points
Control charts highlighting process behavior

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
