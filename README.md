# FlexiSAF-week2
# ML Internship Project — FlexiSAF Internship Programme

This project showcases my understanding of two machine learning techniques: **Deep Learning using Convolutional Neural Networks (CNNs)** and **Ensemble Learning using Random Forests with Hyperparameter Tuning**. Each model is implemented on a different dataset using Python and Jupyter Notebooks on Google Colab.


---

## 🧠 Part 1: Deep Learning (CNN on Fashion-MNIST)

### ✅ Dataset:
- **Fashion-MNIST**, available on [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist).
- Contains 28x28 grayscale images of 10 different fashion categories.

### 🧩 Model Summary:
- **Conv2D layers** to extract spatial features.
- **MaxPooling2D** to reduce dimensionality while retaining key features.
- **Dense (Fully Connected)** layers for classification.
- **Softmax activation** to output class probabilities.
- Trained using **categorical crossentropy loss** and **Adam optimizer**.

### 🧪 Evaluation:
- Plotted images with true labels.
- Tracked performance using accuracy.
- Saved and reloaded the model using the  `.h5` format.
- Demonstrated label prediction and model accuracy through comparison of actual vs predicted values.

---

## 🌲 Part 2: Ensemble Learning (Random Forest on Loan Prediction Dataset)

### ✅ Dataset:
- Kaggle's [Loan Prediction Dataset](https://www.kaggle.com/datasets) .
- Features include applicant income, gender, property area,  etc.

### 🧩 Model Summary:
- Used **RandomForestClassifier** with `class_weight='balanced'` to handle class imbalance.
- Employed **GridSearchCV** for hyperparameter tuning:
  - Tested different values for `n_estimators`, `max_depth`, and more.
  - 4-fold cross-validation for robust evaluation.
- Defined a custom scoring function (`my_accuracy_score`) to evaluate model accuracy.

### 🧪 Evaluation:
- Handled missing values and categorical features.
- Compared training and validation accuracy.
- Analyzed feature importance (optional addition).
- Displayed classification metrics (confusion matrix, precision, recall).

---

## 💡 Key Skills Demonstrated

- Data preprocessing and exploration.
- Deep learning model architecture design.
- Ensemble learning and hyperparameter tuning.
- Custom evaluation metrics and validation strategies.
- Saving, reloading, and testing models.
- Visualizing model performance.

---


## 📬 Contact

**Yusuf Solomon**  
Mechatronics Engineering Student  
Email: ysolomon298@gmail.com 
LinkedIn: https://www.linkedin.com/in/yusuf-solomon/

---

