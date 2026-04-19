# 🍷 Wine Quality Classification using Deep Learning & Hyperparameter Tuning

## 📌 Overview

This project builds a deep neural network model to classify wine quality using the **UCI Wine Quality (White Wine)** dataset.

It goes beyond basic modeling by incorporating:

* Data preprocessing & scaling
* Neural network training with Keras
* Performance evaluation & visualization
* Hyperparameter tuning using Grid Search and Randomized Search

This makes it a **complete, production-style machine learning workflow**.

---

## 🚀 Key Highlights

* ✨ End-to-end ML pipeline (data → model → evaluation → optimization)
* 🧠 Deep Learning model built with TensorFlow/Keras
* 📊 Visual performance analysis (accuracy curves + confusion matrix)
* ⚙️ Advanced tuning with GridSearchCV & RandomizedSearchCV
* 🛑 Early stopping to prevent overfitting

---

## 📂 Project Structure

```id="x81kpl"
lab-2.py        # Main implementation script
README.md       # Documentation
```

---

## 📊 Dataset

* **Source:** UCI Machine Learning Repository
* **Dataset:** White Wine Quality
* **Features:** Physicochemical properties (acidity, sugar, pH, etc.)
* **Target:** Wine quality score (multi-class classification)

---

## ⚙️ Installation

Install all dependencies:

```bash id="k9v2mz"
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow scikeras
```

---

## ▶️ Usage

Run the script:

```bash id="l2v8qe"
python lab-2.py
```

---

## 🔍 Workflow Breakdown

### 1. Data Loading

* Dataset loaded from UCI repository
* Uses pandas for structured handling

### 2. Preprocessing

* Feature scaling using MinMaxScaler
* Label encoding + one-hot encoding

### 3. Train-Test Split

* 80% training / 20% testing
* Fixed random state for reproducibility

### 4. Model Architecture

Fully connected neural network:

* Input layer
* Hidden layers (ReLU activation)
* Dropout layer (regularization)
* Output layer (Softmax)

### 5. Training Strategy

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Callback: EarlyStopping

### 6. Evaluation Metrics

* Accuracy score
* Classification report (Precision, Recall, F1-score)
* Confusion matrix

### 7. Visualization

* Confusion matrix heatmap (Seaborn)
* Training vs validation accuracy plots

### 8. Hyperparameter Tuning

#### 🔹 Grid Search

Exhaustive tuning over:

* Neurons
* Learning rate
* Batch size

#### 🔹 Randomized Search

* Faster exploration of large parameter space

---

## 📈 Example Outputs

* **Accuracy:** ~0.60–0.70 (varies with tuning)

**Classification Report:**

* Precision / Recall / F1-score per class

**Best Parameters:**

```id="q3s7mf"
Grid Search: {...}
Random Search: {...}
```

---

## 🧠 Model Insights

* Neural networks capture nonlinear relationships effectively
* Dropout helps reduce overfitting
* Hyperparameter tuning significantly improves performance

---

## 🧪 Future Improvements

* Use stratified cross-validation
* Add Batch Normalization
* Try deeper architectures
* Apply feature selection / PCA
* Compare with:

  * Random Forest
  * XGBoost
  * SVM

---

## 🛠️ Technologies Used

* Python
* NumPy & Pandas
* Matplotlib & Seaborn
* Scikit-learn
* TensorFlow / Keras
* SciKeras

---

## 👨‍💻 Author

Rahul Kumar
MCA – Semester 2
Course: IMDAI (CSET-654)

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!
Feel free to fork and enhance the project 🚀

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Final Note

This project demonstrates how deep learning can be combined with classical ML optimization techniques, making it a strong portfolio project for data science and AI roles.
