# 🧠 MNIST Digit Classification using Decision Tree

## 📌 Overview

This project demonstrates a basic machine learning pipeline for classifying handwritten digits from the MNIST dataset using a Decision Tree Classifier.

It covers:

* Data loading
* Preprocessing
* Visualization
* Training
* Evaluation

---

## 🚀 Features

* Loads MNIST dataset using TensorFlow/Keras
* Visualizes handwritten digits
* Preprocesses images (flattening + normalization)
* Splits dataset into training and testing sets
* Trains a Decision Tree classifier
* Evaluates model accuracy

---

## 📂 Project Structure

```
lab-1.py       # Main script containing ML workflow
README.md      # Project documentation
```

---

## ⚙️ Installation

Make sure Python (>=3.7) is installed, then run:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## ▶️ Usage

Run the script:

```bash
python lab-1.py
```

---

## 🔍 Workflow Explanation

1. **Load Dataset**

   * MNIST via Keras
   * 60,000 training + 10,000 testing images
   * Image size: 28×28 grayscale

2. **Data Visualization**

   * Sample images displayed using Matplotlib

3. **Preprocessing**

   * Flatten: 28×28 → 784 features
   * Normalize pixel values to [0, 1]

4. **Train-Test Split**

   * 80% training / 20% testing

5. **Model Training**

   * Decision Tree Classifier

6. **Evaluation**

   * Accuracy calculated on test data

---

## 📊 Example Output

* Image shape: `(60000, 28, 28)`
* Labels: `[0–9]`
* Accuracy: **~0.85** (may vary)

---

## 📈 Results

The Decision Tree model provides a simple baseline.
Accuracy is decent but not state-of-the-art.

---

## 🔧 Possible Improvements

* Use Random Forest
* Try Support Vector Machine (SVM)
* Implement Neural Networks / CNNs
* Perform hyperparameter tuning
* Add confusion matrix visualization

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib
* TensorFlow / Keras
* Scikit-learn

---

## 👨‍💻 Author

Rahul Kumar
MCA – Semester 2
Course: IMDAI (CSET-654)

---

## 🤝 Contributing

Feel free to fork this repository and submit improvements 🚀
