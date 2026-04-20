# 🖼️ CIFAR-10 Image Classification using ANN & CNN

## 📌 Overview

This project implements and compares two deep learning approaches for image classification on the **CIFAR-10 dataset**:

* 🧠 Artificial Neural Network (ANN)
* 🧩 Convolutional Neural Network (CNN)

It demonstrates how different architectures impact performance in computer vision tasks.

---

## 🚀 Key Highlights

* ✨ End-to-end deep learning pipeline
* 🧠 ANN model using fully connected layers
* 🧩 CNN model using convolution & pooling layers
* 📊 Performance evaluation using accuracy & confusion matrix
* 📈 Visualization of training vs validation metrics

---

## 📂 Project Structure

```id="p3k2dm"
lab-3_ANN.py    # ANN implementation
lab-3_CNN.py    # CNN implementation
README.md       # Documentation
```

---

## 📊 Dataset

* **Dataset:** CIFAR-10
* **Classes:** 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
* **Image Size:** 32×32 RGB

Dataset is directly loaded using TensorFlow/Keras.

---

## ⚙️ Installation

Install required dependencies:

```bash id="n9s1az"
pip install numpy tensorflow matplotlib seaborn scikit-learn
```

---

## ▶️ Usage

### Run ANN Model

```bash id="z8x1qp"
python lab-3_ANN.py
```

### Run CNN Model

```bash id="l5m2rs"
python lab-3_CNN.py
```

---

## 🔍 Workflow Breakdown

### 1. Data Loading

* CIFAR-10 dataset loaded using Keras
* Automatically split into training & testing

### 2. Preprocessing

* Pixel normalization (0–255 → 0–1)
* One-hot encoding for labels

---

## 🧠 ANN Model Architecture

* Flatten layer (32×32×3 → vector)
* Dense layers: 512 → 256 → 128
* Dropout (0.3) for regularization
* Output layer: Softmax (10 classes)

👉 Implemented in: 

---

## 🧩 CNN Model Architecture

* Conv2D + MaxPooling layers
* Feature extraction using convolution
* Dense layer (128 neurons)
* Dropout (0.5)
* Output layer: Softmax

👉 Implemented in: 

---

## 📊 Evaluation Metrics

* Accuracy
* Loss
* Confusion Matrix (ANN)
* Classification Report (ANN)

---

## 📈 Visualization

* Training vs Validation Accuracy
* Training vs Validation Loss
* Confusion Matrix heatmap

---

## 📊 Example Results

* ANN Accuracy: Moderate (~50–60%)
* CNN Accuracy: Higher (~65–75%)

*(Results may vary depending on training)*

---

## 🧠 Insights

* CNN performs significantly better for image data
* ANN struggles due to loss of spatial information
* Dropout helps reduce overfitting
* CNN captures spatial patterns effectively

---

## 🧪 Future Improvements

* Use deeper CNN architectures
* Add Batch Normalization
* Data augmentation
* Transfer learning (ResNet, VGG)
* Hyperparameter tuning

---

## 🛠️ Technologies Used

* Python
* NumPy
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Scikit-learn

---

## 👨‍💻 Author

Rahul Kumar
MCA – Semester 2
Course: IMDAI (CSET-654)

---

## 🤝 Contributing

Contributions and improvements are welcome!
Feel free to fork and enhance the project 🚀

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Final Note

This project clearly demonstrates the difference between traditional neural networks and convolutional networks for image classification, making it a strong addition to any machine learning portfolio.
