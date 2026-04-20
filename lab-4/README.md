# 🖼️ CIFAR-10 Image Classification using ANN (Performance Analysis)

## 📌 Overview

This project implements an **Artificial Neural Network (ANN)** for image classification on the **CIFAR-10 dataset** and analyzes its performance in terms of accuracy and training time.

It focuses on:

* Model training using deep neural networks
* Performance evaluation
* Confusion matrix visualization
* Training time analysis

---

## 🚀 Key Highlights

* 🧠 ANN model for image classification
* ⏱️ Training time measurement
* 📊 Confusion matrix & classification report
* 📈 Performance evaluation on test data
* 🎯 Multi-class classification (10 classes)

---

## 📂 Project Structure

```id="f8k2mz"
lab-4.py        # ANN model with evaluation & timing
README.md       # Documentation
```

---

## 📊 Dataset

* **Dataset:** CIFAR-10
* **Classes:** 10 categories (Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
* **Image Size:** 32×32 RGB

Dataset is loaded directly using TensorFlow/Keras.

---

## ⚙️ Installation

Install required dependencies:

```bash id="k2s9qp"
pip install numpy tensorflow matplotlib seaborn scikit-learn
```

---

## ▶️ Usage

Run the script:

```bash id="z7x3vn"
python lab-4.py
```

---

## 🔍 Workflow Breakdown

### 1. Data Loading

* CIFAR-10 dataset loaded using Keras
* Automatically split into training & testing

### 2. Preprocessing

* Normalize pixel values (0–255 → 0–1)
* One-hot encoding of labels

---

## 🧠 Model Architecture

* Flatten layer (32×32×3 → vector)
* Dense layers:

  * 512 neurons (ReLU)
  * 256 neurons (ReLU)
  * 128 neurons (ReLU)
* Output layer:

  * 10 neurons (Softmax)

👉 Implemented in: 

---

## ⏱️ Training Details

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Epochs: 50
* Batch Size: 64
* Validation Split: 20%

---

## 📊 Evaluation Metrics

* Test Accuracy
* Training Time
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## 📈 Visualization

* Confusion matrix heatmap (Seaborn)

---

## 📊 Example Results

* **Accuracy:** ~50–60%
* **Training Time:** Depends on system (CPU/GPU)

*(Results may vary)*

---

## 🧠 Insights

* ANN struggles with image data due to lack of spatial feature extraction
* Performance is lower compared to CNN models
* Training time is moderate but not optimal for image tasks

---

## 🧪 Future Improvements

* Replace ANN with CNN for better performance
* Add Dropout for regularization
* Use Batch Normalization
* Try data augmentation
* Compare with advanced architectures (ResNet, VGG)

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

This project highlights the limitations of traditional neural networks for image classification and emphasizes the need for convolutional architectures in computer vision tasks.
