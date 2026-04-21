# 🧩 Image Segmentation using U-Net (Deep Learning)

## 📌 Overview

This project implements a **U-Net deep learning model** for image segmentation.

It demonstrates:

* Semantic segmentation using encoder-decoder architecture
* Training on a real dataset (Oxford-IIIT Pet)
* Prediction on custom images
* Visualization of segmentation results

---

## 🚀 Key Highlights

* 🧠 U-Net architecture for pixel-wise classification
* 🖼️ Works on both dataset & custom images
* 📊 Visualization of predicted segmentation masks
* 🔄 Encoder-Decoder structure with skip connections
* 🎯 Multi-class segmentation

---

## 📂 Project Structure

```id="u8p2kd"
trained_model.py     # U-Net model + custom image prediction
lab-5.py             # Training on Oxford-IIIT Pet dataset
README.md            # Documentation
```

---

## 📊 Dataset

* **Training Dataset:** Oxford-IIIT Pet Dataset
* **Classes:** 3 (background, pet, outline)
* **Image Size:** Resized (128×128 or 256×256)

Dataset loaded using TensorFlow Datasets.

---

## ⚙️ Installation

Install required dependencies:

```bash id="p4n9sk"
pip install tensorflow tensorflow-datasets numpy matplotlib pillow
```

---

## ▶️ Usage

### Train Model

```bash id="k3z8vx"
python lab-5.py
```

### Predict on Custom Image

```bash id="v2m7qp"
python trained_model.py
```

---

## 🔍 Workflow Breakdown

### 1. Data Loading

* Dataset loaded using TensorFlow Datasets
* Train & test split automatically

### 2. Preprocessing

* Image resizing
* Normalization (0–255 → 0–1)
* Mask preprocessing for segmentation

---

## 🧠 Model Architecture (U-Net)

* Encoder (Downsampling with Conv + Pooling)
* Bottleneck (Feature extraction)
* Decoder (Upsampling + Skip Connections)
* Output layer (Softmax for segmentation classes)

👉 Implemented in: 

---

## 🏋️ Training Details

* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Epochs: 5
* Batch Size: 8

---

## 📊 Evaluation & Visualization

* Test Accuracy
* Predicted segmentation masks
* Comparison:

  * Input Image
  * Ground Truth Mask
  * Predicted Mask

👉 Training & visualization: 

---

## 🖼️ Custom Image Prediction

* Input image resized to model size
* Prediction mask generated
* Output saved as `predicted_image.jpg`

---

## 📈 Example Output

* Segmented regions highlighted with different classes
* Output image same size as original

---

## 🧠 Insights

* U-Net performs well for pixel-level classification
* Skip connections help preserve spatial details
* Works effectively even with limited data

---

## 🧪 Future Improvements

* Increase epochs for better accuracy
* Use data augmentation
* Try advanced segmentation models (DeepLab, UNet++)
* Improve color mapping for output masks

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* TensorFlow Datasets
* NumPy
* Matplotlib
* PIL (Image Processing)

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

This project demonstrates the power of U-Net architecture for image segmentation and provides a strong foundation for real-world applications like medical imaging, autonomous driving, and object segmentation.
