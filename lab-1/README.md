🧠 MNIST Digit Classification using Decision Tree

📌 Overview

This project demonstrates a basic machine learning pipeline for classifying handwritten digits from the MNIST dataset using a Decision Tree Classifier. It covers data loading, preprocessing, visualization, training, and evaluation.

🚀 Features
Loads the MNIST dataset directly from TensorFlow/Keras
Visualizes sample handwritten digits
Preprocesses image data (flattening + normalization)
Splits dataset into training and testing sets
Trains a Decision Tree classifier
Evaluates model accuracy

🗂️ Project Structure
lab-1.py        # Main script containing full ML workflow
README.md       # Project documentation

⚙️ Installation

Make sure you have Python installed (>=3.7). Then install required dependencies:

pip install numpy matplotlib tensorflow scikit-learn
▶️ Usage

Run the script:

python lab-1.py
🔍 Workflow Explanation

1. Load Dataset
Uses MNIST dataset via Keras
Dataset contains 60,000 training and 10,000 test images
Each image is 28×28 grayscale

2. Data Visualization
Displays sample images with labels using Matplotlib

3. Preprocessing
Images are flattened from 28×28 → 784 features
Pixel values normalized to range [0, 1]

4. Train-Test Split
80% training, 20% testing
Uses train_test_split from Scikit-learn

5. Model Training
Decision Tree Classifier is trained on the dataset

6. Evaluation
Accuracy is calculated on test data

📊 Example Output
Image shape: (60000, 28, 28)
Labels: [0 1 2 3 4 5 6 7 8 9]
Decision Tree Accuracy: ~0.85 (varies slightly)

📈 Results
The Decision Tree model provides a simple baseline
Accuracy is decent but not state-of-the-art
More advanced models (e.g., CNNs) can significantly improve performance

🧪 Possible Improvements
Replace Decision Tree with:
Random Forest
Support Vector Machine (SVM)
Neural Networks / CNNs
Perform hyperparameter tuning
Use cross-validation
Add confusion matrix visualization

📚 Technologies Used
Python
NumPy
Matplotlib
TensorFlow / Keras
Scikit-learn

👨‍💻 Author
[Rahul kumar]
MCA – Semester 2
Course: IMDAI (CSET-654)

🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements