🍷 Wine Quality Classification using Deep Learning & Hyperparameter Tuning

📌 Overview

This project builds a deep neural network model to classify wine quality using the UCI Wine Quality (White Wine) dataset. It goes beyond basic modeling by incorporating:

Data preprocessing & scaling
Neural network training with Keras
Performance evaluation & visualization
Hyperparameter tuning using Grid Search and Randomized Search

This makes it a complete, production-style machine learning workflow.

🚀 Key Highlights

✨ End-to-end ML pipeline (data → model → evaluation → optimization)
🧠 Deep Learning model built with TensorFlow/Keras
📊 Visual performance analysis (accuracy curves + confusion matrix)
⚙️ Advanced tuning with GridSearchCV & RandomizedSearchCV
🛑 Early stopping to prevent overfitting

📂 Project Structure
lab-2.py        # Main implementation script
README.md       # Documentation

📊 Dataset
Source: UCI Machine Learning Repository
Dataset: White Wine Quality
Features: Physicochemical properties (e.g., acidity, sugar, pH)
Target: Wine quality score (multi-class classification)

⚙️ Installation

Install all dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow scikeras
▶️ Usage

Run the script:

python lab-2.py
🔍 Workflow Breakdown
1. Data Loading
Dataset is loaded directly from UCI repository
Uses pandas for structured data handling

2. Preprocessing
Feature scaling using MinMaxScaler
Label encoding + one-hot encoding for classification

3. Train-Test Split
80% training, 20% testing
Ensures reproducibility with fixed random state

4. Model Architecture
Fully connected neural network:
Input layer
Hidden layers with ReLU activation
Dropout layer (regularization)
Output layer with Softmax

5. Training Strategy
Optimizer: Adam
Loss: Categorical Crossentropy
Callback: EarlyStopping to avoid overfitting

6. Evaluation Metrics
Accuracy score
Classification report (precision, recall, F1-score)
Confusion matrix visualization

7. Visualization
Confusion matrix heatmap (Seaborn)
Training vs validation accuracy plots

8. Hyperparameter Tuning
🔹 Grid Search
Exhaustive search over:
Neurons
Learning rate
Batch size

🔹 Randomized Search
Faster exploration of larger parameter space

📈 Example Outputs
Accuracy: ~0.60–0.70 (varies based on tuning)

Classification Report:
Precision / Recall / F1-score per class

Best Grid Params:
{...}

Best Random Params:
{...}

🧠 Model Insights
Neural networks capture nonlinear relationships better than traditional models
Dropout helps reduce overfitting
Hyperparameter tuning significantly impacts performance

🧪 Future Improvements
Use Cross-validation with stratification
Try Batch Normalization
Experiment with deeper architectures
Apply feature selection / PCA
Compare with:
Random Forest
XGBoost
SVM

🛠️ Technologies Used
Python
NumPy & Pandas
Matplotlib & Seaborn
Scikit-learn
TensorFlow / Keras
SciKeras


👨‍💻 Author
[Rahul kumar]
MCA – Semester 2
Course: IMDAI (CSET-654)

🤝 Contributing

Contributions, suggestions, and improvements are welcome!
Feel free to fork and enhance the project.

📄 License

This project is licensed under the MIT License.

⭐ Final Note

This project is a solid demonstration of combining deep learning with classical ML optimization techniques—a strong portfolio piece for anyone stepping into data science or AI.