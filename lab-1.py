import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Load MNIST dataset
(X, y), (X_test_final, y_test_final) = mnist.load_data()

print("Image shape:", X.shape)
print("Labels:", np.unique(y))

# Visualize sample images
plt.figure(figsize=(6,3))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# Step 2: Preprocessing
# Flatten images
X_flat = X.reshape(X.shape[0], 28*28)

# Normalize pixel values
X_flat = X_flat / 255.0

# Step 3: Train-Test Split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42
)

# Step 4: Build & Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", accuracy)
