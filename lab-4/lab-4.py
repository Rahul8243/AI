import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']
ann_model = Sequential([
    Flatten(input_shape=(32,32,3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

ann_model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

ann_model.summary()

start_time = time.time()

ann_history = ann_model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

ann_training_time = time.time() - start_time

ann_test_loss, ann_test_acc = ann_model.evaluate(X_test, y_test_cat)
print("ANN Test Accuracy:", ann_test_acc)
print("ANN Training Time:", ann_training_time)
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)

cm_ann = confusion_matrix(y_test, y_pred_ann)

plt.figure(figsize=(8,6))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues')
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_test, y_pred_ann, target_names=class_names))
