import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from scikeras.wrappers import KerasClassifier

df_wine = pd.read_csv("winequality-white.csv", sep=';')

X = df_wine.drop("quality", axis=1)
y = df_wine["quality"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)

def create_model(neurons=64, learning_rate=0.001, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons//2, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model()

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

model_gs = KerasClassifier(model=create_model, verbose=0)

param_grid = {
    "model__neurons": [32, 64],
    "model__learning_rate": [0.001, 0.01],
    "batch_size": [16, 32],
    "epochs": [50]
}

grid = GridSearchCV(model_gs, param_grid, cv=3)
grid.fit(X_train, y_train)

print("Best Grid Params:", grid.best_params_)

param_dist = {
    "model__neurons": [32, 64, 128],
    "model__learning_rate": [0.0001, 0.001, 0.01],
    "batch_size": [16, 32, 64],
    "epochs": [50, 75]
}

random_search = RandomizedSearchCV(
    model_gs,
    param_dist,
    n_iter=5,
    cv=3
)

random_search.fit(X_train, y_train)

print("Best Random Params:", random_search.best_params_)
