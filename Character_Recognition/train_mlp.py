import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

# Load dataset
train_df = pd.read_csv("emnist-letters-train.csv", header=None)
test_df = pd.read_csv("emnist-letters-test.csv", header=None)

# Split features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Normalize pixel values (IMPORTANT FIX)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create better MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    solver='adam',
    batch_size=128,
    max_iter=100,
    verbose=True,
    random_state=42
)

# Train
mlp.fit(X_train, y_train)

# Accuracy
print("Training Accuracy:", mlp.score(X_train, y_train))
print("Test Accuracy:", mlp.score(X_test, y_test))

# Save model
pickle.dump(mlp, open("mlp_emnist_model.pkl", "wb"))

print("✅ Model saved successfully!")