# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# --- 1. Load and Prepare Data ---

CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    exit()

# Separate features and labels
X = features_df.drop('genre_label', axis=1)
y = features_df['genre_label']

# Encode labels (VERY IMPORTANT for neural networks)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete:")
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")

# --- 2. Reshape for CNN ---

X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

print("\nAfter reshaping:")
print(f"X_train_cnn: {X_train_cnn.shape}")
print(f"X_test_cnn: {X_test_cnn.shape}")

# --- 3. Build CNN Model with Anti-Overfitting Measures ---

model = Sequential()

# Conv1D Block 1: 24 filters (reduced from 32) with L2 regularization
model.add(Conv1D(
    filters=24,
    kernel_size=3,
    activation='relu',
    kernel_regularizer=l2(0.001),
    input_shape=(X_train_cnn.shape[1], 1)
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Increased from 0.3 to 0.5

# Conv1D Block 2: 48 filters (reduced from 64) with L2 regularization
model.add(Conv1D(
    filters=48,
    kernel_size=3,
    activation='relu',
    kernel_regularizer=l2(0.001)
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Added dropout after Conv block

# Conv1D Block 3: 96 filters (reduced from 128) with L2 regularization
model.add(Conv1D(
    filters=96,
    kernel_size=3,
    activation='relu',
    kernel_regularizer=l2(0.001)
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Added dropout after Conv block

# Flatten and Dense layers with L2 regularization
model.add(Flatten())

# Reduced Dense units from 128 to 64 with L2 regularization
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))  # Increased from 0.3 to 0.5

# Output layer
model.add(Dense(
    units=10,
    activation='softmax'
))
print("\n--- Compiling the CNN Model ---")

# The compile step configures the model for training.
model.compile(
    # `optimizer='adam'`: We use the Adam optimizer, an efficient and popular choice
    # that adapts the learning rate for each parameter.
    optimizer='adam', 
    
    # `loss='sparse_categorical_crossentropy'`: This is the ideal loss function for
    # multi-class classification where the true labels are integers (like our 0-9 genre labels).
    loss='sparse_categorical_crossentropy', 
    
    # `metrics=['accuracy']`: We ask the model to report its accuracy (the percentage
    # of correctly classified samples) during training and evaluation.
    metrics=['accuracy']
)

print("Model compiled successfully. It is now ready to be trained.")
print("\n--- Starting Model Training with EarlyStopping ---")

# EarlyStopping callback to prevent overfitting by stopping when validation loss plateaus
early_stopping = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=10,                  # Stop if no improvement for 10 epochs
    restore_best_weights=True,    # Restore weights from epoch with best validation loss
    verbose=1
)

# Train the model with validation data and EarlyStopping
history = model.fit(
    X_train_cnn, y_train,
    epochs=100,                   # Allow up to 100 epochs, but will stop early
    batch_size=32,
    validation_split=0.2,         # Use 20% of training data for validation
    callbacks=[early_stopping],   # Use EarlyStopping callback
    verbose=1
)

print("\n--- Model Training Complete ---")
print("\n--- Plotting Training and Validation History ---")

import matplotlib.pyplot as plt

# Create a function to plot the training history. This is good practice for reusability.
def plot_history(history):
    """Plots accuracy and loss for training and validation sets."""
    
    # Create a figure with two subplots: one for accuracy, one for loss
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # --- Plot Training & Validation Accuracy ---
    # Access the accuracy history from the history object
    axs[0].plot(history.history["accuracy"], label="Training Accuracy")
    # Access the validation accuracy history
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend(loc="lower right")

    # --- Plot Training & Validation Loss ---
    # Access the loss history
    axs[1].plot(history.history["loss"], label="Training Loss")
    # Access the validation loss history
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend(loc="upper right")

    # Ensure the plots are nicely arranged
    plt.tight_layout()

    # Display the plots
    plt.show()

# Call the function with our history object
plot_history(history)
print("\n--- Saving the trained CNN model to disk ---")

# The model.save() method saves the entire model to a single HDF5 file.
# This file includes the model's architecture, weights, and training configuration.
# We give it a descriptive name for easy identification later.
model.save("music_genre_cnn.h5")

print("\nModel successfully saved as 'music_genre_cnn.h5' in your project directory.")
print("This file can now be loaded for future evaluation or deployment.")

# --- 7. Evaluate Model ---

test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
train_loss, train_acc = model.evaluate(X_train_cnn, y_train, verbose=0)

print("\n" + "="*50)
print("--- FINAL MODEL PERFORMANCE METRICS ---")
print("="*50)
print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Overfitting Gap: {(train_acc-test_acc)*100:.2f}%")
print("\n" + "="*50)
print("Anti-Overfitting Techniques Applied:")
print("  - Dropout: 0.5 (increased from 0.3)")
print("  - L2 Regularization: 0.001 on all Conv and Dense layers")
print("  - Reduced Model Complexity:")
print("    - Conv filters: 24->48->96 (was 32->64->128)")
print("    - Dense units: 64 (was 128)")
print("  - EarlyStopping: Stops when validation loss plateaus for 10 epochs")
print("="*50 + "\n")