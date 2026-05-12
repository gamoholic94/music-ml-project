# Import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

print("--- Model Evaluation Script ---")

try:
    # ---------------------------------------------------
    # 1. Load and Prepare the Test Data
    # ---------------------------------------------------
    print("\n[1/5] Loading and preparing test data...")

    features_df = pd.read_csv("features.csv")

    # Separate features and labels
    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    # Split dataset
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Test data loaded successfully.")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ---------------------------------------------------
    # 2. Load Models and Scaler
    # ---------------------------------------------------
    print("\n[2/5] Loading models and scaler...")

    scaler = joblib.load('scaler.joblib')

    log_reg_model = joblib.load('logistic_regression_model.joblib')
    svm_model = joblib.load('svm_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')

    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5')

    print("All models loaded successfully.")

    # ---------------------------------------------------
    # 3. Prepare Data
    # ---------------------------------------------------
    print("\n[3/5] Preparing data for predictions...")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Reshape for CNN
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

    print(f"Scaled data shape: {X_test_scaled.shape}")
    print(f"CNN input shape: {X_test_cnn.shape}")

    # ---------------------------------------------------
    # 4. Generate Predictions
    # ---------------------------------------------------
    print("\n[4/5] Generating predictions...")

    # Scikit-learn model predictions
    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)

    # CNN predictions
    y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

    print("Predictions generated successfully.")

    # ---------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------
    print("\n[5/5] Evaluating models...")

    genre_names = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock'
    ]

    # ---------------------------------------------------
    # Confusion Matrices
    # ---------------------------------------------------

    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    print("\nConfusion matrices computed successfully.")
    # We create a helper function for plotting to avoid repetitive code.
    def plot_confusion_matrix(cm, labels, title, ax):
        """
        Plots a confusion matrix as a heatmap using Seaborn.
        
        Args:
            cm (np.array): The confusion matrix to plot.
            labels (list): The list of class names for the axes.
            title (str): The title for the plot.
            ax (matplotlib.axis): The subplot axis to plot on.
        """
        # Create a heatmap using seaborn.
        sns.heatmap(
            cm,                  # The confusion matrix data
            annot=True,          # Annotate each cell with its value
            fmt='d',             # Format the annotation as an integer
            cmap='Blues',        # Use the 'Blues' color map
            xticklabels=labels,  # Set the x-axis labels
            yticklabels=labels,  # Set the y-axis labels
            ax=ax                # Plot on the provided subplot axis
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    # Create a 2x2 subplot figure to display all four matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for All Models', fontsize=20)

    # Plot each confusion matrix on its respective subplot
    plot_confusion_matrix(cm_log_reg, genre_names, 'Logistic Regression', axes[0, 0])
    plot_confusion_matrix(cm_svm, genre_names, 'Support Vector Machine', axes[0, 1])
    plot_confusion_matrix(cm_rf, genre_names, 'Random Forest', axes[1, 0])
    plot_confusion_matrix(cm_cnn, genre_names, 'Convolutional Neural Network', axes[1, 1])

    # Adjust the layout to prevent titles and labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect is used to make space for suptitle

    # Display the final figure
    plt.show()


    # ---------------------------------------------------
    # Display Confusion Matrices
    # ---------------------------------------------------

    # Logistic Regression
    disp_log_reg = ConfusionMatrixDisplay(
        confusion_matrix=cm_log_reg,
        display_labels=genre_names
    )

    disp_log_reg.plot(xticks_rotation=45)
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    # SVM
    disp_svm = ConfusionMatrixDisplay(
        confusion_matrix=cm_svm,
        display_labels=genre_names
    )

    disp_svm.plot(xticks_rotation=45)
    plt.title("SVM Confusion Matrix")
    plt.show()

    # Random Forest
    disp_rf = ConfusionMatrixDisplay(
        confusion_matrix=cm_rf,
        display_labels=genre_names
    )

    disp_rf.plot(xticks_rotation=45)
    plt.title("Random Forest Confusion Matrix")
    plt.show()

    # CNN
    disp_cnn = ConfusionMatrixDisplay(
        confusion_matrix=cm_cnn,
        display_labels=genre_names
    )

    disp_cnn.plot(xticks_rotation=45)
    plt.title("CNN Confusion Matrix")
    plt.show()

    # ---------------------------------------------------
    # Classification Reports
    # ---------------------------------------------------

    print("\n" + "=" * 60)
    print("Classification Report: Logistic Regression")
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred_log_reg,
        target_names=genre_names
    ))

    print("\n" + "=" * 60)
    print("Classification Report: SVM")
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred_svm,
        target_names=genre_names
    ))

    print("\n" + "=" * 60)
    print("Classification Report: Random Forest")
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred_rf,
        target_names=genre_names
    ))

    print("\n" + "=" * 60)
    print("Classification Report: CNN")
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred_cnn,
        target_names=genre_names
    ))

    print("\nModel evaluation completed successfully!")

except FileNotFoundError as e:
    print(f"\nERROR: File not found -> {e.filename}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")