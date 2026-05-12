from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # ✅ FIXED
import joblib


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully!")
    return df


def preprocess_labels(y):
    print("\n--- Label Encoding Step ---")
    
    if np.issubdtype(y.dtype, np.integer):
        print("Labels already numeric.")
        return y, None
    else:
        print("Applying LabelEncoder...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print("Classes:", le.classes_)
        return y_encoded, le


def split_data(X, y):
    print("\n--- Train-Test Split ---")
    
    return train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


def scale_data(X_train, X_test):
    print("\n--- Scaling Features ---")
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled Data Check:")
    print("Train Mean (first 5):", X_train_scaled[:, :5].mean(axis=0))
    print("Train Std (first 5):", X_train_scaled[:, :5].std(axis=0))

    return X_train_scaled, X_test_scaled, scaler


# ✅ FIX: added scaler as parameter
def train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test, scaler):
    print("\n--- Training Logistic Regression Model ---")

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    print("Logistic Regression model trained successfully!")

    print("\n--- Training Random Forest Classifier Model ---")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    print("Random Forest Classifier model trained successfully!")

    print("\n--- Training Support Vector Machine (SVM) Model ---")

    svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)

    print("Support Vector Machine model trained successfully!")

    print("\n--- Evaluating Models on the Test Set ---")

    # Logistic Regression
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%")

    # SVM
    y_pred_svm = svm_model.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.2f}%")

    # Random Forest
    accuracy_rf = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")

    # Save models and scaler
    print("\n--- Saving Models and Scaler to Disk ---")

    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')

    print("Scaler and models have been successfully saved to disk.")
    print("The following files have been created in your project directory:")
    print("- scaler.joblib")
    print("- logistic_regression_model.joblib")
    print("- svm_model.joblib")
    print("- random_forest_model.joblib")

    print(f"\nModel learned the following classes: {rf_model.classes_}")
    print("Classes learned:", log_reg.classes_)

    return log_reg


def main():
    CSV_PATH = "features.csv"

    try:
        df = load_data(CSV_PATH)

        X = df.drop('genre_label', axis=1)
        y = df['genre_label']

        y, label_encoder = preprocess_labels(y)

        X_train, X_test, y_train, y_test = split_data(X, y)

        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        # ✅ FIX: pass scaler here
        model = train_logistic_regression(
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            scaler
        )

        accuracy = model.score(X_test_scaled, y_test)
        print(f"\nModel Accuracy: {accuracy:.4f}")

        return model

    except FileNotFoundError:
        print(f"File not found: {CSV_PATH}")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()