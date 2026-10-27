import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

CSV_PATH = os.path.join("data", "gestures.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")


def load_data(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at '{csv_path}'. "
            f"Run ml/collect_data.py to record some gestures first."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("[ERROR] Dataset is empty. Record some gestures before training.")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    return X, y


def train_classifier():
    X, y = load_data(CSV_PATH)
    unique_classes = list(set(y))

    print(f"[INFO] Loaded {len(X)} samples across {len(unique_classes)} classes: {unique_classes}")

    if len(unique_classes) < 2:
        print("[ERROR] You need at least 2 different gesture classes in your CSV to train a classifier.")
        return

    # Stratified 80/20 split keeps label ratios balanced in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[INFO] Fitting RandomForestClassifier (100 trees)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[RESULTS] Validation Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"[SUCCESS] Trained model artifact saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train_classifier()
