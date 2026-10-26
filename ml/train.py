"""
SignSpeak - Training Engine
Trains a Scikit-Learn classifier on data/gestures.csv and saves weights.
"""

def load_data(csv_path="data/gestures.csv"):
    """
    Reads gestures.csv using pandas.
    Separates the spreadsheet into:
      - X: The 63 coordinate columns (features)
      - y: The 'label' column (target words like 'HELLO', 'YES')
    """
    pass


def train_classifier():
    """
    1. Loads X and y with load_data().
    2. Splits data into 80% training / 20% testing (train_test_split).
    3. Trains a RandomForestClassifier on the training set.
    4. Evaluates accuracy on the test set and prints a score.
    5. Saves the trained model to 'models/sign_model.pkl' using joblib.
    """
    pass


if __name__ == "__main__":
    train_classifier()

