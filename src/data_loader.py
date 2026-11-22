from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digit_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_digit_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
