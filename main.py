from src.data_loader import load_digit_data
from src.model import DigitClassifier
from src.utils import accuracy_per_digit
from src.image_processing import image_to_features


def main():
    X_train, X_test, y_train, y_test = load_digit_data()
    classifier = DigitClassifier()
    classifier.train(X_train, y_train)
    score = classifier.evaluate(X_test, y_test)
    print(f"Nauwkeurigheid: {score:.2%}")
    predictions = classifier.predict(X_test)
    for digit, acc in accuracy_per_digit(y_test, predictions).items():
        print(f"Cijfer {digit}: {acc:.2%}")
    pad = input("Sleep een foto van een cijfer hierheen en druk op Enter: ").strip().strip('"')
    try:
        features = image_to_features(pad)
    except (OSError, ValueError):
        print("Dat lukte niet: geen geldige foto, of er staat geen cijfer op.")
        return
    print(f"Ik denk dat dit een {classifier.predict([features])[0]} is")

if __name__ == "__main__":
    main()