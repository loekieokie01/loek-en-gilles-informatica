from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class DigitClassifier:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)