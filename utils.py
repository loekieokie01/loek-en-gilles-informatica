def accuracy_per_digit(y_true, y_pred):
    results = {}
    for digit in range(10):
        correct = 0
        total = 0
        for real, guess in zip(y_true, y_pred):
            if real == digit:
                total += 1
                if guess == real:
                    correct += 1
        if total > 0:
            results[digit] = correct / total
    return results