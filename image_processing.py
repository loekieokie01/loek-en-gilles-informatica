import numpy as np
from PIL import Image


def image_to_features(path):
    gray = np.array(Image.open(path).convert("L"), dtype=float)
    ink = 255 - gray
    ink = np.clip(ink - np.median(ink), 0, None)
    if ink.max() == 0:
        raise ValueError("Geen cijfer gevonden in de afbeelding")
    ink = ink / ink.max()
    ink[ink < 0.3] = 0

    rows = np.where(ink.max(axis=1) > 0)[0]
    cols = np.where(ink.max(axis=0) > 0)[0]
    digit = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    size = max(digit.shape)
    square = np.zeros((size, size))
    top = (size - digit.shape[0]) // 2
    left = (size - digit.shape[1]) // 2
    square[top:top + digit.shape[0], left:left + digit.shape[1]] = digit

    small = Image.fromarray((square * 255).astype("uint8"))
    small = small.resize((8, 8), Image.Resampling.BOX)
    return np.array(small, dtype=float).flatten() / 255 * 16