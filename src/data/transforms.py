import random

import cv2
import numpy as np


class TrainTransform:
    def __init__(self, p_flip: float = 0.5, p_blur: float = 0.2) -> None:
        self.p_flip = p_flip
        self.p_blur = p_blur

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p_flip:
            image = cv2.flip(image, 1)

        if random.random() < self.p_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image


class ValTransform:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image