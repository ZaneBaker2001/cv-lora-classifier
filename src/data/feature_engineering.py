import cv2
import numpy as np
from scipy import ndimage, signal


def normalize_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    denom = x.max() - x.min()
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / denom


def compute_edge_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, threshold1=60, threshold2=140)
    return normalize_channel(edges)


def compute_texture_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sq_mean = cv2.GaussianBlur(gray ** 2, (0, 0), sigmaX=3)
    local_std = np.sqrt(np.maximum(sq_mean - local_mean ** 2, 0.0))

    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = np.abs(texture) + local_std
    return normalize_channel(texture)


def compute_land_index(image_rgb: np.ndarray) -> np.ndarray:
    """
    Satellite-style handcrafted index from RGB only.
    Acts like a pseudo vegetation / land-contrast feature.
    """
    image = image_rgb.astype(np.float32) / 255.0
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    exg = 2.0 * g - r - b
    exg = ndimage.gaussian_filter(exg, sigma=1.2)
    return normalize_channel(exg)


def compute_frequency_saliency(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    blurred = ndimage.gaussian_filter(gray, sigma=2.0)
    high_freq = gray - blurred

    kernel = np.outer(signal.windows.hann(9), signal.windows.hann(9))
    kernel = kernel / kernel.sum()
    filtered = signal.convolve2d(high_freq, kernel, mode="same", boundary="symm")

    return normalize_channel(np.abs(filtered))


def build_multichannel_tensor_input(image_rgb: np.ndarray) -> np.ndarray:
    edge = compute_edge_map(image_rgb)
    texture = compute_texture_map(image_rgb)
    land_index = compute_land_index(image_rgb)
    freq = compute_frequency_saliency(image_rgb)

    rgb = image_rgb.astype(np.float32) / 255.0

    stacked = np.dstack([rgb, edge, texture, land_index, freq])
    return stacked.astype(np.float32)