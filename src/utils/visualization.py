import cv2
import numpy as np
from scipy import ndimage


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def smooth_and_threshold_heatmap(
    heatmap: np.ndarray,
    sigma: float = 3.0,
    threshold: float = 0.45,
    min_region_size: int = 64,
) -> np.ndarray:
    smooth = ndimage.gaussian_filter(heatmap, sigma=sigma)
    binary = (normalize_heatmap(smooth) >= threshold).astype(np.uint8)

    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)
    binary = ndimage.binary_closing(binary, structure=np.ones((5, 5))).astype(np.uint8)

    labeled, num = ndimage.label(binary)
    if num == 0:
        return binary

    sizes = ndimage.sum(binary, labeled, index=np.arange(1, num + 1))
    cleaned = np.zeros_like(binary)

    for i, size in enumerate(sizes, start=1):
        if size >= min_region_size:
            cleaned[labeled == i] = 1

    return cleaned


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * normalize_heatmap(heatmap))
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.6, color_map, 0.4, 0)


def draw_regions(image_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    output = image_rgb.copy()
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(output, [contour], -1, (255, 255, 255), 1)

    return output