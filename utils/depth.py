import numpy as np
from scipy.ndimage import gaussian_filter


def depth2normal(depth_map: np.ndarray):
    rows, cols = depth_map.shape
    depth_map = gaussian_filter(depth_map.astype(float), sigma=1.0)

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Level and strength
    dz = 2.0

    # Calculate the partial derivatives of depth with respect to x and y
    dx, dy = np.gradient(depth_map)
    dx *= dz
    dy *= dz

    # Compute the normal vector for each pixel
    normal = np.dstack((dy, -dx, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    return normal
