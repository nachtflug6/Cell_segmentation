from matplotlib import cm
import numpy as np
from PIL import Image
import os


def save_tensor_to_colormap(tensor, out_path, name):
    inferno = cm.get_cmap('viridis')
    if tensor.max() > 255:
        tensor = tensor / 128
    elif tensor.max() <= 1:
        tensor = tensor * 255
    tensor = tensor.astype(np.uint8)
    img = inferno(tensor)
    img = img * 255
    img = img.astype(np.uint8)
    pil_image = Image.fromarray(img.astype(np.uint8))
    pil_image.save(os.path.join(out_path, name))
    return pil_image
