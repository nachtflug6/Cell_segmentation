import numpy as np


def convert_images_nnunet(img, label):
    label_array = np.asarray(label, dtype=np.float32)
    label_array = np.where(label_array > 0, 1, 0)
    label_array = np.reshape(label_array, (1, *img.shape))

    img = np.asarray(img, dtype=np.float32)
    img = np.reshape(img, (1, *img.shape))

    return img, label_array