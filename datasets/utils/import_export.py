import torch as th
import os
import torchvision.transforms as T

import tifffile as tif
import shutil
import numpy as np
from os import listdir
from PIL import Image


def transform_from_16bit_pil(pil_image):

    np_array = np.asarray(pil_image, dtype=np.float32)
    torch_array = th.from_numpy(np_array)
    torch_array = th.unsqueeze(torch_array, dim=0)

    return torch_array


def export_dataset(ds, export_path):

    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    for entry in ds:
        img, label = entry

        current_export_path = os.path.isdir(export_path, label)

        if not os.path.isdir(current_export_path):
            os.mkdir(current_export_path)

        else:
            pass


def import_custom_dataset(import_path,
                          classes=[1],
                          transform=T.Compose([T.ToTensor(), T.ConvertImageDtype(th.float)])):

    ds = []

    for current_class in classes:
        class_path = os.path.join(import_path, str(current_class))
        imgs = load_images_to_list_of_pil(class_path)

        for img in imgs:
            ds.append((transform(img), current_class))

    return ds


def augment_dataset(ds,
                    transform=T.Compose([T.ToTensor(), T.ConvertImageDtype(th.float)]),
                    num_augmentations=1,
                    with_original=False):

    augmented_ds = []

    for current_datastring in ds:

        img = current_datastring[0]
        label = current_datastring[1]

        if with_original:
            augmented_ds.append(current_datastring)

        for k in range(num_augmentations):
            augmented_ds.append((transform(img), label))

    return augmented_ds


def load_images_to_list_of_pil(import_path):
    # return array of images

    images_list = listdir(import_path)
    loaded_images = []
    for image in images_list:
        current_path = os.path.join(import_path, image)
        img = Image.open(current_path)
        loaded_images.append(img)

    return loaded_images


def transform_images_from_to(import_path, export_path, transform):
    list_images = load_images_to_list_of_pil(import_path)
    new_list_images = transform_list_of_pil(list_images, transform)


def transform_list_of_pil(pils, transform):

    new_pils = []

    for pil in pils:
        new_pils.append(transform(pil))

    return new_pils


def import_tif_stack(import_path):

    transt = T.ToTensor()
    img = tif.imread(import_path)
    img_t = transt(np.int16(img))
    img_t = th.swapaxes(img_t, 0, 1)
    while len(img_t.shape) < 4:
        img_t = th.unsqueeze(img_t, 1)
    return img_t


def export_img_stack_as_single_png(img_stack,
                                   export_path,
                                   transform=T.Compose([T.ToPILImage()]),
                                   index=None):

    assert len(img_stack.shape) == 5

    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    if isinstance(index, type(None)):

        stack_counter = 0

    elif isinstance(index, type(1)):

        img_stack = img_stack[index:index + 1]
        stack_counter = index

    elif isinstance(index, type(tuple((1, 1)))):

        img_stack = img_stack[index[0]:index[1] + 1]
        stack_counter = index[0]

    for img in img_stack:

        slice_counter = 0

        for img_slice in img:
            pil_img = transform(img_slice)
            name = str(stack_counter) + '_' + str(slice_counter) + '.png'
            pil_img.save(export_path + '/' + name)
            slice_counter += 1

        stack_counter += 1
