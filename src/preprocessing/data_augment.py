import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image


def mirror(img, target, p=0.5):
    if np.random.uniform(0, 1) < p:
        img = T.functional.hflip(img)
        target = T.functional.hflip(target)
    if np.random.uniform(0, 1) < p:
        img = T.functional.vflip(img)
        target = T.functional.vflip(target)

    return img, target


def rotate(img, target):
    random_value = np.random.uniform(0, 1)
    if random_value < 0.25:
        img = T.functional.rotate(img, 90)
        target = T.functional.rotate(target, 90)
    elif random_value < 0.5:
        img = T.functional.rotate(img, 180)
        target = T.functional.rotate(target, 180)
    elif random_value < 0.75:
        img = T.functional.rotate(img, 270)
        target = T.functional.rotate(target, 270)

    return img, target


def translate(img, target, padding):
    random_i = np.random.randint(0, padding)
    random_j = np.random.randint(0, padding)
    img_pad = nn.functional.pad(img, (padding, padding, padding, padding), 'reflect')
    target_pad = nn.functional.pad(target, (padding, padding, padding, padding), 'reflect')
    img = img_pad[:, 2 * random_i:2 * random_i + 512, 2 * random_i:2 * random_i + 512]
    target = target_pad[:, 2 * random_i:2 * random_i + 512, 2 * random_j:2 * random_j + 512]

    return img, target


# img = Image.open('../../results/2022-09-03_5/[1]_2_0_tr_img.png')
# img_np = np.asarray(img)
# img_th = th.from_numpy(img_np)
# img_th = th.swapaxes(img_th, 0, 2)
# img_th = th.unsqueeze(img_th, 0)
# img_th, _ = rotate(img_th, img_th)
# img_th, _ = mirror(img_th, img_th)
# img_mir, _ = translate(img_th.float(), img_th.float(), 64)
# print(img_mir.shape)
#
# # plt.imshow(th.swapaxes(img_th[0], 0, 2).numpy())
# plt.imshow(th.swapaxes(img_mir[0], 0, 2).numpy().astype(np.uint8))
# # plt.imshow(th.swapaxes(img_th[0], 0, 2).numpy())
# plt.show()


#
# img_th[0].show()
# print(img_th.shape)


class DataAugmenter:
    def __init__(self, dataset, augment_params):
        self.dataset = dataset
        self.num_data = len(dataset)
        self.augment_params = augment_params

    def get_loader(self, num_augments, batch_size):
        aug_ds = []
        for i in range(num_augments):
            img, target = self.dataset[np.random.randint(0, self.num_data)]
            if self.augment_params['rotate']:
                img, target = rotate(img, target)
            if self.augment_params['mirror']:
                img, target = mirror(img, target)
            if self.augment_params['translate']:
                img, target = translate(img, target, self.augment_params['pad'])
            aug_ds.append((img, target))

        return th.utils.data.DataLoader(aug_ds, batch_size=batch_size, shuffle=True)
