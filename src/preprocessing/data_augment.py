import torch as th
import numpy as np


class DataAugmenter:
    def __init__(self, dataset, augment_transform):
        self.dataset = dataset
        self.num_data = len(dataset)s
        self.augment_transform = augment_transform

    def get_loader(self, num_data, batch_size):
        aug_ds = []
        for i in range(num_data):
            img, target = self.dataset[np.random.randint(0, num_data)]
            img_aug = self.augment_transform(img)
            target_aug = self.augment_transform(target)
            aug_ds.append((img_aug, target_aug))

        return th.utils.data.DataLoader(aug_ds, batch_size=batch_size, shuffle=True)
