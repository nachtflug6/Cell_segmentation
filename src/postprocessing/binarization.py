# import torch as th
# import numpy as np
#
#
# class Binarizer:
#     def __init__(self, dataset, augment_method):
#         self.dataset = dataset
#         self.num_data = len(dataset)
#         self.augment_method = augment_method
#
#     def get_loader(self, num_data, batch_size):
#         aug_ds = []
#         for i in range(num_data):
#             img, target = self.dataset[np.random.randint(0, num_data)]
#             img_aug = self.augment_transform(img)
#             target_aug = self.augment_transform(target)
#             aug_ds.append((img_aug, target_aug))
#
#         return th.utils.data.DataLoader(batch_size, batch_size=batch_size, shuffle=True)