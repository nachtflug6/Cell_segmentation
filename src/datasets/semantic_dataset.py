import os
from torch.utils.data import Dataset
import pandas as pd
import tifffile as tif
import numpy as np


class SemanticDataset(Dataset):
    def __init__(self, ds_dir, folds, transform=None, target_transform=None, dim=2):
        assert dim in [2, 3]

        self.dim = dim

        self.ds_dir = ds_dir
        csv_path = os.path.join(ds_dir, 'data.csv')
        img_label_df = pd.read_csv(csv_path, index_col=False)

        self.img_label_df = None
        for fold in folds:
            if isinstance(self.img_label_df, type(None)):
                self.img_label_df = img_label_df[img_label_df['fold'] == fold]
            else:
                self.img_label_df = pd.concat((self.img_label_df, img_label_df[img_label_df['fold'] == fold]))

        print(self.img_label_df)

        self.len = len(self.img_label_df)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.ds_dir, self.img_label_df.iloc[idx, 0])
        image = tif.imread(img_path)
        image = np.asarray(image, dtype=np.float32)
        label_path = os.path.join(self.ds_dir, self.img_label_df.iloc[idx, 1])
        label = tif.imread(label_path)
        label = np.asarray(label, dtype=np.float32)
        np.where(label > 0, 1, 0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    # def create_nnunet_ds(self, export_path, num_task, name_task):
    #
    #     nnunet_task_path = os.path.join(export_path, 'Task' + str(num_task) + '_' + name_task)
    #
    #     if not os.path.isdir(nnunet_task_path):
    #         os.mkdir(nnunet_task_path)
    #         os.mkdir(os.path.join(nnunet_task_path, 'imagesTr'))
    #         os.mkdir(os.path.join(nnunet_task_path, 'imagesTs'))
    #         os.mkdir(os.path.join(nnunet_task_path, 'labelsTr'))
    #         os.mkdir(os.path.join(nnunet_task_path, 'labelsTs'))
    #
    #     for i in range(len(self)):
    #         img_path = self.img_label_df['Image'][i]
    #         label_path = self.img_label_df['Label'][i]
    #         # convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
    #         #                           )
    #
    #         img_path2 = os.path.join(self.ds_dir, img_path)
    #         print(img_path2)
    #
    #         convert_2d_image_to_nifti(os.path.join(self.ds_dir, img_path),
    #                                   os.path.join(nnunet_task_path, 'imagesTr', name_task + '_' + str(i) + '_0000'),
    #                                   is_seg=True)
    #         convert_2d_image_to_nifti(os.path.join(self.ds_dir, label_path),
    #                                   os.path.join(nnunet_task_path, 'labelsTr', name_task + '_' + str(i)),
    #                                   is_seg=True,
    #                                   transform=lambda x: (x == 255).astype(int))
    #
    #     generate_dataset_json(os.path.join(nnunet_task_path, 'dataset.json'),
    #                           os.path.join(nnunet_task_path, 'imagesTr'),
    #                           os.path.join(nnunet_task_path, 'imagesTs'),
    #                           'T',
    #                           labels={0: 'background', 1: 'cell'}, dataset_name=name_task, license='hands off!')


# ds = SemanticDataset(os.path.join(os.getcwd(), '../../data'))
# ds.create_nnunet_ds(os.path.join(os.getcwd(), '../nnUnet/nnUNet_raw_data_base/nnUNet_raw_data'), 503, 'RealCells')
