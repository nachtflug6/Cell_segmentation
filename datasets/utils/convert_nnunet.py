import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from nnunet.dataset_conversion.utils import generate_dataset_json

import format_dataset


def convert_ds_nnunet(ds_path, export_path, dataset_name, num_task):

    data_path = os.path.join(ds_path, 'data.csv')

    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
    else:
        df = create_csv.create_semantic_csv(ds_path)

    dirname = 'Task' + str(num_task) + '_' + dataset_name
    dir_path = os.path.join(export_path, dirname)

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    export_path_img = os.path.join(dir_path, 'imagesTr')
    export_path_imgts = os.path.join(dir_path, 'imagesTs')
    export_path_labels = os.path.join(dir_path, 'labelsTr')

    if not os.path.isdir(export_path_img):
        os.mkdir(export_path_img)

    if not os.path.isdir(export_path_imgts):
        os.mkdir(export_path_imgts)

    if not os.path.isdir(export_path_labels):
        os.mkdir(export_path_labels)

    for i, row in df.iterrows():
        nii_img = sitk.GetImageFromArray(np.load(os.path.join(ds_path, row['Image'])))
        nii_label = sitk.GetImageFromArray(np.load(os.path.join(ds_path, row['Label'])))
        running_number = str(i)
        num_0 = len(running_number)
        for i in range(3 - num_0):
            running_number = '0' + running_number
        filename_img = dataset_name + '_' + running_number + '_0000.nii.gz'
        filename_label = dataset_name + '_' + running_number + '.nii.gz'
        sitk.WriteImage(nii_img, os.path.join(export_path_img, filename_img))
        sitk.WriteImage(nii_label, os.path.join(export_path_labels, filename_label))

    generate_dataset_json(os.path.join(dir_path, 'dataset.json'), export_path_img, export_path_imgts, ('G'),
                          labels={0: 'background', 1: 'cell'}, dataset_name=dataset_name, license='poly')

cwd = os.getcwd()
ds_path = os.path.join(cwd, '../../data/cell_type_2_3d')
ds_path_export = os.path.join(cwd, '../../../nnUnet/nnUNet_raw_data_base/nnUNet_raw_data')

convert_ds_nnunet(ds_path, ds_path_export, 'test', 510)
