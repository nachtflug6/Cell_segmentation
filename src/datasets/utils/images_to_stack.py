import pandas as pd
import os
import numpy as np

import tifffile

import format_dataset


def z_images_to_np_stack(ds_path, export_path):
    data_path = os.path.join(ds_path, 'data.csv')

    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
    else:
        df = format_dataset.create_semantic_csv(ds_path)

    samples = []

    df_out = pd.DataFrame({'Image': [], 'Label': [], 'Sample': [], 'View': [], 'Zs': []})

    for i, row in df.iterrows():
        sample_view_tuple = (row['Sample'], row['View'])

        if sample_view_tuple not in samples:
            samples.append(sample_view_tuple)

            img_tensor = None
            label_tensor = None

            df_sample = df[df['Sample'] == row['Sample']]
            df_sample_view = df_sample[df_sample['View'] == row['View']]
            df_sample_view = df_sample_view.sort_values('Z')

            for j, row_sample_view in df_sample_view.iterrows():

                img_path = os.path.join(ds_path, row_sample_view['Image'])
                label_path = os.path.join(ds_path, row_sample_view['Label'])
                img = np.asarray(tifffile.imread(img_path))
                label = np.asarray(tifffile.imread(label_path))
                label = np.where(label > 0, 1, 0)

                if isinstance(img_tensor, type(None)):
                    img_tensor = np.expand_dims(img, axis=0)
                    label_tensor = np.expand_dims(label, axis=0)
                else:
                    img_tensor = np.concatenate((img_tensor, np.expand_dims(img, axis=0)))
                    label_tensor = np.concatenate((label_tensor, np.expand_dims(label, axis=0)))

            if not os.path.isdir(export_path):
                os.mkdir(export_path)

            export_name = 's' + str(row['Sample']) + '_v' + str(row['View'])
            print(f'Saving: {export_name}')

            img_export_path = os.path.join(export_path, 'images')
            label_export_path = os.path.join(export_path, 'labels')

            if not os.path.isdir(img_export_path):
                os.mkdir(img_export_path)
            if not os.path.isdir(label_export_path):
                os.mkdir(label_export_path)

            np.save(os.path.join(img_export_path, export_name), img_tensor)
            np.save(os.path.join(label_export_path, export_name), label_tensor)

            df_out.loc[len(df_out.index)] = [os.path.join('images', export_name),
                                             os.path.join('labels', export_name),
                                             row['Sample'],
                                             row['View'],
                                             df_sample_view['Z'].max()]

    df_out.to_csv(os.path.join(export_path, 'data.csv'), index=False)


cwd = os.getcwd()
ds_path = os.path.join(cwd, '../../../data/cell_type_2')
ds_path_export = os.path.join(cwd, '../../data/cell_type_2_3d')

z_images_to_np_stack(ds_path, ds_path_export)
