import os
import pandas as pd


def convert_name_to_dict(name):
    name = name.replace('Sample', '')
    name = name.replace('z', '')
    info_array = name.split('_')

    try:
        info_array = [int(info) for info in info_array]
    except:
        print(f"File{name} not named according to naming convention")

    return {'sample': info_array[0],
            'view': info_array[1],
            'z': info_array[2]}


def create_semantic_csv(path):

    images_path = os.path.join(path, 'images')
    labels_path = os.path.join(path, 'labels')

    assert os.path.isdir(images_path), f"Folder missing: {images_path}"
    assert os.path.isdir(labels_path), f"Folder missing: {labels_path}"

    df = pd.DataFrame({'Image': [], 'Label': [], 'Sample': [], 'View': [], 'Z': []})

    image_list = os.listdir(images_path)
    label_list = os.listdir(labels_path)

    for img in image_list:
        if img in label_list:
            img_path = os.path.join('images', img)
            label_path = os.path.join('labels', img)
            basename_split = os.path.splitext(img)
            name_dict = convert_name_to_dict(basename_split[0])
            df.loc[len(df.index)] = [img_path, label_path, name_dict['sample'], name_dict['view'], name_dict['z']]

    df.to_csv(os.path.join(path, 'data.csv'), index=False)

    return df

cwd = os.getcwd()
ds_path = os.path.join(cwd, '../../../data/cell_type_2')

create_semantic_csv(ds_path)
