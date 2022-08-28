import numpy as np
import os

from format_dataset import create_semantic_csv


def assign_fold(ds_path, num_folds):
    data_path = os.path.join(ds_path, 'data.csv')

    df = create_semantic_csv(ds_path)

    min_df = df.copy()
    min_error = len(df)
    min_folds = None

    for k in range(1000):
        samples = []
        folds = np.zeros(num_folds)
        df['fold'] = np.zeros(len(df))

        for i, row in df.iterrows():
            df_sample = df[df['Sample'] == row['Sample']]
            df_sample_view = df_sample[df_sample['View'] == row['View']]
            sample_view_tuple = (row['Sample'], row['View'])

            if sample_view_tuple not in samples:

                samples.append(sample_view_tuple)
                current_fold = np.random.randint(0, num_folds)
                folds[current_fold] += len(df_sample_view)

                for j, current_row in df.iterrows():
                    if current_row['Sample'] == row['Sample'] and current_row['View'] == row['View']:
                        df.at[j, 'fold'] = current_fold

        current_error = np.sum(np.abs(folds - np.mean(folds)))

        if min_error > current_error:
            min_df = df.copy()
            min_error = current_error
            min_folds = folds
            print(min_folds)
            print(min_error)

    min_df.to_csv(data_path)
