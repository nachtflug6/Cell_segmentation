import pandas as pd
import os
import numpy as np


cwd = os.getcwd()
out_path = os.path.join(cwd, '../to50jego/Cell_segmentation/results')
df = pd.DataFrame()
df['test'] = np.random.randint(0, 10, 10)
print(df['test'])
if not os.path.isdir(out_path):
    os.mkdir(out_path)
df.to_csv(os.path.join(out_path, 'test.csv'), index=False)
