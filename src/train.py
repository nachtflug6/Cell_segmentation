import pandas as pd
import os
import numpy as np


cwd = os.getcwd()
out_path = os.path.join(cwd, '../results')
df = pd.DataFrame()
df['test'] = np.random.randint(0, 10, 10)
print(df['test'])
df.to_csv(os.path.join(out_path, 'test.csv'), index=False)
