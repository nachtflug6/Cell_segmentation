import pandas as pd
import os
import numpy as np
import torch as th
import torch.nn as nn
import torchvision.transforms as T

from custom_models.unet_original import UNet, UNet2
from evaluate.cross_evaluator import SemanticCrossEvaluator


cwd = os.getcwd()
out_path = os.path.join(cwd, '../to50jego/Cell_segmentation/results')
df = pd.DataFrame()
df['test'] = np.random.randint(0, 10, 10)
print(df['test'])
if not os.path.isdir(out_path):
    os.mkdir(out_path)
df.to_csv(os.path.join(out_path, 'test.csv'), index=False)

cwd = os.getcwd()
ds1_path = os.path.join(cwd, '../to50jego/Cell_segmentation/data/cell_type_1')
ds2_path = os.path.join(cwd, '../to50jego/Cell_segmentation/data/cell_type_2')

cv_param = {'interval_img_out': 1,
            'num_images': 5,
            'device': th.device("cuda" if th.cuda.is_available() else "cpu"),
            'datasets_path': [ds1_path, ds2_path],
            'results_path': os.path.join(cwd, '../to50jego/Cell_segmentation/results'),
            'folds': [0, 1, 2, 3],
            'epochs_cv': 3,
            'epochs_ct': 3}

param = {'id': 0,
         'padding_mode': 'zeros',
         'depth': 3,
         'start_layers': 32,
         'dim_multiplier': 2,
         'input_conv_kernel_size': 3,
         'out_classes': 2,
         'criterion': nn.CrossEntropyLoss(),
         'optimizer': 1,
         'augment_transform': T.Compose([T.CenterCrop(512)]),
         'num_augments': 10,
         'binarizer_lr': 0.05,
         'batch_size': 1}

unet = UNet.__new__(UNet)

cte = SemanticCrossEvaluator(unet, cv_param)
report = cte.cross_test_model([param], cv_param['epochs_ct'], cv_param['epochs_cv'])
