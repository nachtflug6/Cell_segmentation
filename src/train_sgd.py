import pandas as pd
import os
import numpy as np
import torch as th
import torch.nn as nn
import torchvision.transforms as T

from custom_models.unet_original import UNet
from evaluate.cross_evaluator import SemanticCrossEvaluator
from train.multi_hyperparameter import MultiHyperparameter

cwd = os.getcwd()
print(cwd)
ds1_path = os.path.join(cwd, '/cluster/to50jego/Cell_segmentation/data/cell_type_1')
ds2_path = os.path.join(cwd, '/cluster/to50jego/Cell_segmentation/data/cell_type_2')

cv_param = {'interval_img_out': 100,
            'num_images': 3,
            'device': th.device("cuda" if th.cuda.is_available() else "cpu"),
            'datasets_path': [ds1_path, ds2_path],
            'results_path': os.path.join(cwd, '../to50jego/Cell_segmentation/results'),
            'folds': [0, 1, 2, 3],
            'epochs_cv': 10,
            'epochs_ct': 100,
            'num_random_params': 3}

param = {'id': 0,
         'padding_mode': 'reflect',
         'out_classes': 2,
         'criterion': nn.CrossEntropyLoss(),
         'optimizer': MultiHyperparameter({'type': [
             'sgd',
             # 'adam',
             # 'rmsprop',
             # 'asgd'
         ],
             'lr_factor': [
                 10,
                 4,
                 2,
                 1,
                 0.5,
                 0.25,
                 0.1
             ],
             'weight_decay': [0,
                              1e-3,
                              1e-5
                              ]
         }).get_full_grid_params(),
         'augment_transform': [{'rotate': False, 'mirror': False, 'translate': False, 'pad': 0},
                               # {'rotate': True, 'mirror': True, 'translate': False, 'pad': 0},
                               # {'rotate': True, 'mirror': True, 'translate': True, 'pad': 16},
                               # {'rotate': True, 'mirror': True, 'translate': True, 'pad': 8}
                               ],
         'num_augments': 100,
         'binarizer_lr': 0.1,
         'batch_size': 2}

unet_hyps = MultiHyperparameter(param)
params = unet_hyps.get_full_grid_params()
print(len(params))
unet = UNet.__new__(UNet)
#
cte = SemanticCrossEvaluator(unet, cv_param)
report = cte.cross_test_model([params[0]], cv_param['epochs_ct'], cv_param['epochs_cv'])
