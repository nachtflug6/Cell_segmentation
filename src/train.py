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
ds1_path = os.path.join(cwd, '../to50jego/Cell_segmentation/data/cell_type_1')
ds2_path = os.path.join(cwd, '../to50jego/Cell_segmentation/data/cell_type_2')

cv_param = {'interval_img_out': 13,
            'num_images': 3,
            'device': th.device("cuda" if th.cuda.is_available() else "cpu"),
            'datasets_path': [ds1_path, ds2_path],
            'results_path': os.path.join(cwd, '../to50jego/Cell_segmentation/results'),
            'folds': [0, 1, 2, 3],
            'epochs_cv': 100,
            'epochs_ct': 10}

param = {'id': 0,
         'padding_mode': 'reflect',
         'out_classes': 2,
         'criterion': nn.CrossEntropyLoss(),
         'optimizer': MultiHyperparameter({'type': ['sgd'#, 'adam', 'rmsprop', 'asgd'
                                                    ], 'lr_factor': [0.5, 0.75, 1, 1.25, 1.5], 'weight_decay': [0, 1e-3, 1e-5]}).get_full_grid_params(),
         'augment_transform': [{'rotate': False, 'mirror': False, 'translate': False, 'pad': 0},
                               # {'rotate': True, 'mirror': True, 'translate': False, 'pad': 0},
                               # {'rotate': True, 'mirror': True, 'translate': True, 'pad': 16},
                               # {'rotate': True, 'mirror': True, 'translate': True, 'pad': 64}
                               ],
         'num_augments': 100,
         'binarizer_lr': 0.05,
         'batch_size': 2}

unet_hyps = MultiHyperparameter(param)
params = unet_hyps.get_random_params(3)
unet = UNet.__new__(UNet)

cte = SemanticCrossEvaluator(unet, cv_param)
report = cte.cross_test_model(params, cv_param['epochs_ct'], cv_param['epochs_cv'])