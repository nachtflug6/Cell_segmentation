import numpy as np
import pandas as pd
import torch as th

from datasets.semantic_dataset import SemanticDataset
from train.unet_trainer import UnetTrainer
from postprocessing.binarization import Binarizer2Class
from utils.report import GeneralReport


class CrossTrainEvaluator:
    def __init__(self, model, datasets_path, device, results_path):
        self.model = model
        self.datasets_path = datasets_path
        self.device = device
        #self.params_to_test = params_to_test
        self.report = GeneralReport(results_path)
        # self.results_path = results_path
        self.folds = [0, 1, 2, 3]
        self.evaluated = []

    def evaluate_folds(self, folds, dataset_path, param, num_epochs):
        current_model = self.model
        train_losses = np.zeros(num_epochs)
        validation_losses = np.zeros(num_epochs)
        for train_fold in folds:
            folds_train = folds.copy()
            folds_train.remove(train_fold)
            folds_validate = [train_fold]
            current_model.__init__(param)
            ds_train = SemanticDataset(dataset_path, folds_train)
            ds_validate = SemanticDataset(dataset_path, folds_validate)

            trainer = UnetTrainer(current_model,
                                  self.device,
                                  param['criterion'],
                                  th.optim.Adam(current_model.parameters(), lr=1e-4, weight_decay=1e-5),
                                  ds_train,
                                  ds_validate,
                                  param['augment_transform'],
                                  param['num_augments'],
                                  param['batch_size'],
                                  Binarizer2Class(self.device, param['binarizer_lr']),
                                  num_classes=param['out_classes'])

            trainer.train(num_epochs, test=True)
            train_loss, validation_loss = trainer.get_losses()
            train_losses += train_loss
            validation_losses += validation_loss

        train_losses /= len(folds)
        validation_losses /= len(folds)

        return train_losses, validation_losses

    def evaluate_param(self, param, folds, num_epochs):
        print(param)
        report = pd.DataFrame()

        comb_train_losses = np.zeros(num_epochs)
        comb_validation_losses = np.zeros(num_epochs)

        for i, dataset_path in enumerate(self.datasets_path, 0):
            train_losses, validation_losses = self.evaluate_folds(folds, dataset_path, param, num_epochs)
            comb_train_losses += train_losses
            comb_validation_losses += validation_losses

            key = 'ds_' + str(i)
            report[key + '_train'] = train_losses
            report[key + '_validate'] = validation_losses

        comb_train_losses /= 2
        comb_validation_losses /= 2
        report['combined_train'] = comb_train_losses
        report['combined_validate'] = comb_validation_losses

        self.report.add_results(param, report)
        return report

    def evaluate_model(self, params, folds, num_epochs):
        current_model = self.model
        train_losses = np.zeros(num_epochs)
        test_accs = np.zeros(num_epochs)
        for train_fold in folds:
            folds_train = folds.copy()
            folds_train.remove(train_fold)
            folds_validate = [train_fold]
            for param in params:
                current_model.__init__(param)
                report = self.evaluate_param()



