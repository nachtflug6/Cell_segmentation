import numpy as np
import pandas as pd
import torch as th

from datasets.semantic_dataset import SemanticDataset
from train.unet_trainer import UnetTrainer


class CrossTrainEvaluator:
    def __init__(self, model, datasets_path, device, results_path):
        self.model = model
        self.datasets_path = datasets_path
        self.device = device
        #self.params_to_test = params_to_test
        self.results_path = results_path
        self.folds = [0, 1, 2, 3]
        self.evaluated = []

    def evaluate_param(self, param, folds, num_epochs):
        print(param)
        current_model = self.model
        report = pd.DataFrame()

        comb_train_losses = np.zeros(num_epochs)
        comb_validation_losses = np.zeros(num_epochs)

        for i, dataset_path in enumerate(self.datasets_path, 0):
            train_losses = np.zeros(num_epochs)
            validation_losses = np.zeros(num_epochs)
            for train_fold in folds:
                folds_train = folds.copy()
                folds_train.remove(train_fold)
                folds_validate = [train_fold]
                current_model.__init__(param)
                ds_train = SemanticDataset(dataset_path, folds_train)
                ds_validate = SemanticDataset(dataset_path, folds_validate)
                trainer = UnetTrainer(current_model, self.device, param['criterion'],
                                      th.optim.Adam(current_model.parameters(), lr=1e-4, weight_decay=1e-5), ds_train, ds_validate)
                trainer.train(num_epochs, test=True)
                train_loss, validation_loss = trainer.get_losses()
                train_losses += train_loss
                validation_losses += validation_loss

            train_losses /= len(folds)
            validation_losses /= len(folds)
            comb_train_losses += train_losses
            comb_validation_losses += validation_losses

            key = 'ds_' + str(i)
            report[key + '_train'] = train_losses
            report[key + '_validate'] = validation_losses

        comb_train_losses /= 2
        comb_validation_losses /= 2
        report['combined_train'] = comb_train_losses
        report['combined_validate'] = comb_validation_losses

        return report

    def evaluate_model(self, params_to_test, num_epochs):

        for train_fold in self.folds:
            current_model = self.model
            folds_train = self.folds.copy()
            folds_train.remove(train_fold)
            folds_test = [train_fold]
            best_loss = 10e10
            best_param = None
            for param in params_to_test:
                report = self.evaluate_param(param, folds_train, num_epochs)
                if report['combined_validate'][-1] < best_loss:
                    best_param = param

            for i, dataset_path in enumerate(self.datasets_path, 0):
                train_losses = np.zeros(num_epochs)
                test_losses = np.zeros(num_epochs)
                current_model.__init__(best_param)
                ds_train = SemanticDataset(dataset_path, folds_train)
                ds_validate = SemanticDataset(dataset_path, folds_test)
                trainer = UnetTrainer(current_model, self.device, param['criterion'],
                                      th.optim.Adam(current_model.parameters(), lr=1e-4, weight_decay=1e-5),
                                      ds_train, ds_validate)
                trainer.train(num_epochs, test=True)
                train_loss, validation_loss = trainer.get_losses()
                train_losses += train_loss
                test_losses += validation_loss

                train_losses /= len(folds)
                validation_losses /= len(folds)
                comb_train_losses += train_losses
                comb_validation_losses += validation_losses

