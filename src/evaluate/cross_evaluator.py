import numpy as np
import pandas as pd
import torch as th

from datasets.semantic_dataset import SemanticDataset
from train.unet_trainer import UnetTrainer
from postprocessing.binarization import Binarizer2Class
from utils.report import GeneralReport


class SemanticCrossEvaluator:
    def __init__(self, model, cv_param):
        self.model = model
        self.datasets_path = cv_param['datasets_path']
        self.device = cv_param['device']
        report = GeneralReport(cv_param['results_path'])
        self.report = report
        self.results_folder = report.out_path
        self.folds = cv_param['folds']
        self.cv_param = cv_param
        self.interval_img_out = cv_param['interval_img_out']
        self.num_images = cv_param['num_images']

    def train_validate(self, param, dataset_path, folds_train, folds_validate, num_epochs):
        model = self.model
        model.__init__(param)
        ds_train = SemanticDataset(dataset_path, folds_train)
        ds_validate = SemanticDataset(dataset_path, folds_validate)

        trainer = UnetTrainer(model,
                              self.device,
                              param['criterion'],
                              th.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
                              ds_train,
                              ds_validate,
                              param['augment_transform'],
                              param['num_augments'],
                              param['batch_size'],
                              Binarizer2Class(self.device, param['binarizer_lr']),
                              id=folds_validate,
                              num_classes=param['out_classes'])

        trainer.train(num_epochs,
                      img_output=True,
                      num_images=self.num_images,
                      out_folder=self.report.out_path,
                      interval=self.interval_img_out)
        train_loss, validation_loss = trainer.get_losses()
        return train_loss, validation_loss

    def cross_validate_param(self, param, folds, dataset_path, epochs):
        train_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)
        for train_fold in folds:
            folds_train = folds.copy()
            folds_train.remove(train_fold)
            folds_validate = [train_fold]
            train_loss, validation_loss = self.train_validate(param,
                                                              dataset_path,
                                                              folds_train,
                                                              folds_validate,
                                                              epochs)
            train_losses += train_loss
            validation_losses += validation_loss

        train_losses /= len(folds)
        validation_losses /= len(folds)

        return train_losses, validation_losses

    def evaluate_param(self, param, folds_train, folds_validate, epochs, test=True):
        results_report = pd.DataFrame()

        comb_train_losses = np.zeros(epochs)
        comb_validation_losses = np.zeros(epochs)

        for i, dataset_path in enumerate(self.datasets_path, 0):
            if test:
                train_losses, validation_losses = self.cross_validate_param(param, folds_train, dataset_path, epochs)
            else:
                train_losses, validation_losses = self.train_validate(param,
                                                                      dataset_path,
                                                                      folds_train,
                                                                      folds_validate,
                                                                      epochs)
            comb_train_losses += train_losses
            comb_validation_losses += validation_losses

            key = 'ds_' + str(i)
            results_report[key + '_train'] = train_losses
            results_report[key + '_validate'] = validation_losses

        comb_train_losses /= 2
        comb_validation_losses /= 2
        results_report['combined_train'] = comb_train_losses
        results_report['combined_validate'] = comb_validation_losses
        if test:
            self.report.add_results(param, folds_validate[0], results_report, mode='test')
        else:
            self.report.add_results(param, folds_validate[0], results_report, mode='validate')
        return results_report

    def cross_test_model(self, params, epochs_ct, epochs_cv):
        self.report.add_param_report(params)
        final_report = None
        for i, train_fold in enumerate(self.folds, 0):
            print(f'Testing Fold: {i} / {len(self.folds)}')
            folds_train = self.folds.copy()
            folds_train.remove(train_fold)
            folds_validate = [train_fold]
            best_param = None
            best_acc = 0
            for j, param in enumerate(params, 0):
                print(f'Evaluating Param: {j} / {len(params)}')
                report = self.evaluate_param(param, folds_train, folds_validate, epochs_cv, test=True)
                if len(report['combined_validate']) > 10:
                    current_acc = np.mean(report['combined_validate'][10:])
                else:
                    current_acc = np.mean(report['combined_validate'])

                if best_acc < current_acc:
                    best_acc = current_acc
                    best_param = param

            report = self.evaluate_param(best_param, folds_train, folds_validate, epochs_ct, test=False)
            if isinstance(final_report, type(None)):
                final_report = report
            else:
                for key in final_report:
                    final_report[key] += report[key]

        final_report = final_report.div(len(self.folds))
        self.report.add_results(param, -1, final_report, mode='final')


