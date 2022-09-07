import numpy as np
import pandas as pd
import torch as th

from datasets.semantic_dataset import SemanticDataset
from train.unet_trainer import UnetTrainer
from postprocessing.binarization import Binarizer2Class
from utils.report import GeneralReport


class SemanticCrossEvaluator:
    def __init__(self, model, cv_param, n_models):
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
        self.failed_params = []
        self.best_n_accs = np.zeros(n_models)
        self.best_n_params = []
        self.num_models = n_models

    def train_validate(self, param, dataset_path, folds_train, folds_validate, num_epochs):
        if param not in self.failed_params:
            model = self.model
            model.__init__(param)
            ds_train = SemanticDataset(dataset_path, folds_train)
            ds_validate = SemanticDataset(dataset_path, folds_validate)

            trainer = UnetTrainer(model,
                                  self.device,
                                  param['criterion'],
                                  param['optimizer'],
                                  ds_train,
                                  ds_validate,
                                  param['augment_transform'],
                                  param['num_augments'],
                                  param['batch_size'],
                                  Binarizer2Class(self.device, param['binarizer_lr']),
                                  id=folds_validate[0],
                                  num_classes=param['out_classes'])

            success = trainer.train(num_epochs,
                                    img_output=True,
                                    num_images=self.num_images,
                                    out_folder=self.report.out_path,
                                    interval=self.interval_img_out)

            train_loss, validation_loss = trainer.get_losses()

            if not success:
                self.failed_params.append(param)

        else:
            print('Skipping previously failed param')
            train_loss = np.zeros(num_epochs)
            validation_loss = np.zeros(num_epochs)
            success = False

        return train_loss, validation_loss, success

    def cross_validate_param(self, param, folds, dataset_path, epochs):
        train_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)
        failed = False
        for i, train_fold in enumerate(folds, 0):
            print(f'Cross validate: {i} / {len(folds) - 1}')
            folds_train = folds.copy()
            folds_train.remove(train_fold)
            folds_validate = [train_fold]
            train_loss, validation_loss, success = self.train_validate(param,
                                                                       dataset_path,
                                                                       folds_train,
                                                                       folds_validate,
                                                                       epochs)
            if not success:
                failed = True
            train_losses += train_loss
            validation_losses += validation_loss

        train_losses /= len(folds)
        validation_losses /= len(folds)

        return train_losses, validation_losses, failed

    def evaluate_param(self, param, folds_train, folds_validate, epochs, test=True):
        results_report = pd.DataFrame()

        comb_train_losses = np.zeros(epochs)
        comb_validation_losses = np.zeros(epochs)
        failed = False

        for i, dataset_path in enumerate(self.datasets_path, 0):
            print(f'Evaluating Dataset: {i} / {len(self.datasets_path) - 1}')
            if test:
                train_losses, validation_losses, success = self.train_validate(param,
                                                                               dataset_path,
                                                                               folds_train,
                                                                               folds_validate,
                                                                               epochs)
                if not success:
                    failed = True
            else:
                train_losses, validation_losses, cv_failed = self.cross_validate_param(param, folds_train, dataset_path,
                                                                                       epochs)
                if cv_failed:
                    failed = True

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
            self.report.add_results(param, folds_validate[0], results_report, failed, mode='test')
        else:
            self.report.add_results(param, folds_validate[0], results_report, failed, mode='validate')
        return results_report, failed

    def cross_test_model(self, params, epochs_ct, epochs_cv):
        self.report.add_param_report(params)
        final_report = None

        for i in range(self.num_models):
            self.best_n_params.append(params[0])

        for i, train_fold in enumerate(self.folds, 0):
            print(f'Testing Fold: {i} / {len(self.folds) - 1}')
            folds_train = self.folds.copy()
            folds_train.remove(train_fold)
            folds_validate = [train_fold]
            self.best_n_accs = np.zeros(self.num_models)
            for j, param in enumerate(params, 0):
                print(f'Evaluating Param: {j} / {len(params) - 1}')
                report, _ = self.evaluate_param(param, folds_train, folds_validate, epochs_cv, test=False)
                if len(report) > 5:
                    current_acc = np.mean(report['combined_validate'][-5:])
                else:
                    current_acc = np.mean(report['combined_validate'])
                new_set = False
                for k, best_acc in enumerate(self.best_n_accs, 0):
                    if best_acc < current_acc and not new_set:
                        self.best_n_accs[k] = current_acc
                        self.best_n_params[k] = param
                        new_set = True

            idx = 0
            report = None
            repeat = True
            while repeat:
                report, failed = self.evaluate_param(self.best_n_params[idx], folds_train, folds_validate, epochs_ct, test=True)
                if failed and idx + 1 < self.num_models:
                    idx += 1
                else:
                    repeat = False

            if isinstance(final_report, type(None)):
                final_report = report
            else:
                for key in final_report:
                    final_report[key] += report[key]

            if i == 0:
                params = self.best_n_params

        final_report = final_report.div(len(self.folds))

        self.report.add_results(param, -1, final_report, mode='final',
                                failed=True if np.mean(final_report['combined_validate']) == 0 else False)
