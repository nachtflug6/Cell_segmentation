import os

import pandas as pd
import numpy as np
import datetime as dt


class GeneralReport:
    def __init__(self, out_path):
        dir_idx = 0
        repeat = True
        while repeat:
            dir_name = str(dt.date.today()) + '_' + str(dir_idx)
            dir_path = os.path.join(out_path, dir_name)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
                self.dir_path = dir_path
                repeat = False
            else:
                dir_idx += 1

        self.id = 0
        self.param_report = None
        self.init_param_report = False
        self.out_path = dir_path
        self.report_path = os.path.join(dir_path, 'report.csv')
        columns = ['mode', 'test_fold', 'param_id', 'report_id']
        self.general_report_columns = columns
        self.general_report = pd.DataFrame(columns=columns)

    def add_results(self, param, test_fold, result_report: pd.DataFrame, mode='validate'):
        assert mode in ['validate', 'test', 'final']
        df = pd.DataFrame(data={self.general_report_columns[0]: [mode],
                                self.general_report_columns[1]: [test_fold],
                                self.general_report_columns[2]: [param['id']],
                                self.general_report_columns[3]: [self.id]})
        self.general_report = pd.concat((self.general_report, df))
        result_report_name = str(self.id) + '_' + mode + '.csv'
        result_report.to_csv(os.path.join(self.out_path, result_report_name))
        self.id += 1

    def add_param_report(self, params):
        params_report = None
        for i, param in enumerate(params, 0):
            if isinstance(params_report, type(None)):
                params_report = ParamReport(param['id'], param, self.report_path)
            else:
                params_report.append(param['id'], param)
        params_report.save()
        self.param_report = params_report


class ParamReport:
    def __init__(self, report_id, param, out_path):
        df = pd.DataFrame()
        df['id'] = report_id
        for key in param:
            df[key] = [param[key]]
        self.df = df
        self.out_path = out_path

    def append(self, report_id, param):
        row = pd.DataFrame()
        row['id'] = report_id
        for key in param:
            row[key] = [param[key]]
        self.df = pd.concat((self.df, row))

    def save(self):
        self.df.to_csv(self.out_path, index=False)
