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
            elif len(os.listdir(dir_path)) == 0:
                repeat = False
            else:
                dir_idx += 1

        self.id = 0
        self.param_report = None
        self.init_param_report = False
        self.out_path = dir_path
        self.report_path = os.path.join(dir_path, 'report.csv')

    def add_results(self, param, result_report: pd.DataFrame):
        if not self.init_param_report:
            self.param_report = ParamReport(param, self.report_path)
            print(self.param_report)
            self.init_param_report = True
        else:
            self.param_report.append(param)
        self.param_report.save()
        result_report.to_csv(os.path.join(self.out_path, str(self.id) + '.csv'))


class ParamReport:
    def __init__(self, param, out_path):
        df = pd.DataFrame()
        for key in param:
            df[key] = [param[key]]
        self.df = df
        self.out_path = out_path

    def append(self, param):
        row = pd.DataFrame()
        for key in param:
            row[key] = [param[key]]
        self.df = pd.concat((self.df, row))

    def save(self):
        self.df.to_csv(self.out_path, index=False)
#
#
# class ResultsReport:
#     def __init__(self):
#         df = pd.DataFrame()
#         self.df = df
#         self.out_path = out_path
#         print(df)
#
#     def append(self, param):
#         row = pd.DataFrame()
#         for key in param:
#             row[key] = [param[key]]
#         self.df = pd.concat((self.df, row))
#
#     def save(self, out_path):
#         self.df.to_csv(self.out_path, index=False)