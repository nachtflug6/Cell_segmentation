import numpy as np
import warnings


def index_params(params):
    for i, param in enumerate(params, 0):
        param['id'] = i
    return params

class MultiHyperparameter:
    def __init__(self, parametric_params):
        self.parametric_params = parametric_params
        self.__reset_idxs()

    def __get_params(self):
        out_params = self.parametric_params.copy()
        for key in self.varying_keys:
            out_params[key] = self.parametric_params[key][self.varying_keys[key]['idx']]
        return out_params

    def __increment(self, random=False):
        if random:
            for key in self.varying_keys:
                self.varying_keys[key]['idx'] = np.random.randint(0, self.varying_keys[key]['maxidx'])
            return True
        else:
            for key in self.varying_keys:
                self.varying_keys[key]['idx'] += 1
                if self.varying_keys[key]['idx'] == self.varying_keys[key]['maxidx']:
                    self.varying_keys[key]['idx'] = 0
                else:
                    return True
            return False

    def __reset_idxs(self):
        varying_keys = {}

        for key in self.parametric_params:
            current_param = self.parametric_params[key]

            if isinstance(current_param, list):
                varying_keys[key] = {'idx': 0, 'maxidx': len(current_param)}
        self.varying_keys = varying_keys
        return varying_keys

    def get_full_grid_params(self, indexed=False):
        out_params = []
        repeat = True

        while repeat:
            out_params.append(self.__get_params())
            repeat = self.__increment()
        self.__reset_idxs()

        if indexed:
            out_params = index_params(out_params)

        return out_params

    def get_random_params(self, num_params, indexed=False):
        out_params = []
        max_num = len(self.get_full_grid_params())
        if num_params > max_num:
            num_params = max_num
            warnings.warn(f'Only max: {max_num} random values possible')

        while len(out_params) < num_params:
            self.__increment(random=True)
            new_params = self.__get_params()
            if new_params not in out_params:
                out_params.append(self.__get_params())
        self.__reset_idxs()

        if indexed:
            out_params = index_params(out_params)

        return out_params
