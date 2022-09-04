import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from preprocessing.data_augment import DataAugmenter
from utils.image import save_tensor_to_colormap


def get_optimizer(net, optim_param):
    assert optim_param['type'] in ['sgd', 'adam', 'rmsprop', 'asgd']
    if optim_param['type'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-3*optim_param['lr_factor'], weight_decay=optim_param['weight_decay'], momentum=0.99)
    elif optim_param['type'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=1e-4*optim_param['lr_factor'], weight_decay=optim_param['weight_decay'])
    elif optim_param['type'] == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=1e-4*optim_param['lr_factor'], weight_decay=optim_param['weight_decay'])
    else:
        optimizer = optim.ASGD(net.parameters(), lr=1e-3*optim_param['lr_factor'], weight_decay=optim_param['weight_decay'])
    return optimizer


class UnetTrainer:
    def __init__(self, model: th.nn.Module, device: th.device, criterion: th.nn.Module, optimizer_param,
                 ds_train, ds_test, augment_transform, num_augments, batch_size, binarizer, id=0, num_classes=2):
        self.model = model.to(device)
        self.id = id
        self.criterion = criterion
        self.optimizer = get_optimizer(model, optimizer_param)
        self.device = device
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.data_augmenter = DataAugmenter(ds_train, augment_transform)
        self.testloader = th.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True)
        self.epochs = 0
        self.num_data = num_augments
        self.train_losses = []
        self.test_accs = []
        self.iou = JaccardIndex(num_classes=num_classes).to(device)
        self.binarizer = binarizer
        self.batch_size = batch_size

    def get_losses(self):
        return self.train_losses, self.test_accs

    def train(self, epochs, img_output=False, num_images=5, out_folder=None, interval=0):
        for i in tqdm(range(epochs), desc="Training", ascii=False, ncols=75):
            if not img_output or i == 0 or ((i % interval != 0) and i != epochs - 1):
                self.train_epoch()
                self.test()
            else:
                self.train_epoch(num_images=num_images, out_folder=out_folder)
                self.test(num_images=num_images, out_folder=out_folder)

    def train_epoch(self, num_images=0, out_folder=None):
        self.model.train()
        current_loss = []
        self.epochs += 1
        loader = self.data_augmenter.get_loader(self.num_data, self.batch_size)

        random_idxs = np.random.permutation(len(loader))[:num_images]
        counter = 0

        for j, data in enumerate(loader, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)

            x_predicted = self.model.forward(img)
            if num_images > 0:
                if j in random_idxs:
                    name = str(self.id) + '_' + str(self.epochs) + '_' + str(counter) + '_tr_'
                    save_tensor_to_colormap(img.cpu().detach().numpy()[0][0], out_folder, name + 'img.png')
                    save_tensor_to_colormap(target.cpu().detach().numpy()[0], out_folder, name + 'tar.png')
                    save_tensor_to_colormap(x_predicted.cpu().detach().numpy()[0][1], out_folder, name + 'pre.png')
                    counter += 1
            loss = self.criterion(x_predicted, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_loss.append(th.mean(loss).item())

        self.train_losses.append(np.mean(current_loss))

    def test(self, num_images=0, out_folder=None):
        self.model.eval()

        loader = th.utils.data.DataLoader(self.ds_train, batch_size=1, shuffle=True)
        data_binarizer = []

        for j, data in enumerate(loader, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)
            x_predicted = self.model.forward(img)
            target_store = target.clone().detach().to('cpu').to(th.int32)
            pred_store = x_predicted.clone().detach().to('cpu')
            data_binarizer.append((target_store, pred_store))

        loader = th.utils.data.DataLoader(data_binarizer, batch_size=1, shuffle=True)
        self.binarizer.train(loader)

        random_idxs = np.random.permutation(len(loader))[:num_images]
        counter = 0

        current_acc = []
        for j, data in enumerate(self.testloader, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)

            x_predicted = self.model.forward(img)
            x_predicted = self.binarizer.forward(x_predicted)

            name = str(self.id) + '_' + str(self.epochs) + '_' + str(counter) + '_ts_'
            save_tensor_to_colormap(img.cpu().detach().numpy()[0][0], out_folder, name + 'img.png')
            save_tensor_to_colormap(target.cpu().detach().numpy()[0], out_folder, name + 'tar.png')
            save_tensor_to_colormap(x_predicted.cpu().detach().numpy()[0][1], out_folder, name + 'pre.png')
            if num_images > 0:
                if j in random_idxs:
                    name = str(self.id) + '_' + str(self.epochs) + '_' + str(counter) + '_ts_'
                    save_tensor_to_colormap(img.cpu().detach().numpy()[0][0], out_folder, name + 'img.png')
                    save_tensor_to_colormap(target.cpu().detach().numpy()[0], out_folder, name + 'tar.png')
                    save_tensor_to_colormap(x_predicted.cpu().detach().numpy()[0][1], out_folder, name + 'pre.png')
                    counter += 1

            #int_target = target.clone().detach().to(th.int32)
            acc = self.iou(x_predicted, target)
            current_acc.append(th.mean(acc).item())

        self.test_accs.append(np.mean(current_acc))
