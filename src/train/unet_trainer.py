import numpy as np
import torch as th
from torchmetrics import JaccardIndex

from preprocessing.data_augment import DataAugmenter

class UnetTrainer:
    def __init__(self, model: th.nn.Module, device: th.device, criterion: th.nn.Module, optimizer: th.optim.Optimizer,
                 ds_train, ds_test, data_binarizer, augment_transform, num_data, batch_size, num_classes=2):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.data_augmenter = DataAugmenter(ds_train, augment_transform)
        self.data_binarizer = data_binarizer
        self.testloader = th.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True)
        self.epochs = 0
        self.num_data = num_data
        self.train_losses = []
        self.test_losses = []
        self.iou = JaccardIndex(num_classes=num_classes).to(device)
        self.batch_size = batch_size

    def get_losses(self):
        return self.train_losses, self.test_losses

    def train(self, epochs, test=False):
        for i in range(epochs):
            self.train_epoch()
            self.test()

    def train_epoch(self):
        current_loss = []
        self.epochs += 1
        loader = self.data_augmenter.get_loader(self.num_data, self.batch_size)

        random_img = random_target = random_pred = None

        for j, data in enumerate(loader, 0):
            img, target = data

            target = target.to(self.device)
            img = img.to(self.device)
            x_predicted = self.model.forward(img)

            if j == 0:
                random_img = img.cpu().detach().numpy()
                random_target = target.cpu().detach().numpy()
                random_pred = x_predicted.cpu().detach().numpy()
            
            loss = self.criterion(x_predicted, target)

            self.optimizer.zero_grad()
            loss.backward() #th.ones_like(loss))
            self.optimizer.step()

            current_loss.append(th.mean(loss).item())

        self.train_losses.append(np.mean(current_loss))

        return random_img, random_target, random_pred

    def test(self):
        current_loss = []
        self.epochs += 1

        random_img = random_target = random_pred = None

        for j, data in enumerate(self.testloader, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)

            x_predicted = self.model.forward(img)

            if j == 0:
                random_img = img
                random_target = target
                random_pred = x_predicted

            int_target = target.clone().detach().to(th.int32)
            acc = self.iou(x_predicted, int_target)

            current_loss.append(th.mean(acc).item())

        self.test_losses.append(np.mean(current_loss))

        return random_img.cpu().detach().numpy(), random_target.cpu().detach().numpy(), random_pred.cpu().detach().numpy()
