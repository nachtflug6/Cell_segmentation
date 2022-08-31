import numpy as np
import torch as th
from torchmetrics import JaccardIndex

from preprocessing.data_augment import DataAugmenter


class UnetTrainer:
    def __init__(self, model: th.nn.Module, device: th.device, criterion: th.nn.Module, optimizer: th.optim.Optimizer,
                 ds_train, ds_test, augment_transform, num_augments, batch_size, binarizer, num_classes=2):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
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

    def train(self, epochs, test=False):
        for i in range(epochs):
            self.train_epoch()
            self.test()

    def train_epoch(self):
        self.model.train()
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
        self.model.eval()
        current_acc = []

        loader = th.utils.data.DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)
        data_binarizer = []

        for j, data in enumerate(loader, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)
            x_predicted = self.model.forward(img)
            target_store = target.clone().detach().to('cpu').to(th.int32)
            pred_store = x_predicted.clone().detach().to('cpu')
            data_binarizer.append((target_store, pred_store))

        loader = th.utils.data.DataLoader(data_binarizer, batch_size=self.batch_size, shuffle=True)
        self.binarizer.train(loader)

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

            current_acc.append(th.mean(acc).item())

        self.test_accs.append(np.mean(current_acc))

        return random_img.cpu().detach().numpy(), random_target.cpu().detach().numpy(), random_pred.cpu().detach().numpy()
