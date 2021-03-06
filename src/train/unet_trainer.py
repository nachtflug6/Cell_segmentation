import numpy as np
import torch as th


class UnetTrainer:
    def __init__(self, model: th.nn.Module, device: th.device, criterion: th.nn.Module, optimizer: th.optim.Optimizer,
                 ds_train, ds_test):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.epochs = 0
        self.train_losses = []
        self.test_losses = []

    def train_epoch(self, trainloader):
        current_loss = []
        self.epochs += 1

        random_img = random_target = random_pred = None

        for j, data in enumerate(trainloader, 0):
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

    def test(self, testloader):
        current_loss = []
        self.epochs += 1

        random_img = random_target = random_pred = None

        for j, data in enumerate(testloader, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)

            x_predicted = self.model.forward(img)

            if j == 0:
                random_img = img.cpu().detach().numpy()
                random_target = target.cpu().detach().numpy()
                random_pred = x_predicted.cpu().detach().numpy()

            loss = self.criterion(x_predicted, target)

            current_loss.append(th.mean(loss).item())

        self.test_losses.append(np.mean(current_loss))

        return random_img, random_target, random_pred
