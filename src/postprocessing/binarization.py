import torch as th
import numpy as np
from torchmetrics import JaccardIndex

from datasets.semantic_dataset import SemanticDataset


class Binarizer2Class:
    def __init__(self, device, learning_rate, initial_threshold=0.5):
        self.device = device
        self.learning_rate = learning_rate
        self.threshold = initial_threshold

    def train(self, loader):
        best_acc = 0
        best_threshold = 0
        for j in range(int(1 / self.learning_rate)):
            threshold = j * self.learning_rate
            iou = JaccardIndex(num_classes=2, threshold=threshold, average='none').to(self.device)
            acc = 0
            for i, data in enumerate(loader, 0):
                pred, target = data
                target = target.to(self.device)
                pred = pred.to(self.device)
                target = target[0, 1]
                pred = pred[0, 1]
                acc += iou(target, pred)[1]
            acc /= len(loader)
            if best_acc < acc:
                best_acc = acc
                best_threshold = threshold
        self.threshold = best_threshold

    def forward(self, prediction):
        pred = prediction.to(self.device)
        pred_binary = th.where(pred > self.threshold, 1.0, 0.0).to(th.float32)

        return pred_binary


