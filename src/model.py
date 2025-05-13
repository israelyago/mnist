import lightning as L
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
)


class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.smoothed_value = None

    def update(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            self.smoothed_value = (
                self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
            )
        return self.smoothed_value

    def get_smoothed_value(self):
        return self.smoothed_value

    def reset(self):
        self.smoothed_value = None


def get_udlbook_arch():
    """Understanding DeepLearning (Book) inspired architecture"""
    return nn.Sequential(
        # 1
        nn.Conv2d(1, 10, kernel_size=3),
        # 2
        nn.MaxPool2d(2),
        # 3
        nn.ReLU(),
        # 4
        nn.Conv2d(10, 20, kernel_size=3),
        # 5
        nn.Dropout2d(),
        # 6
        nn.MaxPool2d(2),
        # 7
        nn.ReLU(),
        # 8
        nn.Flatten(),
        # 9
        nn.Linear(500, 50),
        # 10
        nn.ReLU(),
        # 11
        nn.Linear(50, 10),
        # 12
        nn.Softmax(),
    )


def get_lenet5_arch():
    """LeNet5 Inspired architecture."""
    return nn.Sequential(
        # 0
        nn.ConstantPad2d((2, 2, 2, 2), value=0),
        # 1
        nn.Conv2d(1, 6, kernel_size=5),
        # 2
        nn.MaxPool2d(2),
        # 3
        nn.ReLU(),
        # 4
        nn.Conv2d(6, 16, kernel_size=5),
        # 5
        nn.MaxPool2d(2),
        # 6
        nn.ReLU(),
        # 7
        nn.Flatten(),
        # 8
        nn.Linear(400, 120),
        # 9
        nn.ReLU(),
        # 10
        nn.Linear(120, 10),
        nn.Softmax(),
    )


class MNISTModel(L.LightningModule):

    def __init__(self, arch, params):
        super(MNISTModel, self).__init__()
        self.save_hyperparameters()
        self.lr = params["lr"]
        self.model = get_lenet5_arch() if arch == "lenet5" else get_udlbook_arch()

        self.criterion = nn.CrossEntropyLoss()

        smoothing_factor = 0.18  # Around 10 datapoints
        self.loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.f1_metric = MulticlassF1Score(num_classes=10, average="macro")
        self.accuracy_metric = MulticlassAccuracy(num_classes=10, average="macro")
        self.precision_metric = MulticlassPrecision(num_classes=10, average="macro")

        self.val_loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.val_f1_metric = MulticlassF1Score(num_classes=10, average="macro")
        self.val_accuracy_metric = MulticlassAccuracy(num_classes=10, average="macro")
        self.val_precision_metric = MulticlassPrecision(num_classes=10, average="macro")

        self.test_loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.test_f1_metric = MulticlassF1Score(num_classes=10, average="macro")
        self.test_accuracy_metric = MulticlassAccuracy(num_classes=10, average="macro")
        self.test_precision_metric = MulticlassPrecision(
            num_classes=10, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx, *args, **kwargs):
        x, y = train_batch

        x = x.reshape((len(x), 1, 28, 28))

        logits = self.model(x)

        loss = self.criterion(logits, y)

        self.loss_smoother.update(loss.item())
        self.f1_metric.update(logits, y)
        self.accuracy_metric.update(logits, y)
        self.precision_metric.update(logits, y)

        self.log("train_loss", loss)
        self.log("train_loss_smoothed", self.loss_smoother.get_smoothed_value())
        self.log("train_f1", self.f1_metric.compute().cpu().item())
        self.log("train_acc", self.accuracy_metric.compute().cpu().item())
        self.log("train_prec", self.precision_metric.compute().cpu().item())

        return loss

    def on_validation_start(self):
        self.val_loss_smoother.reset()
        self.val_f1_metric.reset()
        self.val_accuracy_metric.reset()
        self.val_precision_metric.reset()

    def validation_step(self, val_batch, *args, **kwargs):
        x, y = val_batch
        x = x.reshape((len(x), 1, 28, 28))
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.val_loss_smoother.update(loss.item())
        self.val_f1_metric.update(logits, y)
        self.val_accuracy_metric.update(logits, y)
        self.val_precision_metric.update(logits, y)

        self.log("val_loss", loss)
        self.log("val_loss_smoothed", self.val_loss_smoother.get_smoothed_value())
        self.log("val_f1", self.val_f1_metric.compute().cpu().item())
        self.log("val_acc", self.val_accuracy_metric.compute().cpu().item())
        self.log("val_prec", self.val_precision_metric.compute().cpu().item())
        return loss

    def test_step(self, test_batch, *args, **kwargs):
        x, y = test_batch
        x = x.reshape((len(x), 1, 28, 28))
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.test_loss_smoother.update(loss.item())
        self.test_f1_metric.update(logits, y)
        self.test_accuracy_metric.update(logits, y)
        self.test_precision_metric.update(logits, y)

        self.log("test_loss", loss)
        self.log("test_loss_smoothed", self.test_loss_smoother.get_smoothed_value())
        self.log("test_f1", self.test_f1_metric.compute().cpu().item())
        self.log("test_acc", self.test_accuracy_metric.compute().cpu().item())
        self.log("test_prec", self.test_precision_metric.compute().cpu().item())
        return loss

    def backward(self, loss, *args, **kwargs):
        loss.backward()
