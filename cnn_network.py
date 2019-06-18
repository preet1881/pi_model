import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config_
from pi_model_ import train
from utils_ import GaussianNoise, savetime
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CNN(nn.Module):

    def __init__(self, batch_size, std,p=0.5):
        super(CNN, self).__init__()

        self.std = std
        self.gn = GaussianNoise(batch_size, std=self.std)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(p)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=1))
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=1))
        self.mp = nn.MaxPool2d(2, stride=2, padding=1)
        self.fc = nn.Linear(4608, 10)

    def forward(self, x):
        batch_size = x.size(0)

        if self.training:
            x = self.gn(x)
        x = self.act(self.conv1a(x))
        x = self.act(self.conv1b(x))
        x = self.act(self.conv1c(x))
        x = self.mp(x)

        x = self.drop(x)
        x = self.act(self.conv2a(x))
        x = self.act(self.conv2b(x))
        x = self.act(self.conv2c(x))
        x = self.mp(x)

        x = self.drop(x)
        x = self.act(self.conv3a(x))
        x = self.act(self.conv3b(x))
        x = self.act(self.conv3c(x))
        x = self.mp(x)

        # x = self.act(self.mp(self.conv1(x)))
        # x = self.act(self.mp(self.conv2(x)))

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


model = CNN(config_.batch_size, config_.std)
print(model)
# metrics
accs = []
accs_best = []
losses = []
sup_losses = []
unsup_losses = []
idxs = []

ts = savetime()
cfg = vars(config_)


def print_graph(x,label):
    plt.plot(x, label='label')
    plt.show()


# we calculate accuracy of the model against number of labels
#as we set loops as 5 so the model will train 5 times with diffrent number of labels each time
for i in range(cfg['loops']):

    model = CNN(cfg['batch_size'],cfg['std'])
    seed = cfg['seeds'][i]
    acc, acc_best ,l, sl, usl,indices =train(model,seed,**cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)
    print_graph(losses,losses)
    print_graph(sup_losses,sup_losses)
    print_graph(unsup_losses, unsup_losses)
    print("the number of labelled data used = %s" % config_.labeled_data)
    config_.labeled_data = config_.labeled_data + 100
    