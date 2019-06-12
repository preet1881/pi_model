import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config_
from pi_model_ import train
from utils_ import GaussianNoise, savetime
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, batch_size,std, p=0.5,fm1=16,fm2 =32):
        super(CNN,self).__init__()
        self.fm1=fm1
        self.fm2 = fm2
        self.std = std
        self.gn = GaussianNoise(batch_size, std =self.std)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1=weight_norm(nn.Conv2d(1,self.fm1,3,padding =1))
        self.conv2= weight_norm(nn.Conv2d(self.fm1, self.fm2 ,3, padding=1))
        self.mp = nn.MaxPool2d(3,stride =2 , padding=1)
        self.fc = nn.Linear(self.fm2*7*7 ,10)

    def forward(self,x):
        if self.training:
            x= self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view( -1, self.fm2*7*7)
        x = self.drop(x)
        x = self.fc(x)
        return x



model = CNN(config_.batch_size, config_.std, p=0.5, fm1=16, fm2=32)
# metrics
accs = []
accs_best = []
losses = []
sup_losses = []
unsup_losses = []
idxs = []

ts = savetime()
cfg = vars(config_)


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
    print("the number of labelled data used = %s" % config_.labeled_data)
    config_.labeled_data = config_.labeled_data + 100
    