from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.gridspec as gsp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as tf

#right now gaussian noise is not adding any noise just returning x as it is
class GaussianNoise(nn.Module):
    def __init__(self,batch_size,input_shape = (3,32,32), std = 0.05):
        super(GaussianNoise,self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape))
        self.std = std

    def forward(self,x):
        self.noise.data.normal_(0,std=self.std)
        return x + self.noise


def prepare_CIFAR10():
    # normalize data
    m = (0.5,)
    st = (0.5,)
    normalize = tf.Normalize(m, st)
    transform_train = tf.Compose(
        [tf.RandomHorizontalFlip(), tf.RandomRotation(30), tf.ToTensor(), normalize])
    # load train data
    train_dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        transform=transform_train,
        download=True)

    # load test data
    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        transform=tf.Compose([tf.ToTensor(), normalize]))

    return train_dataset, test_dataset


def prepare_mnist():
    # normalize data
    m = (0.5,)
    st = (0.5,)
    normalize = tf.Normalize(m, st)

    # load train data
    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        transform=tf.Compose([tf.ToTensor(), normalize]),
        download=True)

    # load test data
    test_dataset = datasets.MNIST(
        root='../data',
        train=False,
        transform=tf.Compose([tf.ToTensor(), normalize]))

    return train_dataset, test_dataset


def ramp_up(epoch , max_epochs ,max_val , mult):
    if epoch == 0:
         return 0.

    elif epoch >=max_epochs:
         return max_val

    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)   #return as proposed in the paper


def weight_schedule(epoch ,max_epochs, max_val , mult , n_labeled, n_samples):
    max_val = max_val * (float(n_labeled/n_samples))
    return ramp_up(epoch , max_epochs, max_val, mult)

#this is used to calculate accuracy and test the model
def calc_metrics(model , loader):
    correct= 0
    total = 0

    for i , (samples, labels) in enumerate(loader):
        samples = Variable(samples, volatile = True)
        labels = Variable(labels)
        outputs = model(samples)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted ==labels.data.view_as(predicted)).sum()

    acc = 100*float(correct)/total
    return acc


def savetime():
    return datetime.now().strftime('%Y_%m_%d_%H%M%S')
