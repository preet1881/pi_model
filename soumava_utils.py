import torch
import numpy as np
import torch.nn as nn
import os

import torch.nn.functional    as F
import torchvision.transforms as tf
import torchvision.datasets   as datasets

from torch.utils.data   import DataLoader
from torch.nn.utils     import weight_norm
from torch.utils.data   import Dataset
from timeit             import default_timer as timer
from scipy.io           import loadmat, savemat

def ramp_up(epoch , max_epochs ,max_val , mult):
    if epoch == 0:
         return 0.
    elif epoch >=max_epochs:
         return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)   #return as proposed in the paper

def weight_schedule(epoch ,max_epochs, max_val , mult , n_labeled, n_samples):
    max_val = max_val * (float(n_labeled/n_samples))
    return ramp_up(epoch , max_epochs, max_val, mult)

def prepare_CIFAR10(data_dir, normalize):
    if not normalize:
        transform_train = tf.Compose([tf.RandomHorizontalFlip(), tf.RandomAffine(degrees=20, translate=[0.1,0.1]), tf.ToTensor()])
        transform_test  = tf.Compose([tf.ToTensor()])
    else:
        m = 0.5
        st = 0.5
        normalize       = tf.Normalize(m, st)
        transform_train = tf.Compose([tf.RandomHorizontalFlip(), tf.RandomRotation(30), tf.ToTensor(), normalize])
        transform_test  = tf.Compose([tf.ToTensor(), normalize])

    train_dataset   = datasets.CIFAR10(root='../data', train=True, transform=transform_train, download=True)
    test_dataset    = datasets.CIFAR10(root='../data',train=False,transform=transform_test)
    num_classes = 10; num_train_samples = 50000; num_test_samples = 10000; res_dir = 'CIFAR10'
    return num_classes, train_dataset, test_dataset, res_dir, num_train_samples, num_test_samples


class DataFolder(Dataset):
    def __init__(self, data, labels, transforms):
        self.to_pil     = tf.ToPILImage()
        self.data       = data
        self.targets    = labels - 1
        self.transforms = transforms
        assert(self.targets.min() == 0)

    def __len__(self):
        return self.targets.numel()

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        data   = self.data[index]
        data   = self.to_pil(data)# THIS IS NOW A PIL TENSOR
        if self.transforms is not None:
            data = self.transforms(data)
        labels = int(self.targets[index])
        return data, labels

def prepare_CIFAR10_MAT(file_name):
    DATA   = loadmat(file_name)
    data   = torch.from_numpy(DATA['data'])
    labels = torch.from_numpy(DATA['labels'])
    labels = labels.view(-1)
    set    = torch.from_numpy(DATA['set'])
    set    = set.view(-1)

    idx_train       = (set==1).nonzero()
    data_train      = data[idx_train.view(-1)]
    labels_train    = labels[idx_train.view(-1)]
    transform_train = tf.Compose([tf.RandomHorizontalFlip(), tf.RandomAffine(degrees=20, translate=(0.0625, 0.0625)), tf.ToTensor()])
    train_dataset   = DataFolder(data=data_train, labels=labels_train, transforms=transform_train)

    idx_test       = (set==3).nonzero()
    transform_test = tf.ToTensor()
    data_test      = data[idx_test.view(-1)]
    labels_test    = labels[idx_test.view(-1)]
    test_dataset   = DataFolder(data=data_test, labels=labels_test, transforms=transform_test)

    num_classes = 10; num_train_samples = 50000; num_test_samples = 10000; res_dir = 'CIFAR10'
    return num_classes, train_dataset, test_dataset, res_dir, num_train_samples, num_test_samples

def prepare_mnist(data_dir, normalize):
    # normalize data
    if not normalize:
        transform_train = tf.Compose([tf.ToTensor()])
        transform_test = tf.Compose([tf.ToTensor()])
    else:
        m = TODO; st = TODO
        normalize       = tf.Normalize(m, st)
        transform_train = tf.Compose([tf.ToTensor(), normalize])
        transform_test  = tf.Compose([tf.ToTensor(), normalize])

    train_dataset = datasets.MNIST(root=data_dir, train=True,  transform=transform_train, download=True)
    test_dataset  = datasets.MNIST(root=data_dir, train=False, transform=transform_test)
    num_classes = 10; num_train_samples = 60000; num_test_samples = 10000; res_dir = 'MNIST'
    return num_classes, train_dataset, test_dataset, res_dir, num_train_samples, num_test_samples

def sample_train(train_dataset, test_dataset, batch_size, labeled_data, n_classes,
                 num_workers, shuffle_train=True ,return_idxs = True):
    n       = len(train_dataset)
    cpt     = 0
    indices = torch.zeros(labeled_data)
    other   = torch.zeros(n-labeled_data)
    labeled_data_perclass = labeled_data// n_classes
    for i in range(n_classes):
        train_dataset.targets = torch.Tensor(train_dataset.targets)
        class_items = (train_dataset.targets == i).nonzero()
        n_class     = len(class_items)
        rd          = np.random.permutation(np.arange(n_class))

        indices[i*labeled_data_perclass: (i+1)*labeled_data_perclass] = class_items[rd[:labeled_data_perclass],0].type(torch.FloatTensor)
        other[cpt:cpt+n_class - labeled_data_perclass] = class_items[rd[labeled_data_perclass:],0]
        cpt += n_class - labeled_data_perclass

    other = other.long()
    train_dataset.targets[other] = -1
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = shuffle_train)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = False)
    if return_idxs:
        return train_loader, test_loader, indices
    return train_loader, test_loader, None

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std = 0.05):
        super(GaussianNoise,self).__init__()
        self.mean = mean
        self.std  = std

    def forward(self,x):
        noise = torch.randn(x.size()) * self.std + self.mean
        return x+noise

class CNN(nn.Module):
    def __init__(self, mean, std, p):
        super(CNN, self).__init__()
        self.mean   = mean
        self.std    = std
        self.drop   = p
        self.gn     = GaussianNoise(mean=self.mean, std=self.std)
        self.act    = nn.LeakyReLU(0.1)
        self.drop   = nn.Dropout(p)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=1))
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=1))
        self.mp     = nn.MaxPool2d(2, stride=2, padding=1)
        self.fc     = nn.Linear(4608, 10)

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
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

def get_hyper_parameters(need, optimizer):
    setting = 'R' + str(need)
    if setting == 'R1':
        lr, wd = 0.0001, 0.0001
    elif setting == 'R2':
        lr, wd = 0.0001, 0.0005
    elif setting == 'R3':
        lr, wd = 0.0005, 0.0001
    elif setting == 'R4':
        lr, wd = 0.0005, 0.0005
    elif setting == 'R5':
        lr, wd = 0.001, 0.0001
    elif setting == 'R6':
        lr, wd = 0.001, 0.0005
    elif setting == 'R7':
        lr, wd = 0.005, 0.0005
    elif setting == 'R8':
        lr, wd = 0.005, 0.0005
    else:
        raise ValueError("Wrong HyperParameters Setting")

    if optimizer.lower() == 'rmsprop':
        lr = lr
    elif optimizer.lower() == 'sgd':
        lr = lr * 100
    elif optimizer.lower() == 'adam':
        lr = lr* 2
    else:
        raise ValueError("Wrong Optimizer")
    return lr, wd

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum    += val * n
        self.count  += n
        self.avg    = self.sum / self.count

def temporal_loss(out1, out2 , w , labels):
    #use mse for calculating unsup loss
    def mse_loss(out1,out2):
        quad_diff = torch.sum((F.softmax(out1,dim =1 )- F.softmax(out2, dim = 1))**2)
        return quad_diff/ out1.data.nelement()
    #use to caculate sup loss
    def masked_crossentropy(out , labels):
        cond             = (labels >=0)
        labels_in_batch  = torch.nonzero(cond)
        number_of_labels = len(labels_in_batch)
        if number_of_labels >0 :
            masked_outputs = torch.index_select(out, 0 , labels_in_batch.view(number_of_labels))
            labels         = labels.to(dtype = torch.long)
            masked_labels  = labels[cond]
            loss           = F.cross_entropy(masked_outputs , masked_labels)
            return loss , number_of_labels
        return torch.FloatTensor([0.]),0 # You cannot back propagate in this case

    sup_loss , number_of_labels = masked_crossentropy(out1 , labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w* unsup_loss, sup_loss, unsup_loss , number_of_labels

def train(epoch, num_epochs, model, train_loader, w, optimizer, use_cuda, c=200):
        model.train()
        losses       = AverageMeter()
        sup_losses   = AverageMeter()
        unsup_losses = AverageMeter()
        processed_data = 0
        total_steps    = len(train_loader)
        t = timer()
        for i, (images, labels) in enumerate(train_loader):
            batch_size = images.size(0)
            processed_data += batch_size
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            out1   = model(images)  #z
            out2 =model(images)    #zcap
           

            loss, suploss, unsuploss, number_of_labels = temporal_loss(out1, out2, w, labels)

            losses.update(loss.item(), batch_size)
            sup_losses.update(suploss.item(), batch_size)
            unsup_losses.update(unsuploss.item(), batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % c == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss :%.6f, Time (this epoch): %.2f s' % (epoch + 1, num_epochs, i + 1, total_steps, loss.item(), timer() - t))
        return losses, sup_losses, unsup_losses

def test(model, test_loader, use_cuda):
    correct = 0
    total = 0
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        out = model(images)
        _, predicted = torch.max(out, 1)
        total += labels.size(0)
        correct += (predicted == labels.data.view_as(predicted)).sum()
    acc = 100.0 * float(correct) / total
    return acc

def save_checkpoint( state, epoch, save_dir ):
    model_file = os.path.join(save_dir, 'model%s.pth.tar' % (str(epoch)))
    torch.save(state, model_file)

def save_best_checkpoint( state, save_dir, suffix ):
    if suffix is None:
        best_file = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        best_file = os.path.join(save_dir, 'model_best_' + suffix + '.pth.tar')
    torch.save(state, best_file)
