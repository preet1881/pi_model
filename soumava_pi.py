import torch
import argparse
import os
import pandas as pd
import numpy as np
import torch.optim as optim
import datetime
import time
from soumava_utils import prepare_mnist, prepare_CIFAR10, sample_train, CNN, get_hyper_parameters, weight_schedule, \
                          train, test, prepare_CIFAR10_MAT, save_best_checkpoint
torch.set_default_tensor_type('torch.FloatTensor')
'''
--------------------------- ARGUMENT PARSER ---------------------------------
'''
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',   type = int,   default = 200)  # Number of Epochs
parser.add_argument('--num_workers',  type = int,   default = 4)  # Number of Workers
parser.add_argument('--batch_size',   type = int,   default = 100)  # Batch Size
parser.add_argument('--momentum',     type = float, default = 0.9)  # Momentum for SGD
parser.add_argument('--use_cuda',     type = int,   default = 1)  # use cuda
'''
--------------------------------------------------------------------------------
'''
parser.add_argument('--model_type',   type = str,   default = 'pi')
parser.add_argument('--dataset',      type = str,   default = 'cifar10')  # Dataset
parser.add_argument('--labeled_data', type = int,   default = 4000)
parser.add_argument('--need', 		  type = int,   default = '1')
parser.add_argument('--optimizer', 	  type = str,   default = 'RMSProp')
parser.add_argument('--normalize',    type = int,   default = 0) # 0 for no normalization|1 for normalization
parser.add_argument('--drop_rate', 	  type = float, default = '0.5')
parser.add_argument('--mean_noise',   type = float, default = 0.0)
parser.add_argument('--std_noise',    type = float, default = 0.1)
'''
--------------------------------------------------------------------------------
'''
parser.add_argument('--data_dir',     type = str, default = '')
parser.add_argument('--main_res_dir', type = str, default = 'RESULTS-PI')
parser.add_argument('--contrast_norm', action='store_true', default=False)
parser.add_argument('--zca',           action='store_true', default=False)
'''
--------------------------------------------------------------------------------
'''
rng    = np.random.RandomState(42)
seed   = rng.randint(200)
config = parser.parse_args()
print(config)
torch.manual_seed(seed)
if torch.cuda.is_available() and config.use_cuda == 1:
    config.cuda = 1
else:
    config.cuda = 0

if not (config.contrast_norm) and not (config.zca):
    print ("WITHOUT USING CONTRAST NORMALIZATION or ZCA")
    if config.dataset.lower() == 'cifar10':
        print ("USING CIFAR 10 WITH NORMALIZATION SET TO {}".format(config.normalize))
        num_classes, train_dataset, test_dataset, res_dir, num_train_samples, num_test_samples = prepare_CIFAR10(normalize=config.normalize, data_dir=config.data_dir)
    else:
        print ("USING MNIST WITH NORMALIZATION SET TO {}".format(config.normalize))
        num_classes, train_dataset, test_dataset, res_dir, num_train_samples, num_test_samples = prepare_mnist(normalize=config.normalize, data_dir=config.data_dir)
else:
    assert(config.dataset.lower() == 'cifar10')
    print ("CIFAR10"),
    if (config.contrast_norm) and (config.zca):
        print ("CONTRAST NORM + ZCA")
        data_mat_file_name = 'cifar10_constrast_ZCA.mat'
    elif (config.contrast_norm):
        print ("CONTRAST NORM")
        data_mat_file_name = 'cifar10_constrast.mat'
    else:
        print ("ZCA")
        data_mat_file_name = 'cifar10_ZCA.mat'

    #data_mat_file_name = os.path.join(config.data_dir, data_mat_file_name)
    #num_classes, train_dataset, test_dataset, res_dir, num_train_samples, num_test_samples = prepare_CIFAR10()

train_loader, test_loader, indices = sample_train(train_dataset=train_dataset, test_dataset=test_dataset,
                                                  batch_size=config.batch_size, labeled_data=config.labeled_data,
                                                  n_classes=num_classes, num_workers=config.num_workers,shuffle_train=True)
model  = CNN(mean=config.mean_noise, std=config.std_noise, p=config.drop_rate)
lr, wd = get_hyper_parameters(need=config.need, optimizer=config.optimizer)
if config.optimizer.lower() == 'rmsprop':
    print ("USING RMSPROP OPTIMIZER")
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
elif config.optimizer.lower() == 'adam':
    print ("USING ADAM OPTIMIZER")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
else:
    print ("USING SGD OPTIMIZER")
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=config.momentum)

res_dir = os.path.join(res_dir+ '-' + str(config.labeled_data), config.optimizer.upper() + '-DROP-'+str(config.drop_rate) + '-NEED-'+str(config.need))
if config.cuda:
    model   = model.cuda()

save_dir = os.path.join(config.main_res_dir, res_dir)
print ("Saving in the Directory", save_dir)
if not os.path.exists(save_dir):
    print ('Creating the new Directory')
    os.makedirs(save_dir)
'''
TODO
LOADING AND RESTORING PREVIOUS MODELS
'''
results     = pd.DataFrame( index=np.arange(config.num_epochs), columns={'LOSS', 'ACC', 'UNSUP', 'SUP'})
file_suffix = res_dir+ '-' + str(config.labeled_data) + '-' + config.optimizer.upper() + '-DROP-'+str(config.drop_rate) + '-NEED-'+str(config.need)
best_acc    = 0.0
best_model  = model
best_epoch  = 1
csv_file_name = 'results_'+file_suffix+'.csv'
train_time = 0
start_time = time.time()
for epoch in range(1, config.num_epochs + 1):
    start_train_time = time.time()
    print ("Training begins for epoch {}".format(epoch))
    w = weight_schedule(epoch=epoch, max_epochs=config.num_epochs, max_val=80, mult=-5, n_labeled=config.labeled_data, n_samples=num_train_samples)
    loss_epoch, sup_loss_epoch, unsup_loss_epoch = train(epoch=epoch, num_epochs=config.num_epochs, model=model,
                                    use_cuda=config.cuda, train_loader=train_loader, w=w,  optimizer=optimizer, c=200)
    train_time += round(time.time() - start_train_time)
    acc_epoch = test(model=model, test_loader=test_loader, use_cuda=config.cuda)
    if acc_epoch > best_acc:
        best_acc   = acc_epoch
        best_model = model
        best_epoch = epoch

    print ("MODEL PI  OVER-ALL LOSS::{:.2f}   ACCURACY::{}".format(loss_epoch.avg, acc_epoch))
    results.loc[epoch] = pd.Series({'LOSS': loss_epoch.avg, 'ACC': acc_epoch, 'UNSUP':sup_loss_epoch.avg, 'SUP':sup_loss_epoch.avg})
    results.to_csv(os.path.join(save_dir, 'results.csv'), columns=['ACC', 'UNSUP', 'SUP', 'LOSS'])

    epoch_time = round(time.time() - start_train_time)
    epoch_time = str(datetime.timedelta(seconds=epoch_time))
    print("Total elapsed epoch:: {} time ::(h:m:s): {}".format(epoch, epoch_time))

elapsed    = round(time.time() - start_time)
elapsed    = str(datetime.timedelta(seconds=elapsed))
train_time = str(datetime.timedelta(seconds=train_time))

print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
print ("-------------------------------- BEST ------------------------------------------")
print (" Model PI ----> best_epoch::{}  best_acc::{:.2f} ".format(best_epoch, best_acc))
save_best_checkpoint(state={'best_epoch':best_epoch, 'best_model':best_model.state_dict(), 'best_acc':best_acc},
                     save_dir=save_dir, suffix=file_suffix)
