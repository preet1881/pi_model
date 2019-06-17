import numpy as np
from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils_ import calc_metrics, prepare_mnist, weight_schedule ,prepare_CIFAR10

#data loader
def sample_train(train_dataset, test_dataset , batch_size ,
                 labeled_data , n_classes,seed, shuffle_train= True ,return_idxs = True):


                 n = len(train_dataset)
                 cpt = 0
                 indices = torch.zeros(labeled_data)

                 other = torch.zeros(n-labeled_data)
                 labeled_data_perclass = labeled_data// n_classes

                 for i in range(n_classes):
                     train_dataset.targets = torch.Tensor(train_dataset.targets)
                     class_items = (train_dataset.targets == i).nonzero()   #for mnist we use train_labels to read the lables but for cifar this gives error
                     n_class = len(class_items)
                     rd = np.random.permutation(np.arange(n_class))

                     indices[i*labeled_data_perclass: (i+1)*labeled_data_perclass]=class_items[rd[:labeled_data_perclass],0].type(torch.FloatTensor)
                     other[cpt:cpt+n_class - labeled_data_perclass] = class_items[rd[labeled_data_perclass:],0]
                     cpt += n_class - labeled_data_perclass


                 other = other.long()
                 train_dataset.targets[other] = -1

                 train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=batch_size,
                                                           num_workers=4,
                                                           shuffle=shuffle_train)
                 test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          num_workers=4,
                                                          shuffle=False)

                 if return_idxs:
                     return train_loader, test_loader, indices
                 return train_loader, test_loader




def temporal_loss(out1, out2 , w , labels):
    #use mse for calculating unsup loss
    def mse_loss(out1,out2):
        quad_diff = torch.sum((F.softmax(out1,dim =1 )- F.softmax(out2, dim = 1))**2)
        return quad_diff/ out1.data.nelement()

    #use to caculate sup loss
    def masked_crossentropy(out , labels):
        cond = (labels >=0)
        labels_in_batch = torch.nonzero(cond)
        number_of_labels = len(labels_in_batch)
        if number_of_labels >0 :
            masked_outputs = torch.index_select(out, 0 , labels_in_batch.view(number_of_labels))
            labels = labels.to(dtype = torch.long)
            masked_labels = labels[cond]
            loss = F.cross_entropy(masked_outputs , masked_labels)

            return loss , number_of_labels
        return Variable(torch.FloatTensor([0.]),requires_grad = False),0

    sup_loss , number_of_labels = masked_crossentropy(out1 , labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w* unsup_loss, sup_loss, unsup_loss , number_of_labels


#change num_epochs for better training here and in config_ too
def train( model , seed , labeled_data =4000 , alpha =0.6 , lr = 0.002 , beta2 =0.99
           , num_epochs =150 , batch_size =100, drop =0.5 , std =0.15 , fm1=16 ,fm2 =32, divide_by_bs = False
           , w_norm = False , data_norm= 'pixelwise', early_stop = None, c=300 , n_classes =10 , max_epochs=80,
           max_val= 30. , ramp_up_mult= -5 , n_samples= 60000 , print_res=True , **kwargs):


           #retrive data
           train_dataset, test_dataset = prepare_CIFAR10()
           ntrain = len(train_dataset)

           #make data loaders
           train_loader, test_loader, indices = sample_train(train_dataset, test_dataset, batch_size,
                                                             labeled_data, n_classes, seed, shuffle_train=False)

           # setup param optimization
           optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

           #train the model
           model.train()
           losses = []
           sup_losses = []
           unsup_losses = []
           best_loss = 20.

           z = torch.zeros(ntrain, n_classes).float()
           outputs = torch.zeros(ntrain, n_classes).float()  # current outputs


           for epoch in range(num_epochs):
               t = timer()
               #evaluate unsup cost weight
               w = weight_schedule(epoch , max_epochs, max_val, ramp_up_mult, labeled_data, n_samples)

               if(epoch+1)% 10 ==0:
                   print('unsupervised loss weight : {}'.format([w]))

               w = torch.autograd.Variable(torch.FloatTensor([w]), requires_grad = False)

               l = []
               supl=[]
               unsupl=[]
               for i , (images,labels) in enumerate(train_loader):
                   images = Variable(images)
                   labels = Variable(labels, requires_grad = False)

                   #get output and calculate loss
                   optimizer.zero_grad()
                   out = model(images)
                   zcomp = Variable(z[i*batch_size:(i+1)*batch_size], requires_grad = False)
                   loss , suploss , unsuploss , number_of_labels = temporal_loss(out , zcomp,w , labels)
                   #save output and losses
                   outputs[i* batch_size : (i+1)*batch_size]= out.data.clone()
                   l.append(loss.item())
                   supl.append(number_of_labels*suploss.item())
                   unsupl.append(unsuploss.item())

                   #backdrop
                   loss.backward()
                   optimizer.step()
                   #print loss
                   if (epoch+1)%10 ==0:
                       if i+1 ==2 *c :
                           print('Epoch [%d/%d], Step [%d/%d], Loss :%.6f, Time (this epoch): %.2f s'%(epoch+1, num_epochs, i+1 , round(len(train_dataset)//batch_size) , np.mean(l), timer()-t))
                       elif (i+1)%c ==0:
                           print('Epoch [%d/%d], Step [%d/%d], Loss: %.6f'% (epoch + 1, num_epochs, i + 1, round(len(train_dataset) // batch_size), np.mean(l)))


               eloss =np.mean(l)
               losses.append(eloss)
               sup_losses.append((1./labeled_data)*np.sum(supl))
               unsup_losses.append(np.mean(unsupl))

               if eloss < best_loss:

                   best_loss = eloss
                   torch.save({'state_dict': model.state_dict()}, 'model_best.csv')


           model.eval()
           acc = calc_metrics(model, test_loader)
           if print_res:
               print('Accuracy of the network on the 10000 test images: %.2f %%' % (acc))

           checkpoint = torch.load('model_best.csv')
           model.load_state_dict(checkpoint['state_dict'])
           model.eval()
           acc_best = calc_metrics(model, test_loader)
           if print_res:
               print('Accuracy of the network (best model) on the 10000 test images: %.2f %%' % (acc_best))


           return acc, acc_best, losses, sup_losses, unsup_losses, indices
