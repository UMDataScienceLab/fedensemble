import pickle
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import modeloperations as mo
import languageprocess as lp
from collections import OrderedDict

from testhandling import test_acc, test_loss

class local_language_dataset(torch.utils.data.Dataset):
    def __init__(self, wholedata, indices):
        self.all_data = wholedata
        self.indices = indices

    def __getitem__(self, index):
        x = self.all_data[self.indices[index]]
        return torch.tensor(x)

    def __len__(self):
        return len(self.indices)

    def some_function(self):
        pass

class local_dataset(torch.utils.data.Dataset):
    def __init__(self, wholedata, indices):
        self.all_data = wholedata
        self.indices = indices

    def __getitem__(self, index):
        img, label = self.all_data[self.indices[index]]
        return torch.tensor(img), torch.tensor(label)

    def __len__(self):
        return len(self.indices)

    def some_function(self):
        pass

class client():
    def __init__(self, source_model, dataset, dataset_indices, testset, test_indices, args):
        self.model = copy.deepcopy(source_model)
        self.rz_model = None
        self.c = None
        self.ci = None
        self.ci_new = None

        #if len(self.model) > 0:
        #    self.model = self.model[0]

        #char2int = {ch: ii for ii, ch in int2char.items()}
        if args['dataset-type'] in {'language'}:
            #chars = self.model.chars
            #int2char = dict(enumerate(chars))
            self.dataset = np.array([dataset[i] for i in dataset_indices])
            self.testset = np.array([testset[i] for i in test_indices])

            #print("new client")
            #print("".join([int2char[i] for i in self.dataset]))
        else:
            self.dataset = local_dataset(dataset, dataset_indices)
            self.testset = local_dataset(testset, test_indices)

        self.model_list = []
        self.samples_along_trajectory = []
        self.type = args['dataset-type']

  
    def get_weight(self, source_model_list, i):
        #self.model = pickle.loads(pickle.dumps(source_model_list[i]))
        self.model = copy.deepcopy(source_model_list[i])

  

    def local_update(self, optimizer_method, epoch, batch_size, device):
        self.samples_along_trajectory = []
        #print('at the beginning of local udate, model l2 norm sq is {}'.format(mo.inner_p(self.model, self.model).item()))
        if optimizer_method['local'] == 'adam':
            #optimizer = optim.Adam(self.model.parameters(), lr=optimizer_method['initial_lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=optimizer_method['weight_decay'],
            #           amsgrad=False)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer_method['initial_lr'])
        elif optimizer_method['local'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), momentum=optimizer_method['momentum'], lr=optimizer_method['initial_lr'], weight_decay=optimizer_method['weight_decay'])
      
        else:
            raise Exception('No such optimizer: '+optimizer_method)
        #model0 = pickle.loads(pickle.dumps(self.model)).to(device)
        model0 = copy.deepcopy(self.model).to(device)
        the_model = self.model.to(device)
    
        if self.type in {'language'}:
            batches = lp.get_batches(self.dataset, batch_size, optimizer_method['seq_length'])
            h = the_model.init_hidden(batch_size)
        else:
            data_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)
       
        sample_intervals = self.dataset.__len__()//optimizer_method['number_of_samples_sent_back']


        for j in range(epoch):
            tot_loss = 0
            nos = 0
            l2n = 0
            if self.type in {'language'}:
                #batches = lp.get_batches(self.dataset, batch_size, optimizer_method['seq_length'])
                h = the_model.init_hidden(batch_size)
                i = 0
                for x, y in lp.get_batches(self.dataset, batch_size, optimizer_method['seq_length']):
                    i += 1
                    nos += 1
                    # One-hot encode our data and make them Torch tensors
                    x = lp.one_hot_encode(x, len(the_model.chars))
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    if optimizer_method['device'] == 'cuda':
                        inputs, targets = inputs.cuda(), targets.cuda()

                    h = tuple([each.data for each in h])
                    the_model.zero_grad()
                    output, h = the_model(inputs, h)

                    loss = nn.CrossEntropyLoss()(output, targets.view(batch_size * optimizer_method['seq_length']).long())
                    tot_loss += loss.item()
                    if optimizer_method['fed'] in {'fedprox'}:
                        alpha = optimizer_method['mu']
                        beta = 0.00001
                        # epsilon = torch.randn(1).to(device)
                        add_loss = torch.zeros(1).to(device)

                        state_dict = the_model.state_dict()
                        weight_keys = list(state_dict.keys())
                        state_dict0 = model0.state_dict()
                        for key in weight_keys:
                            add_loss = add_loss + alpha * torch.sum(
                                (state_dict[key] - state_dict0[key]) * (state_dict[key] - state_dict0[key]))
                        loss = loss + add_loss
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    nn.utils.clip_grad_norm_(the_model.parameters(), 5)
                    optimizer.step()
                    l2n += mo.gradient_l2_norm(the_model)
                 
            else:
                for i, (images, labels) in enumerate(data_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    if True:
                        outputs = the_model(images)
                        if optimizer_method['task'] == 'classification':
                            loss = nn.CrossEntropyLoss()(outputs, labels)
                        elif optimizer_method['task'] == 'regression':
                            loss = nn.MSELoss()(outputs, labels)#.norm(2)
                            #print("outputs",outputs)
                            #print("labels", labels)
                        else:
                            raise Exception('Undefined task: ' + optimizer_method['task'])

                        if optimizer_method['fed'] in {'fedprox', 'fedensemble'}:
                            alpha = optimizer_method['mu']

                            beta = 0.00001
                            #epsilon = torch.randn(1).to(device)
                            add_loss = torch.zeros(1).to(device)

                            state_dict = the_model.state_dict()
                            weight_keys = list(state_dict.keys())
                            state_dict0 = model0.state_dict()

                            # totlen = 0.
                            
                            for key in weight_keys:
                                
                                add_loss = add_loss + alpha * torch.sum(
                                        (state_dict[key] - state_dict0[key]) * (state_dict[key] - state_dict0[key]))
                            loss1 = loss + add_loss[0]#/totlen
                        else:
                            loss1 = loss
                        #loss = nn.CrossEntropyLoss()(outputs, labels)
                        #print(mo.gradient_l2_norm(the_model))
                        optimizer.zero_grad()
                        loss1.backward()
                        optimizer.step()
                        tot_loss += loss.item()
                        nos += 1

                        l2n += mo.gradient_l2_norm(the_model)
                        if j==0:
                            self.ci_new = mo.reconstruct_gradient(the_model)
                    #print('[{},{}], loss is {}'.format(j,i,loss.item()))
        return the_model.cpu(), tot_loss / nos, l2n/nos

