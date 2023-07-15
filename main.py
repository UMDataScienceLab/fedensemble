import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from client import client
from network import CNNMnist, Twohiddenlayerfc,  Resnet,  CharRNN
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import time
import copy
import pickle
from mip import Model, xsum, BINARY
import matplotlib.pyplot as plt

import languageprocess as lp
from buffer import Memory
import modeloperations as mo
from server import *

import utils
import federatedlearning as fl
from testhandling import test_acc, test_loss

def create_client_list(model, dataset, test_dataset=None, args={}):
    how_many_clients = args['number_of_clients']
    client_list = []
    impv = []
    if args['fed'] in {'fedbe'}:
        len_ds = int(dataset.__len__()*0.8)
        validation_idx = list(range(len_ds, dataset.__len__()))
        validation_client = client(model, dataset, validation_idx, args)

    else:
        len_ds = dataset.__len__()
        validation_client = []
    mu = len_ds / (how_many_clients)
    test_mu = test_dataset.__len__()/ how_many_clients
    log('There are {} train datapoints'.format(len_ds))
    if args['DATASET'] in {'SINE-noniid'}:
        stt = 0
        test_stt = 0
        for i in range(how_many_clients):
            client_i = client(model, dataset, range(int(stt),int(stt+mu)), test_dataset,range(int(test_stt),int(test_stt+test_mu)),  args)
            stt+=mu
            client_list.append(client_i)
            impv = torch.ones(how_many_clients)/how_many_clients
    else:
        if args['noniid'] == 1 and args['DATASET'] in {'MNIST','CIFAR10'}:
            # if we want to use non iid partition
            idx=[]
            test_idx = []
            if args['DATASET'] == 'MNIST':
                labels = dataset.train_labels.numpy()
                test_labels = test_dataset.train_labels.numpy()
            else:
                labels = np.array(dataset.targets)
                test_labels = test_dataset.targets
            for i in range(10):
                print(np.where(labels == i)[0])
                idx.append(np.where(labels == i)[0])
                test_idx.append(np.where(test_labels == i)[0])
            for i in range(10):
                random.shuffle(idx[i])
                random.shuffle(test_idx[i])
            start = [0 for i in range(10)]
            test_start = [0 for i in range(10)]

            # use integer programming to determine the assignment
            c = 10
            lc = args['largest_categories']
            queens = Model()
            x = [[queens.add_var('x({},{})'.format(i, j), var_type=BINARY)
                  for j in range(c)] for i in range(how_many_clients)]
            # lc per row
            for i in range(how_many_clients):
                queens += xsum(x[i][j] for j in range(c)) == lc, 'row({})'.format(i)

            # (how_many_clients*lc)/(10) per column
            for j in range(c):
                queens += xsum(x[i][j] for i in range(how_many_clients)) == (how_many_clients*lc)/c, 'col({})'.format(j)
            queens.optimize()

            res = []
            if queens.num_solutions:
                for i, v in enumerate(queens.vars):
                    if v.x >= 0.99:
                        res.append([i // c, i % c])
            else:
                raise Exception('Cannot find a proper assignment')
            #raise Exception('Assignment created')

            # print('length of res is:{}'.format(len(res)))
            dt = int(mu / lc)
            test_dt = int(test_dataset.__len__() / (lc*how_many_clients))

            #calculate statistics about assignment
            all_assignment = set()
            for i in range(0,len(res),lc):
                to_concatenate = []
                test_to_concatenate = []
                which_client = i//lc
                which_class = []
                for j in range(lc):
                    # print(res[i*lc+j][1])
                    try:
                        which_class.append(res[i+j][1])
                        st = start[res[i+j][1]]
                        test_st = test_start[res[i + j][1]]

                    except:
                        print('error at this position:')
                        print(i  + j)
                        print("i is {}, lc is {}, j is {}".format(i,lc,j))
                        print(res[i+j][1])

                    to_concatenate.append(idx[res[i+j][1]][st:(st+dt)])
                    test_to_concatenate.append(test_idx[res[i+j][1]][test_st:(test_st+test_dt)])

                    start[res[i+j][1]] = st + dt
                    test_start[res[i + j][1]] = test_st + test_dt

                which_class = np.sort(np.array(which_class))
                hash_class = sum([c**i*which_class[i] for i in range(len(which_class))])
                all_assignment.add(hash_class)
                
                #slice = [res[i][1],res[i+1][1]]
                #st1 = start[slice[0]]
                #st2 = start[slice[1]]
                index_i = copy.deepcopy(np.concatenate((to_concatenate)))
                test_index_i = copy.deepcopy(np.concatenate((test_to_concatenate)))

                #index_i = copy.deepcopy(np.concatenate((idx[slice[0]][st1:(st1+dt)],idx[slice[1]][st2:(st2+dt)])))
                client_i = client(None, dataset, index_i, test_dataset, test_index_i, args)
                client_list.append(client_i)
                #start[slice[0]] = st1 + dt
                #start[slice[1]] = st2 + dt
                if len(client_list) > 100:
                    print('{} clients are created. Distribution is non-iid'.format(len(client_list)))
                    raise Exception('Assignment created')


            log('{} clients are created. Distribution is non-iid'.format(len(client_list)))
            log('all assignments')
            log(all_assignment)
            log('length')
            log(len(all_assignment))
            return client_list, np.ones(len(client_list))/len(client_list), validation_client
        # if we want to use iid partition
        delta = int(np.floor(np.sqrt(12/(how_many_clients*how_many_clients+2))*mu*args['variance_ratio']))
        if mu - delta*how_many_clients*0.5<1:
            delta = int(np.floor(2*len_ds/how_many_clients**2-2/how_many_clients))
        rd_list = list(range(len_ds))
        if not args['dataset-type'] in {'language'}:
            random.shuffle(rd_list)
        step = int(np.floor(mu))-int(delta*how_many_clients/2)#int(len(dataset)/how_many_clients)
        begin = 0
        test_begin = 0
        test_dt = int(test_dataset.__len__()/how_many_clients)
        for i in range(how_many_clients):
            index_i = rd_list[begin:(begin+step)]
            client_i = client(None, dataset, index_i, test_dataset, list(range(test_begin,test_begin+test_dt)), args)
            client_list.append(client_i)
            begin+=step
            test_begin+=test_dt
            impv.append(step/len_ds)
            step += delta
        log('Data set distributed with std'+str(np.sqrt(np.var(impv))))
        args['std'] = np.sqrt(np.var(impv))
    args['total_data'] = how_many_clients
    return client_list, impv, validation_client



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='fed-ensemble')
    parser.add_argument('--args_dir', type=str, default="cifar10.txt")
    
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
  
    inputargs = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(inputargs.seed)
    else:
        torch.manual_seed(inputargs.seed)
    random.seed(inputargs.seed)
    np.random.seed(inputargs.seed)
    torch.manual_seed(inputargs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    f = open("settings/"+inputargs.args_dir,'r')
    args = eval(f.read())
    f.close()


    jour = "logs/"+time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())+'.log'
    global LOGFILE
    LOGFILE = jour
    args['LOGFILE'] = LOGFILE

    log("training started!",logfile=LOGFILE)
    log("hyper parameters:",logfile=LOGFILE)
    log(args,logfile=LOGFILE)

    

    # prepare dataset and model
    # tested datasets are cifar10, mnist, and cifar100
    if args['DATASET'] == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                               download=True, transform=transform)
        if args['fed'] in {'fedensemble'}:
            model = []
            for i in range(args['K']):
                if args['architecture'] in {'Resnet34'}:
                    modeli = Resnet(num_blocks=[3,4,6,3])
                else:
                    modeli = Resnet()
                if args['load_model'] == 1:
                    modeli.load_state_dict(torch.load(args['modeldir']+'model'+str(i)+'.ckpt'))
                    modeli.eval()
                model.append(modeli)

        else:
            if args['architecture'] in {'Resnet34'}:
                model = Resnet(num_blocks=[3, 4, 6, 3])
            else:
                model = Resnet()
    elif args['DATASET'] == 'MNIST':
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.MNIST(root='data', train=False,
                                               download=True, transform=transform)
        cnnarg = {'num_channels': 1, 'num_classes': 10,'feature':False}
        if args['fed'] in {'fedensemble'}:
            model = []
            for i in range(args['K']):
                model.append(CNNMnist(cnnarg))
        elif args['fed'] in {'pfed'}:
            model = []
            cnnarg['feature'] = True
            for i in range(args['number_of_clients']):
                model.append(CNNMnist(cnnarg))
        else:
            model = CNNMnist(cnnarg)
      
    elif args['DATASET'] == 'CIFAR100':
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR100(root='data', train=True,
                                              download=True, transform=transform)

        testset = torchvision.datasets.CIFAR100(root='data', train=False,
                                             download=True, transform=transform)

        if args['fed'] in {'fedensemble'}:
            model = []
            for i in range(args['K']):
                if args['architecture'] in {'Resnet34'}:
                    modeli = Resnet(num_blocks=[3, 4, 6, 3],num_classes=100)
                else:
                    modeli = Resnet(num_classes=100)
                model.append(modeli)
        else:
            if args['architecture'] in {'Resnet34'}:
                model = Resnet(num_blocks=[3, 4, 6, 3], num_classes=100)
            else:
                model = Resnet(num_classes=100)

    elif args['DATASET'] == 'SHAKESPEARE':
        chars, encoded = lp.prepare_dataset_from_txt('data/shakespeare.txt')
        fc=0.1
        end_of_train_ds = int(len(encoded)*(1-fc))
        trainset = encoded[:end_of_train_ds]
        testset = encoded[end_of_train_ds:]

        n_hidden = 512
        n_layers = 2
        drop_prob = 0.5

        if args['fed'] in {'fedensemble'}:
            model = []
            for i in range(args['K']):
                model.append(CharRNN(chars, n_hidden, n_layers, drop_prob))
        else:
            model = CharRNN(chars, n_hidden, n_layers, drop_prob)

        #model = CNNMnist(cnnarg)
    
    # create client list, partition the trainset into many clients
    client_list, importance_vec, validation_client = create_client_list(model, trainset, testset, args)

    log("The dataset %s is divided into %s clients"%(args['DATASET'],len(client_list)))
    #assert False
    # train

    algorithm_class = get_algorithm_class(args['fed'])
    algorithm = algorithm_class()
    algorithm.train_model(client_list, importance_vec, validation_client, model, args, testset)
    #train(client_list, importance_vec, validation_client, model, args, testset=testset)
