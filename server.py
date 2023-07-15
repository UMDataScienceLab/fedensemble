#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import os,time,re,tempfile,json,pickle
import math
from mip import Model, xsum, BINARY

#from replay_parser import parse_replay_syn
from torch.multiprocessing import Process,Queue
import torch.nn as nn
import torch.nn.functional as F

from utils import log
import torchvision
from torchvision import transforms

from client import *
from network import *
import modeloperations as mo
import federatedlearning as fl

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
     

class fedavg(torch.nn.Module):
    def __init__(self):
        super(fedavg, self).__init__()
        

    def train_model(self, client_list, importance_vec, validation_client, model, args, testset):
        """
            implementation of fedavg
        """
        if args['dataset-type'] in {'language'}:
            testloader = testset
        else:
            testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                         shuffle=False, num_workers=0)
   
        dw = []
        for i in range(args['outer_epoch']):
            # i is communication round
            if i%10 == 0:
                args['initial_lr'] = args['initial_lr']*0.99
            time_i = time.time()
            log('Communiaction round {}, training started'
              .format(i), logfile=args['LOGFILE'])
            impv = importance_vec
            # let clients train
            dwt, loss_i = fl.one_round_communication_0(model, client_list, impv, dw, 1, args)
                # average client trained models
            coef = torch.ones(len(dwt)+1)
            coef = coef/len(dwt)
            coef[-1] = 1
            model = mo.scalar_mul(coef, dwt+[model])                
            torch.save(model.state_dict(), 'model.ckpt')
            
            time_e = time.time()

            # calculate test accuracy and print them
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args['task'] == 'classification':
       
                res = test_acc(model, testloader, device, 0, args)

                duration = time_e - time_i
                log('Communication round {}, training finished with average training loss {:.4f}. Test acc is {:.2f}, This takes {:.2f} seconds'
                    .format(i, loss_i, res["test_acc"], duration), logfile=args['LOGFILE'])
                

            elif args['task'] == 'regression':
                tst_loss = test_loss(model, testloader, device, 0, args)
                
                duration = time_e - time_i

                log(
                    'Communication round {}, training finished with average training loss {:.4f}. Test loss is {:.4f}, This takes {:.2f} seconds'
                    .format(i, loss_i, tst_loss, duration), logfile=args['LOGFILE'])
                
            else:
                raise Exception('Undefined task: '+args['task'])

            

        #os.system(mount_command)

    def predict(self, x): 
        return self.model(x)


class fedensemble(torch.nn.Module):
    def __init__(self):
        super(fedensemble, self).__init__()
       

    def train_model(self,client_list, importance_vec, validation_client, model, args, testset):
        """
            implementation of fedensemble
        """
        ini_var = 1

        if args['dataset-type'] in {'language'}:
            testloader = testset
        else:
            testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                            shuffle=False, num_workers=0)
        # creates a logfile named by date and time
        
        # some preparations before training starts
        K = args['K']
        sigma_list = torch.ones(K)
        model_list = copy.deepcopy(model)
        # phi is related to variational inference, not sampling
        phi = torch.zeros(len(client_list), K)
        phi = phi + 1/K
        phi = phi + 0.001*torch.randn(len(client_list), K)

        # global phi is related to sampling probabilities
        global_phi = torch.zeros(len(client_list), K) + 1/K
        loss_mat = torch.ones(len(client_list), K)
        count = torch.ones(K)
        
        # create a sampling schedule
        sampling_schedule = np.zeros((args["number_of_clients"], args['K']))
        for i in range(len(sampling_schedule)):
            sampling_schedule[i] = np.random.permutation(args['K'])
        


        for i in range(args['outer_epoch']):
            # i is communication round
            if i%10 == 0:
                args['initial_lr'] = args['initial_lr']*0.99
            time_i = time.time()
            log('Communiaction round {}, training started'
                .format(i), logfile=args['LOGFILE'])
            
            if args['fed'] in {'fedensemble'}:
                # multi-model algorithms, only a part of clients participate in training
                if args['fed'] in {'fedensemble4'}:
                    impvec = np.ones(args['number_of_clients_used'])/args['number_of_clients_used']
                else:
                    impvec = importance_vec
                model_list, sigma_list, phi, loss_i, loss_mat, global_phi, sld = \
                    fl.one_round_communication_1(model_list, sigma_list, phi, client_list, impvec, loss_mat,
                                                sampling_schedule, i, count, args)
                if i%args["K"] == 0:
                    for ii in range(len(sampling_schedule)):
                        sampling_schedule[ii] = np.random.permutation(args['K'])
                for index, model in enumerate(model_list):
                    torch.save(model.state_dict(), args['model_folder']+'model'+str(index)+'.ckpt')

            time_e = time.time()

            # calculate test accuracy and print them
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args['task'] == 'classification':
                if args['fed'] in {'fedensemble'}:
                    res = test_acc(model_list, testloader, device, np.ones(len(model_list))/len(model_list), args)
                    log("model test acc highest: {}".format(torch.max(res['test_acc_mode_avg']).item()), logfile=args['LOGFILE'])
                    log("model test acc lowest: {}".format(torch.min(res['test_acc_mode_avg']).item()), logfile=args['LOGFILE'])
                    log("model test acc average: {}".format(torch.mean(res['test_acc_mode_avg']).item()), logfile=args['LOGFILE'])
                    

                duration = time_e - time_i
                log('Communication round {}, training finished with average training loss {:.4f}. Test acc is {:.2f}, This takes {:.2f} seconds'
                    .format(i, loss_i, res["test_acc"], duration), logfile=args['LOGFILE'])
                

