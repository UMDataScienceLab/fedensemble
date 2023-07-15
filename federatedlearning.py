import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from client import client
from network import CNNMnist, Twohiddenlayerfc, Resnet
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import time
import copy
import pickle
import random

from buffer import Memory
import modeloperations as mo
import languageprocess as lp
from utils import log

def server_update(models, odr, importance_vec, mu_list, sigma_list, old_phi, sigma, phi0, mu0, args):
    # models are data, mu_list and sigma_list are all neural nets representing centers of pd
    # phi is N by K tensor
    with torch.no_grad():
        N = len(models)
        K = len(odr)
        sm = nn.Softmax(dim=1)
        phi = torch.zeros((N, K))
        phi = phi + 1 / K
        for i in range(N):
            for k in range(K):
                phi[i][k] = mo.inner_p(mu_list[odr[k]], models[i])-0.5*mo.inner_p(mu_list[odr[k]],mu_list[odr[k]])
            #phi[i,:] = args['beta']*(phi[i,:]-phi[i,:].mean())
            #print(phi[i, :])

        phi = sm(phi)
        #print(phi)
        for k in range(K):
            mu_list[odr[k]] = mo.scalar_mul(phi[:,k]/(1/sigma**2 +phi[:,k].sum()), models)
            sigma_list[odr[k]] = 0*sigma_list[odr[k]] + 1/(1/sigma**2 +phi[:,k].sum())
        return mu_list, sigma_list, phi


def model_average(fl_model_list, models, importance_vec, phi0, args, sld=0):
    if sld == 0:
        odr = list(range(len(fl_model_list)))
    else:
        odr = copy.deepcopy(sld)
    if args['fed'] in {'fedavg'}:
        # simply take average of all weights for federated average
        worker_state_dict = [x.state_dict() for x in models]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = worker_state_dict[0]
        for key in weight_keys:
            key_sum = 0
            for i in range(len(models)):
                key_sum = key_sum + worker_state_dict[i][key]*importance_vec[i]
            fed_state_dict[key] = key_sum #/ len(models)
        #### update fed weights to fl model
        fl_model_list.load_state_dict(fed_state_dict)
        return fl_model_list
    elif args['fed'] in {'fedensemble'}:
        # update model weights by clusters
        N = len(models)
        K = len(fl_model_list)
        phi = torch.zeros((N, K))
        phi = phi+1/K # set phi to 1/K for each round

        sigma_list = torch.ones(K)

        mu0 = pickle.loads(pickle.dumps(models))

        #update
        for i in range(args['number_of_updates']):
            fl_model_list, sigma_list, phi = server_update(models, odr, importance_vec, fl_model_list, sigma_list, phi, 1e10, phi0, mu0, args)

        return fl_model_list, sigma_list, phi
   

def average_loss(loss_vec, importance_vec):
    lv = np.array(loss_vec)
    iv = np.array(importance_vec)
    return np.sum(lv*iv)



def one_round_communication_0(model, client_list, importance_vec, dw_tm1, gamma, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = copy.deepcopy(model)
    weight_vec = []
    #variance_vec = []
    loss_vec = []
    local_args = copy.deepcopy(args)
    if len(dw_tm1) >0:
        dw_tm1_sum = mo.scalar_mul(torch.ones(len(dw_tm1)), dw_tm1)
        n = len(dw_tm1)
    if args['fed'] in {'fedavg2'}:
        nocu = args['number_of_clients_used']
        selected = np.random.choice(list(range(len(client_list))), size=args['number_of_clients_used'], replace=False, p=None)
    else:
        nocu = args['number_of_clients']
    for index in range(nocu):
        if args['fed'] in {'fedavg2'}:
            #si = addargs['stratum_index'][index]
            #the_one = np.random.choice(si, size=1, replace=False, p=None)
            #client = client_list[the_one]
            client = client_list[selected[index]]
        else:
            client = client_list[index]
        if len(dw_tm1) > 0:
            if args['fed'] in {'nfed'}:
                adj_model = mo.scalar_mul([1/n*gamma, -1/n*gamma, 1], [dw_tm1_sum, dw_tm1[index], model])
            elif args['fed'] in {'nfed2'}:
                adj_model = mo.scalar_mul([1 * gamma, -1 * gamma, 1], [dw_tm1_sum, dw_tm1[index], model])
            client.get_weight([adj_model], 0)
        else:
            client.get_weight([model], 0)

        if args['random_epoch'] == 1:
            local_args['inner_epoch'] = random.randint(1,20)
        weight, loss, l2 = client.local_update(local_args, local_args['inner_epoch'], args['batch_size'], device)
        weight_vec.append(mo.scalar_mul([1,-1],[weight,model]))
        #variance_vec.append(variance)
        loss_vec.append(loss)
        #print(mo.inner_p(model, model0))
        if args['DATASET'] not in {'SINE'}:
            print('Client {} finished training with loss {}'.format(index, loss))
    #model = mo.scalar_mul(weight_vec+[model])
    #model_average(model, weight_vec, importance_vec,0, args)
    loss = average_loss(loss_vec, importance_vec)
    return weight_vec, loss


def one_round_communication_1(model_list, sigma_list, phi, client_list, importance_vec, original_loss_mat, gl_phi, i, count, args):
    # one round communication of a bayesian federated learning algorithm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_vec = []
    loss_vec = []
    l2_vec = []
    local_args = copy.deepcopy(args)
    if len(model_list)>len(client_list):
        raise Exception("Have not implemented for the case where N<K")
    K = len(model_list)
    # for some algorithms we need to sample modes uniformly randomly
    rs = list(range(K))
    random.shuffle(rs)
    # for bfed3-3 and bfed4, we only choose pre-specified mode
    if args['fed'] in {'fedensemble'}:
        mc = torch.tensor(gl_phi[:,i % args['K']]).unsqueeze(dim=1)
        #print(mc)
    
    # sld is a list representing which models are selected in each round
    sld = list(set(mc[:,0].numpy().tolist()))
    sld = [int(i) for i in sld]
    print(sld)
    if args['fed'] in {'fedensemble4'}:
        nocu = args['number_of_clients_used']
    else:
        nocu = len(client_list)
    for index in range(nocu):
        
        if args['fed'] in {'fedensemble4'}:
            # choose a mode according to pre-specified order
            mpp = int(gl_phi[index][i % args["K"]])
            the_stratum = args['stratum']
            # choose a client from stratum i by random sampling
            client_idx = int(np.random.choice(the_stratum[index], size=1, replace=False, p=None)[0])
            client = client_list[client_idx]
        elif args['fed'] in {'fedensemble'}:
            if index < K*int(args['enforce_sampling']):
                # choose a mode from pre-specified order
                mpp = int(gl_phi[index][i % args["K"]])
                client = client_list[index]
                
            else:
                mpp = int(torch.randint(0,K,(1,))[0].item())
                client = client_list[index]


        else:
            if index < K*int(args['enforce_sampling']):
                mpp = rs[index]
            else:
                mpp = int(torch.randint(0,K,(1,))[0].item())
            client = client_list[index]
        log('Client %s is training model %s'%(index, mpp) , logfile=args['LOGFILE'])
        count[mpp] += 1

        if args['fed'] in {'fedensemble'}:
            #print('the l2 norm sq of model sent in is: {}'.format(mo.inner_p(model_list[int(mpp.item())],model_list[int(mpp.item())]).item()))
            client.get_weight(model_list, mpp)
        
        if args['random_epoch'] == 1:
            local_args['inner_epoch'] = random.randint(1,20)
        # client performs update
        weight, loss, l2 = client.local_update(local_args, local_args['inner_epoch'], args['batch_size'], device)
        if args['fed'] in {'fedensemble','fedensemble4'}:
            weight_vec.append(weight)
        else:
            weight_vec = weight_vec+weight
        l2_vec.append(l2)
        loss_vec.append(loss)
     
    phi0 = pickle.loads(pickle.dumps(phi))
    model_list, sigma_list, phi = model_average(model_list, weight_vec, importance_vec, phi0, args, sld)
    # update the sampling probability by different rules
    if args['fed'] in {'fedensemble'}:
        # don't update
        l_m = None
        new_phi = torch.ones(args['number_of_clients_used'], args['K'])/args['K']
    else:
        l_m = None
        new_phi = torch.ones(args['number_of_clients'], args['K'])/args['K']
    # average loss
    loss = average_loss(loss_vec, importance_vec)
    return model_list, sigma_list, phi, loss, l_m, new_phi, sld

