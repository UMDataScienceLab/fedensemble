import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import modeloperations as mo
import pickle

def test_acc(model, testloader, device, importance_vec, args):
    if args['dataset-type'] in {'language'}:
        if args['fed'] in {'fedavg'}:
            net = copy.deepcopy(model).to(device)
            val_h = net.init_hidden(args['batch_size'])
            val_losses = []
            #net.eval()
            tot = 0
            tot_correct = 0
            for x, y in lp.get_batches(testloader, args['batch_size'], args['seq_length']):
                # One-hot encode our data and make them Torch tensors
                x = lp.one_hot_encode(x, len(net.chars))
                x, y = torch.from_numpy(x), torch.from_numpy(y)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                inputs, targets = x, y
                if True:
                    inputs, targets = inputs.cuda(), targets.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = nn.CrossEntropyLoss()(output, targets.view(args['batch_size'] * args['seq_length']).long())

                useless, maxed = torch.max(output, dim=1)
                tot += len(maxed)
                # maxed = maxed.reshape(targets.shape)
                tot_correct += (maxed == targets.view(args['batch_size'] * args['seq_length'])).sum().item()

                val_losses.append(val_loss.item())
            return {"test_acc":tot_correct/tot * 100, "test_loss":np.mean(val_losses)}
        else:
            models = copy.deepcopy(model)
            models = [i.to(device) for i in models]
            val_h = [modeli.init_hidden(args['batch_size']) for modeli in models]
            print("length of models : {}".format(len(models)))
            print("length of val_h : {}".format(len(val_h)))
            val_losses = []
            indiv_loss_vec = []
            tot = 0
            tot_correct = 0
            # net.eval()
            for x, y in lp.get_batches(testloader, args['batch_size'], args['seq_length']):
                x = lp.one_hot_encode(x, len(models[0].chars))
                x, y = torch.from_numpy(x), torch.from_numpy(y)
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h[0] = tuple([each.data for each in val_h[0]])
                inputs, targets = x, y
                if True:
                    inputs, targets = inputs.cuda(), targets.cuda()
                output, val_h[0] = models[0](inputs, val_h[0])
                val_loss0 = nn.CrossEntropyLoss()(output, targets.view(args['batch_size'] * args['seq_length']).long())

                indiv_loss_vec.append(val_loss0.item())
                for ii in range(1,args['K']):
                    #print("ii={}".format(ii))
                    val_h[ii] = tuple([each.data for each in val_h[ii]])
                    outputii, val_h[ii] = models[ii](inputs, val_h[ii])
                    output = output + outputii
                    val_lossii = nn.CrossEntropyLoss()(outputii, targets.view(args['batch_size'] * args['seq_length']).long())

                    indiv_loss_vec.append(val_lossii.item())
                output = output/args['K']
                val_loss = nn.CrossEntropyLoss()(output, targets.view(args['batch_size'] * args['seq_length']).long())
                useless, maxed = torch.max(output, dim=1)
                tot += len(maxed)
                # maxed = maxed.reshape(targets.shape)
                tot_correct += (maxed == targets.view(args['batch_size'] * args['seq_length'])).sum().item()

                val_losses.append(val_loss.item())
            return {"test_loss":np.mean(val_losses), "test_loss_mode_avg":np.array(indiv_loss_vec), "test_acc":tot_correct/tot*100}

    if args['fed'] in {'fedavg'}:
        print('calculating test acc')
        correct = 0.
        total = 0.
        net = copy.deepcopy(model).to(device)
        #net.eval()
        with torch.no_grad():
            loss = 0
            ct = 0
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                if args['fed'] in {'pfed'}:
                    outputs, _ = net(images)
                else:
                    outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                loss += nn.CrossEntropyLoss()(outputs, labels)
                ct += 1
                correct += (predicted == labels).sum().item()
            try:
                correct / total
                loss / ct
            except:
                print(total, ct, correct, loss, testloader.__len__())
        return {"test_acc": 100 * correct / total, "test_loss" : loss/ct}
    elif args['fed'] in {'fedensemble'}:
        correct = 0
        total = 0
        correct_vec = torch.zeros(len(model))
        total_vec  = torch.zeros(len(model))
        entp_vec = torch.zeros(len(model)).to(device)
        entp_ensemble = 0
        with torch.no_grad():
            std_dev = mo.model_std(model)
            star_dis, model_belong_dict = mo.model_star(model)
            #print(star_dis)
            star_dis = torch.tensor(star_dis)
            l2normlist = torch.tensor([torch.sqrt(mo.inner_p(mi,mi)) for mi in model])
            loss = 0
            ct = 0
            for data in testloader:
                input, routp = data[0].to(device), data[1].to(device)
                model0 = pickle.loads(pickle.dumps(model[0])).to(device)
                outputs1 = F.softmax(model0.to(device)(input))
                outputs = outputs1*importance_vec[0]
                _, predicted = torch.max(outputs.data, 1)
                entp_vec[0] += -(outputs1*torch.log(outputs1+1e-8)).sum()
                correct_vec[0] += (predicted == routp).sum().item()
                total_vec[0] += routp.size(0)
                for i in range(1, len(model)):
                    modeli = pickle.loads(pickle.dumps(model[i])).to(device)
                    oi = F.softmax(modeli(input))
                    _, predicted = torch.max(oi.data, 1)
                    correct_vec[i] += (predicted == routp).sum().item()
                    total_vec[i] += routp.size(0)
                    entp_vec[i] += -(oi * torch.log(oi + 1e-8)).sum()

                    outputs = outputs + oi*importance_vec[i]
                #print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == routp).sum().item()
                total += routp.size(0)
                loss += nn.CrossEntropyLoss()(outputs, routp)
                # entp_ensemble += torch.sum(entp_vec / routp.size(0))
                entp_ensemble += -(outputs * torch.log(outputs + 1e-8)).sum() #/ routp.size(0)
                ct += 1
        return {"test_acc":100*correct / total,
                "test_acc_mode_avg": 100 * correct_vec/total_vec,
                "test_loss": loss/ct,
                "distance_max":torch.max(star_dis),
                "distance_min":torch.min(star_dis),
                "distance_avg":torch.mean(star_dis),
                "l2_norm_avg":torch.mean(l2normlist),
                "average_individual_entropy":torch.mean(entp_vec)/total,
                "ensemble_prediction_entropy":entp_ensemble/total}


def test_loss(model, testloader, device, importance_vec, args):
    if args['fed'] in {'fedavg'}:
        net = pickle.loads(pickle.dumps(model)).to(device)
        loss = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                input, routp = data[0].to(device), data[1].to(device)
                #print(input.size(), routp.size())
                outputs = net(input)
                #print(input.size(), routp.size(), outputs.size())
                if args['DATASET']=='SINE':
                    loss += (routp-outputs).norm(2)
                else:
                    loss += (routp-outputs).norm(2)
                total += 1
        return loss / total
    elif args['fed'] in {'fedensemble'}:
        loss = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                input, routp = data[0].to(device), data[1].to(device)
                model0 = pickle.loads(pickle.dumps(model[0])).to(device)
                outputs = model0.to(device)(input)*importance_vec[0]
                for i in range(1, len(model)):
                    modeli = pickle.loads(pickle.dumps(model[i])).to(device)
                    outputs = outputs + modeli(input)*importance_vec[i]
                loss += (routp - outputs).norm(2)
                total += 1
        return loss / total
