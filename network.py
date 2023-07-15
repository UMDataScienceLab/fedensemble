import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import random
#from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
#                               MetaBatchNorm2d, MetaLinear)

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args['num_channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args['num_classes'])
        self.feature = args['feature']

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = F.log_softmax(self.fc2(x), dim=1)
        if self.feature:
            return out, x
        else:
            return out

class Twohiddenlayerfc(nn.Module):
    def __init__(self):
        super(Twohiddenlayerfc, self).__init__()
        self.fc1 = nn.Linear(1,1024)
        self.fc2 = nn.Linear(1024,1024)
        #self.fc21 = nn.Linear(100, 100)
        #self.fc22 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(1024,1)

    def forward(self, x):
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))#+x
        #x = F.relu(self.fc21(x))+x
        #x = F.relu(self.fc22(x))+x
        x = self.fc3(x)
        return x

class Encodinglayer(nn.Module):
    def __init__(self, r_dim=10):
        super(Encodinglayer, self).__init__()
        self.fc1 = nn.Linear(2,400)
        self.fc2 = nn.Linear(400,400)
        #self.fc21 = nn.Linear(100, 100)
        #self.fc22 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(400,r_dim)

    def forward(self, x, y):
        #print(torch.cat((x,y),dim=1).size())
        x = F.relu(self.fc1(torch.cat((x,y), dim=1)))
        x = F.relu(self.fc2(x))+x
        #x = F.relu(self.fc21(x))+x
        #x = F.relu(self.fc22(x))+x
        x = self.fc3(x)
        return x#F.softmax(x)

class R2Z(nn.Module):
    def __init__(self, r_dim=10, zdim=10):
        super(ER2Z, self).__init__()
        self.fc1 = nn.Linear(r,400)
        self.fc2 = nn.Linear(400,400)
        #self.fc21 = nn.Linear(100, 100)
        #self.fc22 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(400,z_dim)

    def forward(self, r):
        #print(torch.cat((x,y),dim=1).size())
        x = F.relu(self.fc1(r))
        x = F.relu(self.fc2(x))+x
        #x = F.relu(self.fc21(x))+x
        #x = F.relu(self.fc22(x))+x
        x = self.fc3(x)
        return x

class Decodinglayer(nn.Module):
    def __init__(self, r_dim=10):
        super(Decodinglayer, self).__init__()
        self.rzfc1 = nn.Linear(r_dim, 400)
        self.rzfc2 = nn.Linear(400, r_dim)
        self.fc1 = nn.Linear(r_dim+1,128)
        self.fc2 = nn.Linear(128,128)
        #self.fc21 = nn.Linear(100, 100)
        #self.fc22 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(128,1)

    def forward(self, x, r_vec):
        r_mat = r_vec.unsqueeze(0)
        r_mat = F.relu(self.rzfc1(r_mat))
        r_mat = F.relu(self.rzfc2(r_mat))
        r_stack = torch.cat([r_mat for i in range(len(x))])
        input = torch.cat((x,r_stack),dim=1)
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))+x
        #x = F.relu(self.fc21(x))+x
        #x = F.relu(self.fc22(x))+x
        x = self.fc3(x)
        return x
'''
class Twohiddenlayerfcmeta(MetaModule):
    def __init__(self):
        super(Twohiddenlayerfcmeta, self).__init__()
        self.fc1 = MetaLinear(1,512)
        self.fc2 = MetaLinear(512,512)
        #self.fc21 = nn.Linear(100, 100)
        #self.fc22 = nn.Linear(100, 100)
        self.fc3 = MetaLinear(512,1)

    def forward(self, x, params=None):
        #print(x.size())
        x = F.relu(self.fc1(x, params=self.get_subdict(params, 'fc1')))
        x = F.relu(self.fc2(x, params=self.get_subdict(params, 'fc2')))#+x
        #x = F.relu(self.fc21(x))+x
        #x = F.relu(self.fc22(x))+x
        out = self.fc3(x)
        return out#, x
'''
class Twohiddenlayerfcwithfeature(nn.Module):
    def __init__(self):
        super(Twohiddenlayerfcwithfeature, self).__init__()
        self.fc1 = nn.Linear(1,512)
        self.fc2 = nn.Linear(512,512)
        #self.fc21 = nn.Linear(100, 100)
        #self.fc22 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(512,1)

    def forward(self, x):
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))#+x
        #x = F.relu(self.fc21(x))+x
        #x = F.relu(self.fc22(x))+x
        out = self.fc3(x)
        return out, x


class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5,  train_on_gpu=True):
        super().__init__()
        self.train_on_gpu = train_on_gpu
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden


        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        # get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        # pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)

        # put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10, input_chan=3):
        super(Resnet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_chan, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x.float())))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = F.log_softmax(out, dim=1)
        return out

class Linearnet(nn.Module):
    def __init__(self, num_features):
        super(Linearnet, self).__init__()
        self.fc1 = nn.Linear(num_features,1)

    def forward(self, x):
        return self.fc1(x.float())


class P1Net(nn.Module):
    def __init__(self):
        super(P1Net, self).__init__()
        self.conv11 = nn.Conv2d(1, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.conv31 = nn.Conv2d(128, 256, 3)
        self.conv32 = nn.Conv2d(256, 256, 3)
        self.conv33 = nn.Conv2d(256, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def convs(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.max_pool2d(x, (2, 2))
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 8 * 8)
        x1 = self.fc1(x1)
        x1 = self.sigmoid(self.fc2(x1))
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 8 * 8)
        x2 = self.fc1(x2)
        x2 = self.sigmoid(self.fc2(x2))
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

