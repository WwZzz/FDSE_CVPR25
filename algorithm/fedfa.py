import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf

class FFALayer(nn.Module):
    def __init__(self, prob=0.5, eps=1e-6, momentum1=0.99, momentum2=0.99, nfeat=None):
        super(FFALayer, self).__init__()
        self.prob = prob
        self.eps = eps
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.nfeat = nfeat

        self.register_buffer('running_var_mean_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_var_std_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_mean_bmic', torch.zeros(self.nfeat))
        self.register_buffer('running_std_bmic', torch.ones(self.nfeat))

    def forward(self, x):
        if not self.training: return x
        if np.random.random() > self.prob: return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)
        std = std.sqrt()

        self.momentum_updating_running_mean_and_std(mean, std)

        var_mu = self.var(mean)
        var_std = self.var(std)

        running_var_mean_bmic = 1 / (1 + 1 / (self.running_var_mean_bmic + self.eps))
        gamma_mu = x.shape[1] * running_var_mean_bmic / sum(running_var_mean_bmic)

        running_var_std_bmic = 1 / (1 + 1 / (self.running_var_std_bmic + self.eps))
        gamma_std = x.shape[1] * running_var_std_bmic / sum(running_var_std_bmic)

        var_mu = (gamma_mu + 1) * var_mu
        var_std = (gamma_std + 1) * var_std

        var_mu = var_mu.sqrt().repeat(x.shape[0], 1)
        var_std = var_std.sqrt().repeat(x.shape[0], 1)

        beta = self.gaussian_sampling(mean, var_mu)
        gamma = self.gaussian_sampling(std, var_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    def gaussian_sampling(self, mu, std):
        e = torch.randn_like(std)
        z = e.mul(std).add_(mu)
        return z

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean_bmic = self.running_mean_bmic * self.momentum1 + \
                                     mean.mean(dim=0, keepdim=False) * (1 - self.momentum1)
            self.running_std_bmic = self.running_std_bmic * self.momentum1 + \
                                    std.mean(dim=0, keepdim=False) * (1 - self.momentum1)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean_bmic = self.running_var_mean_bmic * self.momentum2 + var_mean * (1 - self.momentum2)
            self.running_var_std_bmic = self.running_var_std_bmic * self.momentum2 + var_std * (1 - self.momentum2)

class Server(fedavg.Server):
    def aggregate(self, models: list, *args, **kwargs):
        mds = [m.state_dict() for m in models]
        md = fuf._modeldict_weighted_average(mds)
        for key in self.model.state_dict().keys():
            if 'running_var_mean_bmic' in key or 'running_var_std_bmic' in key:
                tmp = []
                for mdi in mds:
                    tmp.append(mdi[key.replace('running_var_', 'running_')])
                tmp = torch.stack(tmp)
                var = torch.var(tmp)
                md[key].data.copy_(var)
        self.model.load_state_dict(md)
        return self.model

Client = fedavg.Client


class AlexNetEncoder(nn.Module):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.layer1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True))
            ])
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True))
            ])
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer3 = nn.Sequential(
            OrderedDict([
                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True))
            ])
        )
        self.layer4 = nn.Sequential(
            OrderedDict([
                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True))
            ])
        )
        self.layer5 = nn.Sequential(
            OrderedDict([
                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True))
            ])
        )
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fcs = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 1024)),
                ('bn6', nn.BatchNorm1d(1024)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(1024, 1024)),
                ('bn7', nn.BatchNorm1d(1024)),
                ('relu7', nn.ReLU(inplace=True)),

                # ('fc3', nn.Linear(1024, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = self.fcs(f)

        return x

class DomainnetModel(fuf.FModule):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self):
        super().__init__()
        self.model = AlexNetEncoder()
        self.head = nn.Linear(1024, 10)
        self.ffa_layer1, self.ffa_layer2, self.ffa_layer3, self.ffa_layer4, self.ffa_layer5 = self.make_layers()

    def make_layers(self):
        layer1 = FFALayer(nfeat=64)
        layer2 = FFALayer(nfeat=192)
        layer3 = FFALayer(nfeat=384)
        layer4 = FFALayer(nfeat=256)
        layer5 = FFALayer(nfeat=256)
        return layer1, layer2, layer3, layer4, layer5

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def encoder(self, x):
        x = self.model.layer1(x)
        x = self.model.pool1(x)
        x = self.ffa_layer1(x)

        x = self.model.layer2(x)
        x = self.model.pool2(x)
        x = self.ffa_layer2(x)

        x = self.model.layer3(x)
        x = self.ffa_layer3(x)

        x = self.model.layer4(x)
        x = self.ffa_layer4(x)

        x = self.model.layer5(x)
        x = self.model.pool5(x)
        x = self.ffa_layer5(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fcs(x)
        return x

class PACSModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.model = AlexNetEncoder()
        self.head = nn.Linear(1024, 7)
        self.ffa_layer1, self.ffa_layer2, self.ffa_layer3, self.ffa_layer4, self.ffa_layer5 = self.make_layers()

    def make_layers(self):
        layer1 = FFALayer(nfeat=64)
        layer2 = FFALayer(nfeat=192)
        layer3 = FFALayer(nfeat=384)
        layer4 = FFALayer(nfeat=256)
        layer5 = FFALayer(nfeat=256)
        return layer1, layer2, layer3, layer4, layer5

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def encoder(self, x):
        x = self.model.layer1(x)
        x = self.model.pool1(x)
        x = self.ffa_layer1(x)

        x = self.model.layer2(x)
        x = self.model.pool2(x)
        x = self.ffa_layer2(x)

        x = self.model.layer3(x)
        x = self.ffa_layer3(x)

        x = self.model.layer4(x)
        x = self.ffa_layer4(x)

        x = self.model.layer5(x)
        x = self.model.pool5(x)
        x = self.ffa_layer5(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fcs(x)
        return x

class OfficeModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.model = AlexNetEncoder()
        self.head = nn.Linear(1024, 10)
        self.ffa_layer1, self.ffa_layer2, self.ffa_layer3, self.ffa_layer4, self.ffa_layer5 = self.make_layers()

    def make_layers(self):
        layer1 = FFALayer(nfeat=64)
        layer2 = FFALayer(nfeat=192)
        layer3 = FFALayer(nfeat=384)
        layer4 = FFALayer(nfeat=256)
        layer5 = FFALayer(nfeat=256)
        return layer1, layer2, layer3, layer4, layer5

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def encoder(self, x):
        x = self.model.layer1(x)
        x = self.model.pool1(x)
        x = self.ffa_layer1(x)

        x = self.model.layer2(x)
        x = self.model.pool2(x)
        x = self.ffa_layer2(x)

        x = self.model.layer3(x)
        x = self.ffa_layer3(x)

        x = self.model.layer4(x)
        x = self.ffa_layer4(x)

        x = self.model.layer5(x)
        x = self.model.pool5(x)
        x = self.ffa_layer5(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fcs(x)
        return x

model_map = {
    'PACS': PACSModel,
    'domainnet': DomainnetModel,
    'office': OfficeModel,
}

data_map = {

}

def init_dataset(object):
    pass

def init_local_module(object):
    pass

def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map[dataset]().to(object.device)