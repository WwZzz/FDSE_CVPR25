from collections import OrderedDict
import flgo.algorithm.fedavg as fedavg
import torch.nn as nn
import flgo.utils.fmodule as fmodule
import torch
import torch.nn.functional as F
import torch.distributions as distributions
import os

class CMIModule(nn.Module):
    def __init__(self, dim, num_classes):
        super(CMIModule, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.r_mu = nn.Parameter(torch.zeros(self.num_classes, self.dim))
        self.r_sigma = nn.Parameter(torch.ones(self.num_classes, self.dim))
        self.C = nn.Parameter(torch.ones([]))

    def forward(self, z_mu, z_sigma, y):
        r_sigma_softplus = F.softplus(self.r_sigma)
        r_mu = self.r_mu[y]
        r_sigma = r_sigma_softplus[y]
        z_mu_scaled = z_mu * self.C
        z_sigma_scaled = z_sigma * self.C
        regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
        loss_CMI = regCMI.sum(1).mean()
        return loss_CMI

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'beta1':0.01, 'beta2':0.0005, 'num_samples':20})
        self.model.set_num_samples(self.num_samples)

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = self.aggregate(models)
        self.model.set_num_samples(self.num_samples)

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.cmi = CMIModule(self.server.model.zdim, self.server.model.num_classes).to(self.device)

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        optimizer.add_param_group({'params':self.cmi.parameters(), 'lr': self.learning_rate, 'momentum': 0.9})
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            if model.zstat is not None:
                z, z_mu, z_sigma = model.zstat
                loss_L2R = z.norm(dim=1).mean()
                loss_CMI = self.cmi(z_mu, z_sigma, batch_data[-1])
                loss = loss + self.beta1*loss_L2R + self.beta2*loss_CMI
                model.clear_zstat()
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return


class OfficeModel(fmodule.FModule):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10, zdim=512):
        super().__init__()
        self.num_samples = 1
        self.zdim = zdim
        self.num_classes = num_classes
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2*zdim)
        self.head = nn.Linear(zdim, num_classes)
        self.zstat = None

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        if self.training:
            z, z_mu, z_sigma = self.resample_z(x)
            self.zstat = (z, z_mu, z_sigma)
            out = self.head(z)
            return out
        else:
            z,_,_ = self.resample_z(x, num_samples=self.num_samples)
            preds = torch.softmax(self.head(z), dim=1)
            preds = preds.view([self.num_samples, -1, self.num_classes]).mean(0)
            return torch.log(preds)

    def get_zstat(self):
        return self.zstat

    def clear_zstat(self):
        self.zstat = None

    def resample_z(self, x, num_samples=1):
        z_mu = x[:, :self.zdim]
        z_sigma = F.softplus(x[:, self.zdim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.zdim])
        return z, z_mu, z_sigma

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

class DomainnetModel(fmodule.FModule):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10, zdim=512):
        super(DomainnetModel, self).__init__()
        self.num_samples = 1
        self.zdim = zdim
        self.num_classes = num_classes
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024,1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2*zdim)
        self.head = nn.Linear(zdim, num_classes)
        self.zstat = None

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        if self.training:
            z, z_mu, z_sigma = self.resample_z(x)
            self.zstat = (z, z_mu, z_sigma)
            out = self.head(z)
            return out
        else:
            z, _, _ = self.resample_z(x, num_samples=self.num_samples)
            preds = torch.softmax(self.head(z), dim=1)
            preds = preds.view([self.num_samples, -1, self.num_classes]).mean(0)
            return torch.log(preds)

    def get_zstat(self):
        return self.zstat

    def clear_zstat(self):
        self.zstat = None

    def resample_z(self, x, num_samples=1):
        z_mu = x[:, :self.zdim]
        z_sigma = F.softplus(x[:, self.zdim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.zdim])
        return z, z_mu, z_sigma

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

class PACSModel(fmodule.FModule):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=7, zdim=512):
        super(PACSModel, self).__init__()
        self.num_samples = 1
        self.zdim = zdim
        self.num_classes = num_classes
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2*zdim)
        self.head = nn.Linear(zdim, num_classes)
        self.zstat = None

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        if self.training:
            z, z_mu, z_sigma = self.resample_z(x)
            self.zstat = (z, z_mu, z_sigma)
            out = self.head(z)
            return out
        else:
            z,_,_ = self.resample_z(x, num_samples=self.num_samples)
            preds = torch.softmax(self.head(z), dim=1)
            preds = preds.view([self.num_samples, -1, self.num_classes]).mean(0)
            return torch.log(preds)


    def get_zstat(self):
        return self.zstat


    def clear_zstat(self):
        self.zstat = None


    def resample_z(self, x, num_samples=1):
        z_mu = x[:, :self.zdim]
        z_sigma = F.softplus(x[:, self.zdim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.zdim])
        return z, z_mu, z_sigma

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples


model_map = {
    'PACS': PACSModel,
    'office': OfficeModel,
    'domainnet': DomainnetModel,
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
