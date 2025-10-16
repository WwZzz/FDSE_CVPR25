import torch
import os
import torch.nn as nn
from torchvision import models
import flgo.utils.fmodule as fmodule

class DomainnetModel(fmodule.FModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet = model

    def forward(self, x):
        x = self.resnet(x)
        return x

class PACSModel(fmodule.FModule):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet = model

    def forward(self, x):
        x = self.resnet(x)
        return x

class OfficeModel(fmodule.FModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet = model

    def forward(self, x):
        x = self.resnet(x)
        return x

model_map = {
    'PACS': PACSModel,
    'office': OfficeModel,
    'domainnet': DomainnetModel,
}


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map[dataset]().to(object.device)