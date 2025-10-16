from flgo.utils.fmodule import FModule
import torchvision

def get_model():
    return torchvision.models.get_model('resnet18', num_classes=10)


class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = get_model().to(object.device)