import os
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None
try:
    import ujson as json
except:
    import json

class TaskGenerator(FromDatasetGenerator):
    def __init__(self, classes=['bird', 'feather','headphones','ice_cream','teapot','tiger','whale', 'windmill','wine_glass','zebra']):
        self.classes = classes
        if train_data is not None: train_data.set_classes(classes)
        if val_data is not None: val_data.set_classes(classes)
        if test_data is not None: test_data.set_classes(classes)
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

class TaskPipe(FromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)
        if hasattr(self, 'feddata'):
            self.classes = self.feddata['classes']
            if train_data is not None: self.train_data.set_classes(self.classes)
            if val_data is not None: self.val_data.set_classes(self.classes)
            if test_data is not None: self.test_data.set_classes(self.classes)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'classes':generator.classes}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

TaskCalculator = GeneralCalculator