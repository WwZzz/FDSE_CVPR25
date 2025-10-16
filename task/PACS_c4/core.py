import os
from .config import train_data # config必须包含train_data
try:
    from .config import test_data # config可选包含test_data和val_data
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
try:
    from .config import DataLoader as MyDataloader
except:
    MyDataloader = None
try:
    from .config import collate_fn
except:
    collate_fn = None
try:
    from .config import split_dataset
except:
    split_dataset = None
from .config import data_to_device, eval, compute_loss

import flgo.benchmark.base as fbb
import torch.utils.data
import os

class TaskGenerator(fbb.FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

class TaskPipe(fbb.FromDatasetPipe):
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            super().__init__()
            if original_dataset is None: return None
            self.data = [d for d in original_dataset]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            return self.data[item]

    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)
        self.my_split_dataset = split_dataset

    def split_dataset(self, dataset, p=0.0):
        if dataset is None: return None, None
        if self.my_split_dataset is None:
            if p == 0: return dataset, None
            s1 = int(len(dataset) * p)
            s2 = len(dataset) - s1
            if s1==0:
                return dataset, None
            elif s2==0:
                return None, dataset
            else:
                return torch.utils.data.random_split(dataset, [s2, s1])
        else:
            return self.my_split_dataset(dataset, p)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option: dict) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        if val_data is None and test_data is not None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else: server_data_test, server_data_val = test_data, val_data
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        for cid, cdata in enumerate(train_data):
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, running_time_option['local_test_ratio'])
            else:
                cdata_test = None
            task_data[f'Client{cid}'] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

class TaskCalculator(fbb.BasicTaskCalculator):
    r"""
    Support task-specific computation when optimizing models, such
    as putting data into device, computing loss, evaluating models,
    and creating the data loader
    """

    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = MyDataloader if MyDataloader is not None else torch.utils.data.DataLoader
        self.collate_fn = collate_fn

    def to_device(self, data, *args, **kwargs):
        return data_to_device(data, self.device)

    def get_dataloader(self, dataset, batch_size=64, *args, **kwargs):
        return self.DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs,)

    @torch.no_grad()
    def test(self, model, data, batch_size=64, num_workers=0, pin_memory=False, **kwargs):
        data_loader = self.get_dataloader(data, batch_size=64, num_workers=num_workers, pin_memory=pin_memory)
        return eval(model, data_loader, self.device)

    def compute_loss(self, model, data, *args, **kwargs):
        return compute_loss(model, data, self.device)