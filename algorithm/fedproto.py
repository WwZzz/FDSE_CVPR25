"""
This is a non-official implementation of personalized FL method FedProto (https://ojs.aaai.org/index.php/AAAI/article/view/20819).
The original implementation is at https://github.com/yuetan031/FedProto
"""
import collections
import copy
import torch
import torch.utils.data.dataset
import torch.nn as nn
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule
import numpy as np
from itertools import chain


def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'lmbd':0.1})
        lb_counters = [c.count_label() for c in self.clients]
        self.num_classes = len(set(list(chain.from_iterable([list(lbc.keys()) for lbc in lb_counters]))))
        for c, lbc in zip(self.clients, lb_counters):
            c.num_classes, c.sizes_label, c.probs_label = c.set_label_dist(self.num_classes, lbc)
        self.sample_option = 'full'
        self.output_layer = "".join([f'[{m}]' if m.isdigit() else f'.{m}' for m in list(self.model.state_dict().keys())[-1].split('.')[:-1]])
        self.dim = eval('self.model'+self.output_layer).weight.shape[1]
        self.c = {}

    def pack(self, client_id, mtype=0):
        return {'c': copy.deepcopy(self.c),}

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        cs, sizes_label = res['c'], res['sizes_label']
        self.c = self.aggregate(cs, sizes_label)
        return

    def aggregate(self, cs:list, sizes_label:list):
        if len(cs)==0: return self.c
        num_samples = np.sum(sizes_label, axis=0)
        num_j_clients = np.zeros(self.num_classes)
        new_c = {j:torch.zeros((1, self.dim)) for j in range(self.num_classes)}
        for j in range(self.num_classes):
            for ci, i, si in zip(cs, self.received_clients, sizes_label):
                if si[j]==0:continue
                num_j_clients += 1
                new_c[j] += (ci[j]*si[j]/num_samples[j])
            # new_c[j]/=num_j_clients
        return new_c

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.output_layer = self.server.output_layer
        self.loss_mse = nn.MSELoss()

    def count_label(self):
        data_loader = self.calculator.get_dataloader(self.train_data, self.batch_size)
        labels = torch.cat([batch[-1] for _, batch in enumerate(data_loader)], dim=0).tolist()
        label_counter = collections.Counter(labels)
        return label_counter

    def set_label_dist(self, num_classes, label_counter):
        sizes_label = np.zeros(num_classes)
        for lb in range(num_classes):
            if lb in label_counter.keys():
                sizes_label[lb] = label_counter[lb]
        probs_label = sizes_label/sizes_label.sum()
        return num_classes, sizes_label, probs_label

    def reply(self, svr_pkg):
        cg = self.unpack(svr_pkg)
        self.train(self.model, cg)
        return self.pack()

    def unpack(self, svr_pkg):
        return svr_pkg['c']

    def pack(self):
        c = {}
        feature_maps = []
        def hook(model, input, output):
            feature_maps.append(input)
        hook_handler = eval("self.model{}".format(self.output_layer)).register_forward_hook(hook)
        self.model.to(self.device)
        with torch.no_grad():
            dataloader = self.calculator.get_dataloader(self.train_data, self.batch_size)
            for batch_id, batch_data in enumerate(dataloader):
                batch_data = self.calculator.to_device(batch_data)
                _ = self.model(batch_data[0])
                protos = feature_maps.pop()[0].detach()
                labels = batch_data[-1]
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi not in c.keys():
                        c[yi] = protos[i]
                    else:
                        c[yi] += protos[i]
            for j in range(len(self.sizes_label)):
                if self.sizes_label[j]==0: continue
                c[j]/=self.sizes_label[j]
                c[j] = c[j].to('cpu')
        hook_handler.remove()
        return {'c': c, 'sizes_label': self.sizes_label}

    @fmodule.with_multi_gpus
    def train(self, model, cg):
        feature_maps = []
        def hook(model, input, output):
            feature_maps.append(input)
        hook_handler = eval("self.model{}".format(self.output_layer)).register_forward_hook(hook)
        optimizer = self.calculator.get_optimizer(self.model, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        for iter in range(self.num_steps):
            model.zero_grad()
            batch_data = self.calculator.to_device(self.get_batch_data())
            labels = batch_data[-1]
            model.zero_grad()
            loss_erm = self.calculator.compute_loss(model, batch_data)['loss']
            protos = feature_maps.pop()[0]
            protos_new = copy.deepcopy(protos.data)
            for i,label in enumerate(labels):
                if label.item() in cg.keys():
                    protos_new[i, :] = cg[label.item()][0].data
            loss_reg = self.loss_mse(protos_new, protos)
            loss = loss_erm + self.lmbd*loss_reg
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        hook_handler.remove()
        return