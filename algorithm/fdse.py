import os
import sys
import numpy as np
import copy
import flgo.algorithm.fedavg as fedavg
from collections import OrderedDict
import cvxopt
import torch
import torch.nn as nn
import flgo.utils.fmodule as fuf
import math


class DSEConv(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(DSEConv, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.Vshare = nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=True)
        self.norm_V = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.Up = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False)
        self.Vbias = nn.Parameter(torch.zeros(oup))
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        x1 = self.relu(self.norm_V(self.Vshare(x)))
        x2 = self.Up(x1)
        out = torch.cat([x1, x2], dim=1)
        # return out[:, :self.oup, :, :]
        out = out[:, :self.oup, :, :] + self.Vbias.expand(out.shape[0], self.Vbias.shape[0]).reshape(out.shape[0], self.Vbias.shape[0], 1, 1)
        return self.relu(self.bn(out))

class DSELinear(nn.Module):
    def __init__(self, inp, oup, ratio=2, relu=True):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = oup
        # self.Vshare = nn.Linear(inp, init_channels)
        self.Vshare = nn.Conv2d(inp, init_channels, 1, bias=True)
        self.norm_V = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.Up = nn.Conv2d(init_channels, new_channels, 1, groups=init_channels, bias=False)
        self.Vbias = nn.Parameter(torch.zeros(oup))
        self.bn = nn.BatchNorm1d(oup)
        # self.Up = nn.Linear(init_channels, new_channels)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x1 = self.relu(self.norm_V(self.Vshare(x)))
        x2 = self.Up(x1)
        x3 = x2.squeeze(-1).squeeze(-1)
        out = self.bn(x3 + self.Vbias)
        return self.relu(out)

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        # initialize user embedding for all clients
        self.init_algo_para({'lmbd': 10.0, 'beta':0.5, 'tau': 10.0,})
        self.model = self.model.__class__()
        shared_names = ['Vshare', 'bn', 'head', ]
        local_names = ['norm_V.running_', ]
        self.shared_weight_keys = [k for k in self.model.state_dict() if any([(s in k) for s in shared_names])]
        self.local_keys = [k for k in self.model.state_dict() if
                           any([(s in k) for s in local_names]) and k not in self.shared_weight_keys]
        self.personalized_weight_keys = [k for k in self.model.state_dict() if
                                         k not in self.shared_weight_keys and k not in self.local_keys]
        per_state = {k: v for k, v in self.model.state_dict().items() if k in self.personalized_weight_keys}
        self.client_states = [copy.deepcopy(per_state) for _ in self.clients]
        client_weights = np.array([len(c.train_data) for c in self.clients])
        self.client_weights = torch.from_numpy(client_weights / client_weights.sum()).unsqueeze(-1).to(self.device)

    def pack(self, client_id, mtype=0, *args, **kwargs):
        client_dict = {k: v for k, v in self.model.state_dict().items() if k in self.shared_weight_keys}
        client_dict.update(self.client_states[client_id])
        return {'md': copy.deepcopy(client_dict)}

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        # self.model = self.aggregate(models)
        mdicts = [m.state_dict() for m in models]
        # agg shared dict
        current_shard_dict = {k: v for k, v in self.model.state_dict().items() if k in self.shared_weight_keys}
        shared_dicts = [{k: v for k, v in md.items() if k in self.shared_weight_keys} for md in mdicts]
        new_shared_dicts = {}
        for k in self.shared_weight_keys:
            if "running_" in k or 'num_batches_tracked' in k:
                k_vecs = [md[k] for md in shared_dicts]
                if 'num_batches_tracked' in k:
                    new_shared_dicts[k] = torch.stack(k_vecs).sum(dim=0).squeeze(0)
                else:
                    # new_shared_dicts[k] = (torch.stack(k_vecs)*self.client_weights).sum(dim=0).squeeze(0)
                    new_shared_dicts[k] = torch.stack(k_vecs).mean(dim=0).squeeze(0)
            else:
                shape = shared_dicts[0][k].shape
                crt_vec_k = current_shard_dict[k].reshape(-1).to(self.device)
                k_vecs = [md[k].reshape(-1) - crt_vec_k for md in shared_dicts]
                k_norms = [t.norm() for t in k_vecs]
                k_vecs = [t / (tn + 1e-8) for t, tn in zip(k_vecs, k_norms)]
                old_stdout = sys.stdout  # backup current stdout
                sys.stdout = open(os.devnull, "w")
                op_lambda_k = self.optim_lambda(k_vecs)
                sys.stdout = old_stdout
                op_lambda_k = torch.tensor([ele[0] for ele in op_lambda_k]).float().to(self.device)
                new_vec_k = (op_lambda_k.unsqueeze(0) @ torch.stack(k_vecs))[0]
                new_shared_dicts[k] = (torch.stack(k_norms).mean() * (new_vec_k) + crt_vec_k).reshape(shape)
        # aggregate grads
        self.model.load_state_dict(new_shared_dicts, strict=False)

        # agg personalized dict
        per_dicts = [{k: v for k, v in md.items() if k in self.personalized_weight_keys} for md in mdicts]
        agg_dicts = [{} for _ in mdicts]
        for k in self.personalized_weight_keys:
            if 'num_batches_tracked' in k: continue
            per_dict_k = [md[k].reshape(-1) for md in per_dicts]
            per_dict_k = [p / p.norm() for p in per_dict_k]
            all_vecs = torch.stack(per_dict_k)
            sims_k = all_vecs @ all_vecs.T
            per_dict_k = [{k: md[k]} for md in per_dicts]
            for cid, sim_cid, agg_dict in zip(self.selected_clients, sims_k, agg_dicts):
                weight_cid = torch.nn.Softmax()(sim_cid/self.tau)
                self.client_states[cid].update({k: fuf._modeldict_weighted_average(per_dict_k, weight_cid)[k]})
        return

    def optim_lambda(self, grads):
        # create H_m*m = 2J'J where J=[grad_i]_n*m
        n = len(grads)
        Jt = []
        for gi in grads:
            Jt.append((copy.deepcopy(gi).cpu()).numpy())
        Jt = np.array(Jt)
        # target function
        P = 2 * np.dot(Jt, Jt.T)

        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([0. for i in range(n)])
        ub = np.array([1. for i in range(n)])
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        res = self.quadprog(P, q, G, h, A, b)
        return res

    def quadprog(self, P, q, G, h, A, b):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.model = copy.deepcopy(self.server.model)
        self.bnlayers = list(dict.fromkeys(
            ["".join([f'[{m}]' if m.isdigit() else f'.{m}' for m in k.split('.')[:-1]]) for k in
             self.model.state_dict().keys() if 'bn' in k]))

    def unpack(self, received_pkg):
        self.model.load_state_dict(received_pkg['md'], strict=False)
        return self.model

    def pack(self, model, *args, **kwargs):
        return {
            "model": copy.deepcopy(model.to(self.server.device)),
        }
    #
    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        eps = model.head.weight.shape[0]
        layers = []
        for ln in self.bnlayers:
            name = 'model' + ln
            l = eval(name)
            layers.append(l)
        # layers = [eval('model'+ln) for ln in self.bnlayers]
        global_means = copy.deepcopy([l.running_mean.to(self.device).detach() for l in layers])
        global_vars = copy.deepcopy([l.running_var.to(self.device).detach() for l in layers])
        weights = np.exp(np.array([self.beta*i for i in range(len(layers))]))
        weights/=weights.sum()
        feature_maps = []

        def hook(model, input, output):
            feature_maps.append(
                input[0].mean(dim=0).mean(dim=-1).mean(dim=-1) if len(input[0].shape) > 3 else input[0].mean(dim=0))

        lhooks = []
        for l in layers:
            lh = l.register_forward_hook(hook)
            lhooks.append(lh)
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss_reg = 0.
            if self.server.current_round > 1:
                loss_mean = 0.
                loss_var = 0.
                for g, w, f, v, ln in zip(global_means, weights, feature_maps, global_vars, layers):
                    mf = f.mean(dim=0)
                    vf = f.var(dim=0)
                    fn = (1. - ln.momentum) * ln.running_mean + ln.momentum * mf
                    vn = (1.- ln.momentum) * ln.running_var + ln.momentum * vf
                    loss_mean += w*((g - fn).pow(2)).mean()
                    loss_var += w*((v.mean() - vn.mean()).pow(2))
                loss_reg += (loss_mean + loss_var)
                loss += self.lmbd*loss_reg
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            feature_maps.clear()
        for lh in lhooks:
            lh.remove()
        return


class AlexNetEncoder(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self):
        super(AlexNetEncoder, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', DSEConv(3, 64, kernel_size=11, stride=4)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', DSEConv(64, 192, kernel_size=5)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', DSEConv(192, 384, kernel_size=3)),
                ('conv4', DSEConv(384, 256, kernel_size=3)),
                ('conv5', DSEConv(256, 256, kernel_size=3)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = DSELinear(256 * 6 * 6, 1024)
        self.fc2 = DSELinear(1024, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DomainnetModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.head = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class PACSModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.head = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class OfficeModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.head = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


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
