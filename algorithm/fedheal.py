"""This is non-official implementation of FedHEAL (Fair Federated Learning under Domain Skew with  Local Consistency and Domain Diversity) from CVPR2024.
The official implementation lies in https://github.com/yuhangchen0/FedHEAL
"""

import copy
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
import torch

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'tau': 0.3, 'beta':0.4})
        self.increase_history = [None for _ in self.clients]
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.previous_weights = [1.0/self.clients_per_round for _ in self.clients]
        self.previous_delta_weights = [0 for _ in self.clients]
        self.client_norms = [0 for _ in self.clients]

    def consistency_mask(self, client_id, update_diff):
        updates = update_diff
        if self.increase_history[client_id] is None:
            self.increase_history[client_id] = {key: torch.zeros_like(val) for key, val in updates.items()}
            for key in updates: self.increase_history[client_id][key] = (updates[key] >= 0).float()
            return {key: torch.ones_like(val) for key, val in updates.items()}
        mask = {}
        for key in updates:
            positive_consistency = self.increase_history[client_id][key]
            negative_consistency = 1 - self.increase_history[client_id][key]
            consistency = torch.where(updates[key] >= 0, positive_consistency, negative_consistency)
            mask[key] = (consistency > self.tau).float()
        for key in updates:
            increase = (updates[key] >= 0).float()
            self.increase_history[client_id][key] = (self.increase_history[client_id][key] * self.current_round + increase) / (self.current_round + 1)
        return mask

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        masks, client_updates = [],[]
        for cid, mi in zip(self.selected_clients, models):
            ui = fuf._modeldict_sub(mi.state_dict(), self.model.state_dict())
            mask_i = self.consistency_mask(cid, ui)
            masked_ui = fuf._modeldict_multiply(mask_i, ui)
            self.client_norms[cid] = sum([torch.norm(masked_ui[key]).item() for key in masked_ui if key in self.param_names])
            masks.append(mask_i)
            client_updates.append(masked_ui)
        weights = self.get_params_diff_weights(self.selected_clients)
        self.gv.logger.info("Weights: {}".format(weights))
        self.aggregate(weights, masks, client_updates)

    def get_params_diff_weights(self, selected_clients):
        weight_dict = []
        for i,client in enumerate(selected_clients):
            client_distance = self.client_norms[client]
            delta_weight = (1 - self.beta) * (self.previous_delta_weights[client]) + self.beta * ((client_distance) / (sum(self.client_norms)+1e-8))
            new_weight = self.previous_weights[client] + delta_weight
            weight_dict.append(new_weight)
            self.previous_weights[client] = new_weight
            self.previous_delta_weights[client] = delta_weight
        weight_dict = np.array(weight_dict)
        weight_dict = weight_dict/weight_dict.sum()
        return weight_dict

    def aggregate(self, freq=None, masks=[], client_update=[]):
        selected_clients = self.selected_clients
        global_w = self.model.state_dict()
        global_params_new = copy.deepcopy(global_w)
        for param_key in global_params_new:
            adjusted_weights_list = []
            for i, client_id in enumerate(selected_clients):
                weight_for_client = freq[i] * masks[i][param_key]
                adjusted_weights_list.append(weight_for_client)
            adjusted_weights = torch.stack(adjusted_weights_list).to(self.device)
            total_weight_per_param = torch.sum(adjusted_weights, dim=0).unsqueeze(0).to(self.device)
            original_weights = torch.Tensor([freq[i] for i in range(len(selected_clients))]).to(self.device)
            original_weights = original_weights.view(len(selected_clients), *[1 for _ in range(len(adjusted_weights.shape) - 1)]).expand_as(adjusted_weights)
            weights_for_param = torch.where(total_weight_per_param == 0, original_weights, adjusted_weights / total_weight_per_param)
            for idx, client_id in enumerate(selected_clients):
                update_for_client = client_update[idx][param_key]
                weight_for_client = weights_for_param[idx]
                global_params_new[param_key] = global_params_new[param_key] + update_for_client * weight_for_client
        self.model.load_state_dict(global_params_new)

Client = fedavg.Client