import torch
import numpy as np
import types
import copy
import pathlib
import sys

from typing import Union

from ..utils.training import Trainer
from ..data.neuralop_datasets.dataloader import ns_contextual_loader
from ..utils.training.callbacks import Callback

def to_torch_tensor(data: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(data, list):
        return torch.tensor(np.array(data))
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError("Input type must be list, np.array or torch.tensor")

"""
    the structure is as follows:
    2 types of similar datasets:
    Type 1: dt *= k, k is integer:
        Then the model would need use the model to sequentially predict k steps 
        to compare it's loss with the original outcome.
        In this scenario, for different k, we should calculate them seperately.
        Hence we need k dataloaders, and deal them with the evaluation function 
        with correspondent number of iterations.
    Type 2: dt /= p, p is integer:
        In this scenario, the model would need to predict the p-th step forward.
        So we only need to change the time steps of the dataset, then this dataloader is done.
        This dataloader can be directly put into use, just use the original evaluation function.
"""
def gen_similar_dataloaders_dt_times_k(origin_loader, scaling_ks, batch_size=0):

    def new_get_item(self, index):
        # returns: {'x', 'y', other features}
        p = self.scaling_p
        origin_item =  self.get_item(index)
        new_item = {}
        new_item['x'] = origin_item['x'] * p
        new_item['y'] = origin_item['y'] * p
        new_item['mu'] = origin_item['mu'] * p
        new_item['f'] = origin_item['f'] * (p * p)
        return new_item
    
    def new_get_len(self):
        return self.n_samples * self.n_ticks

    sim_loaders = {}
    scaling_ps = 1. / to_torch_tensor(scaling_ks)
    n_scale_coeff = scaling_ps.shape[0]
    # shallow copy to share the same raw data
    sim_dataset = copy.copy(origin_loader.dataset)
    sim_dataset.__getitem__ = types.MethodType(new_get_item, sim_dataset)
    sim_dataset.__len__ = types.MethodType(new_get_len, sim_dataset)
    sim_dataset.scaling_k = int(1)
    sim_dataset.scaling_p = 1.
    sim_dataset.is_k_integer = True
    if batch_size==0:
        batch_size=origin_loader.batch_size
    for i in range(n_scale_coeff):
        new_dataset = copy.copy(sim_dataset)
        new_dataset.scaling_p = scaling_ps[i]
        new_dataset.scaling_k = scaling_ks[i]
        new_loader = ns_contextual_loader(new_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                          append_positional_encoding=True
                                          )
        sim_loaders[f"dt*=k_{scaling_ks[i]}"] = new_loader

    return sim_loaders


def gen_similar_dataloaders_dt_divided_p(origin_loader, scaling_ps, batch_size=0):

    def new_get_item(self, index):
        # returns: {'x', 'y', other features}
        p = self.scaling_p
        p_int = int(p)
        origin_item =  self.get_item(index, p_int)
        new_item = {}
        new_item['x'] = origin_item['x'] * p
        new_item['y'] = origin_item['y'] * p
        new_item['mu'] = origin_item['mu'] * p
        new_item['f'] = origin_item['f'] * (p * p)
        return new_item
    
    # def new_get_len(self):
    #     return self.n_samples * self.n_ticks

    sim_loaders = {}
    n_scale_coeff = to_torch_tensor(scaling_ps).shape[0]
    # shallow copy to share the same raw data
    sim_dataset = copy.copy(origin_loader.dataset)
    sim_dataset.__getitem__ = types.MethodType(new_get_item, sim_dataset)
    # sim_dataset.__len__ = types.MethodType(new_get_len, sim_dataset)
    sim_dataset.scaling_p = 1.
    sim_dataset.scaling_k = 1.
    sim_dataset.is_k_integer = False
    if batch_size==0:
        batch_size=origin_loader.batch_size
    for i in range(n_scale_coeff):
        new_dataset = copy.copy(sim_dataset)
        new_dataset.scaling_p = scaling_ps[i]
        sim_dataset.scaling_k = 1 / new_dataset.scaling_p
        total_time_number = new_dataset.time_step + new_dataset.n_ticks
        new_dataset.time_step = new_dataset.time_step * new_dataset.scaling_p
        new_dataset.n_ticks = total_time_number - new_dataset.time_step
        new_loader = ns_contextual_loader(new_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                          append_positional_encoding=True
                                          )
        sim_loaders[f"dt/=p_{scaling_ps[i]}"] = new_loader

    return sim_loaders


class SimilarDataloadersCallback(Callback):
    """
        on train start, when the trainer gets the train loader, 
        the callback would automatically generate similar dataloaders, 
        and udpates them to the original loaders dict.
    """
    def __init__(self, scaling_ks, scaling_ps, train_aug=False, test_aug=False):
        super().__init__()
        self.scaling_ks = scaling_ks
        self.scaling_ps = scaling_ps
        self.train_aug = train_aug
        self.test_aug = test_aug

    def on_train_start(self, train_loader, test_loaders, **kwargs):
        self.scaling_ks = self.scaling_ks
        self.scaling_ps = self.scaling_ps
        if isinstance(train_loader, torch.utils.data.DataLoader):
            # This is the case where train_loaders is a single DataLoader instance.
            train_loader = {'default': train_loader}
        elif isinstance(train_loader, dict):
            # This is the case where train_loaders is a dictionary of DataLoader instances.
            pass
        else:
            raise ValueError("train_loaders must be either a DataLoader or a dictionary of DataLoader instances.")
        print(type(train_loader))
        print(train_loader.keys())
        dict_k_loaders = gen_similar_dataloaders_dt_times_k(train_loader['default'], scaling_ks=self.scaling_ks)
        dict_p_loaders = gen_similar_dataloaders_dt_divided_p(train_loader['default'], scaling_ps=self.scaling_ps)

        if self.train_aug:
            train_loader.update(dict_k_loaders)
            train_loader.update(dict_p_loaders)
        for key, loader in train_loader.items():
            item = None
            # print(loader.__len__())
            for item in loader:
                break
            print(f"{key}: {item['x'].shape=}, len: {loader.dataset.__len__()}, len: {loader.dataset.n_ticks}, N: {loader.dataset.n_samples}")
            sys.stdout.flush()
        
        if self.test_aug:
            test_loaders.update(dict_k_loaders)
            test_loaders.update(dict_p_loaders)
        for key, loader in test_loaders.items():
            item = None
            # print(loader.__len__())
            for item in loader:
                break
            print(f"{key}: {item['x'].shape=}, len: {loader.dataset.__len__()}, len: {loader.dataset.n_ticks}, N: {loader.dataset.n_samples}")
            sys.stdout.flush()

import time

class ns_contextual_trainer(Trainer):
    def __init__(self, model, 
                 n_epochs, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False, 
                 callbacks = [], 
                 scaling_ks=[1,4,16,64], scaling_ps=[1,4,16,64],
                 simaug_test_data=False, simaug_train_data=False, 
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 checkpoint_to_load: pathlib.Path=None, 
                 verbose=True):
        if simaug_test_data or simaug_train_data:
            print(simaug_test_data, simaug_train_data)
            callbacks.append(SimilarDataloadersCallback(scaling_ks=scaling_ks, scaling_ps=scaling_ps, 
                                                        train_aug=simaug_train_data, test_aug=simaug_test_data))
        super(ns_contextual_trainer, self).__init__(
                    model=model, 
                    n_epochs=n_epochs, 
                    wandb_log=wandb_log, 
                    device=device, 
                    amp_autocast=amp_autocast, 
                    callbacks=callbacks,
                    log_test_interval=log_test_interval, 
                    log_output=log_output, 
                    use_distributed=use_distributed, 
                    checkpoint_to_load=checkpoint_to_load,
                    verbose=verbose
                )
        
    
    def evaluate(self, loss_dict, data_loader, log_prefix=''):
        use_k_steps = False
        divide_p_parts = False
        # print(time.time())
        if hasattr(data_loader.dataset, 'is_k_integer'):
            if data_loader.dataset.is_k_integer:
                use_k_steps = True
            else: divide_p_parts = True
        if use_k_steps:
            return self.evaluate_k_steps_forward(loss_dict, data_loader, k=data_loader.dataset.scaling_k, log_prefix=log_prefix)
        if divide_p_parts:
            return super().evaluate(loss_dict, data_loader, log_prefix=log_prefix)
        return super().evaluate(loss_dict, data_loader, log_prefix)

    def evaluate_k_steps_forward(self, loss_dict, data_loader, k, log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # print('using k steps forward!')
        k = int(k)
        # log_prefix = log_prefix + str(k)

        if self.callbacks:
            self.callbacks.on_val_epoch_start(loss_dict = loss_dict, data_loader=data_loader)

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        eval_start_time = time.time()
        evaluation_time = 0
        with torch.no_grad():
            
            for idx, sample in enumerate(data_loader):
                batch_start_time = time.time()
                y = sample['y']
                n_samples += y.size(0)

                # Note that we should predict the next k steps, with the same forces and mu conditions.
                # Here we assume that the output represents the dame physical quantity as first out-dim inputs.
                out_dim = self.model.out_channels
                for _ in range(k):
                    out = self.predict_step(idx=idx, sample=sample)
                    sample['x'][:, :out_dim, ...] = out
                
                if self.callbacks:
                    self.callbacks.on_before_val_loss(out=out)
                
                for loss_name, loss in loss_dict.items():
                    if self.overrides_loss:
                        if isinstance(out, torch.Tensor):
                            val_loss = self.callbacks.compute_training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            val_loss = self.callbacks.compute_training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            val_loss = loss(out, **sample).item()
                        elif isinstance(out, dict):
                            val_loss = loss(out, **sample).item()

                    errors[f'{log_prefix}_{loss_name}'] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
                    evaluation_time += time.time() - batch_start_time
        
        preprocess_time = time.time() - eval_start_time - evaluation_time

        del y, out

        for key in errors.keys():
            errors[key] /= n_samples

        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors)

        if self.verbose: print(log_prefix, preprocess_time)
        return errors

    # def evaluate_dt_divided_p(self, loss_dict, log_prefix='dt/=p_'):
    #     return self.evaluate(loss_dict,self.p_loader,log_prefix)
    


# def gen_similar_dataloaders_dt_divided_p(origin_loader, scaling_ps, batch_size=0):

#     sim_dataset = copy.copy(origin_loader.dataset)
#     sim_dataset.scaling_ps = to_torch_tensor(scaling_ps)
#     sim_dataset.n_scale_coeff = sim_dataset.scale_coeffs.shape[0]
#     total_time_number = sim_dataset.time_step + sim_dataset.n_ticks
#     max_p = int(torch.max(sim_dataset.scaling_ps))
#     sim_dataset.n_ticks = total_time_number - max_p

#     def new_get_item(self, index):
#         # returns: {'x', 'y', other features}
#         origin_idx = index // self.n_scale_coeff
#         p_idx = index % self.n_scale_coeff
#         p = self.scale_coeffs[p_idx]
#         p_int = int(p)
#         # should return n steps later
#         origin_item =  self.get_item(origin_idx, p_int)
#         new_item = {}
#         new_item['x'] = origin_item['x'] * p
#         new_item['y'] = origin_item['y'] * p
#         new_item['mu'] = origin_item['mu'] * p
#         new_item['f'] = origin_item['f'] * (p * p)
#         return new_item
    
#     def new_get_len(self):
#         return self.n_scale_coeff * self.n_samples * self.n_ticks

#     sim_dataset.__getitem__ = types.MethodType(new_get_item, sim_dataset)
#     sim_dataset.__len__ = types.MethodType(new_get_len, sim_dataset)

#     if batch_size==0:
#         batch_size=origin_loader.batch_size * sim_dataset.n_scale_coeff
#     sim_loader = ns_contextual_loader(sim_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
#                                       append_positional_encoding=True
#                                       )
#     return sim_loader