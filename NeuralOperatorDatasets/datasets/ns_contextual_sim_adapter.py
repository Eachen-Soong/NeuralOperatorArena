from .autoregressive_dataset import AutoregressiveDataset
import torch
import numpy as np
import types
import copy
from typing import Union

def to_torch_tensor(data: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(data, list):
        return torch.tensor(np.array(data))
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError("Input type must be list, np.array or torch.tensor")


def gen_similar_dataloader(origin_loader, scale_coeffs, batch_size=0):

    sim_dataset = copy.copy(origin_loader.dataset)
    sim_dataset.scale_coeffs = to_torch_tensor(scale_coeffs)
    sim_dataset.n_scale_coeff = sim_dataset.scale_coeffs.shape[0]

    def new_get_item(self, index):
        print('fuck')
        # returns: {'x', 'y', other features}
        origin_idx = index // self.n_scale_coeff
        k_idx = index % self.n_scale_coeff
        k = self.scale_coeffs[k_idx]
        origin_item =  self.get_item(origin_idx)
        new_item = {}
        new_item['x'] = origin_item['x'] / k
        new_item['y'] = origin_item['y'] / k
        new_item['mu'] = origin_item['mu'] / k
        new_item['f'] = origin_item['f'] / (k**2)
        return new_item
    
    def new_get_len(self):
        return self.n_scale_coeff * self.n_samples * self.n_ticks

    sim_dataset.__getitem__ = types.MethodType(new_get_item, sim_dataset)
    sim_dataset.__len__ = types.MethodType(new_get_len, sim_dataset)

    if batch_size==0:
        batch_size=origin_loader.batch_size * sim_dataset.n_scale_coeff
    sim_loader = torch.utils.data.DataLoader(sim_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0, pin_memory=True, persistent_workers=False)

    return sim_loader