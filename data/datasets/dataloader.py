import torch

from einops import repeat
from torch.utils.data.dataloader import DataLoader
from .positional_encoding import get_grid_positional_encoding
from .transforms import PositionalEmbedding
from torch.utils.data import default_collate # it's a function
import copy

class ns_contextual_loader(DataLoader):
    """
        the collate function is based on the dim appenders that we've automatically generated.
        This structure can preprocess the data in batches, 
        unlike the original version, where all these functions are defined within a dataset,
        and the dataset has to calculate these things repeatedly.
        (for TorusLi: concating the grids batch_size-1 more times)
        (for ns_contextual: the grid stuff, and expanding the 'f', 'mu' batch_size-1 more times)
        Note that The dataset should be 2D here.
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, append_positional_encoding=False, positional_encoding=None, grid_boundaries=[[0,1],[0,1]], channel_dim=1, **kwargs):
        super(ns_contextual_loader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        self.spacial_resolution = None
        self.dim_appenders = [] # [(name, broadcasdefault_collatet function)]
        self.append_positional_encoding = append_positional_encoding
        self.positional_encoding = positional_encoding
        self.grid_boundaries = grid_boundaries
        self.collate_fn = default_collate
        self.channel_dim = channel_dim
        
        """ Build the dim_app enders according to the dataset dims """
        item = None
        for item in dataset:
            break
        self.spacial_resolution  = item['x'].shape # [(space)]
        # print(f"self.spacial_resolution: {self.spacial_resolution}")
        self.dimension = len(self.spacial_resolution)
        # TODO: grids of 1D, 3D
        # if self.append_positional_encoding and self.positional_encoding is None:
        #     assert self.dimension == 2, "Positional encoding currently only supported for 2D"
            # self.positional_encoding = PositionalEmbedding(grid_boundaries, channel_dim=1)
        if self.positional_encoding is None and self.append_positional_encoding:
            assert self.dimension == 2, "Positional encoding currently only supported for 2D"
            grids = get_grid_positional_encoding(item['x'], self.grid_boundaries, channel_dim=0)
            self.positional_encoding = torch.cat((grids[0], grids[1]), dim=0)
            # self.positional_encoding = self.positional_encoding.to(self.device)
        
        for name in item:
            if name == 'x' or name == 'y':
                continue
            length = len(item[name].shape)
            if length == self.dimension: broadcast_function = lambda data: data.unsqueeze(self.channel_dim)
            else:
                if self.dimension == 2: broadcast_function = lambda data: repeat(data, 'b -> b m n', m=self.spacial_resolution[0], n=self.spacial_resolution[1]).unsqueeze(self.channel_dim)
                elif self.dimension == 1: broadcast_function = lambda data: repeat(data, 'b -> b m', m=self.spacial_resolution[0]).unsqueeze(self.channel_dim).unsqueeze(self.channel_dim)
                elif self.dimension == 3: broadcast_function = lambda data: repeat(data, 'b -> b m n p', m=self.spacial_resolution[0], n=self.spacial_resolution[1], p=self.spacial_resolution[2]).unsqueeze(self.channel_dim)
                else: raise ValueError(f"Invalid dimension {self.dimension}")
            self.dim_appenders.append((name, broadcast_function))

        if self.append_positional_encoding:
            self.dim_appenders.append(('x', lambda data: repeat(self.positional_encoding, 'm n c -> b m n c', b=data.shape[0])))

        def _collate_fn(batch):
            batch = default_collate(batch)
            batch = self.build_features(batch)
            return batch
        
        self.collate_fn = _collate_fn

    def build_features(self, batch):
        x = batch['x']
        x.unsqueeze_(self.channel_dim)
        for (name, appender) in self.dim_appenders:
            feature = appender(batch[name])
            x = torch.cat([x, feature], dim=self.channel_dim)

        new_batch = batch
        new_batch['x'] = x
        # If the dataset only has one predictable target
        new_batch['y'].unsqueeze_(self.channel_dim)
        return new_batch
