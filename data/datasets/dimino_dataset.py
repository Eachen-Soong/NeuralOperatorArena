import h5py
import torch
from torch.utils.data import Dataset
from ..utils import UnitGaussianNormalizer
from .transforms import PositionalEmbedding
from .dataloader import ns_contextual_loader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import default_collate # it's a function
from h5py import File
import random
import scipy
from typing import Union
import numpy as np
from einops import repeat

def to_torch(x: Union[np.ndarray, torch.Tensor]):
    if type(x) == torch.Tensor:
        return x
    return torch.from_numpy(x)

class FieldPredDataset(Dataset):
    """
    the dataset seperates the constants and 
    """
    def __init__(self, data, time_step=1, predict_features=['u']):
        super().__init__()
        self.time_step = time_step
        self.predict_features = predict_features
        var_field_pred = []
        var_field_nonpred = []
        const_field = []
        constants = []
        self.var_field_pred_names = []
        self.var_field_nonpred_names = []
        self.const_field_names = []
        self.constants_names = []
        standard_shape = data[predict_features[0]].shape
        n_dim = len(standard_shape) - 2
        self.n_ticks = data[predict_features[0]].shape[-1] - time_step
        self.n_samples = data[predict_features[0]].shape[0]
        self.time_step = time_step
        self.num_preds = len(predict_features)
        for name in data.keys():
            tmp = to_torch(data[name])
            if len(tmp.shape) == 1:
                constants.append(tmp)
                self.constants_names.append(name)
            elif len(tmp.shape) == n_dim+1:
                const_field.append(tmp)
                self.const_field_names.append(name)
            elif len(tmp.shape) == n_dim+2:
                if name in predict_features:
                    var_field_pred.append(tmp)
                    self.var_field_pred_names.append(name)
                else:
                    var_field_nonpred.append(tmp)
                    self.var_field_nonpred_names.append(name)

        self.constants = torch.stack(constants, dim=1)
        self.const_field = torch.stack(const_field, dim=1)
        var_field_pred.extend(var_field_nonpred)
        self.var_field = torch.stack(var_field_pred, dim=1)

    def __len__(self):
        return self.n_samples * self.n_ticks

    def __getitem__(self, index):
        b = index // self.n_ticks
        t = index % self.n_ticks
        fields = torch.cat([self.var_field[b, ..., t], self.const_field[b]])
        y = self.var_field[b, -self.num_preds:, ..., t+self.time_step]
        return {'consts': self.constants[b], 'x': fields, 'y': y}

class FieldPredLoader(DataLoader):
    """
        This dataloader is suitable for all datasets that with some const-dims as input but no const-dims as output.
        the collate function is based on the dim appenders that we've automatically generated.
        This structure can preprocess the data in batches, 
        unlike the original version, where all these functions are defined within a dataset,
        and the dataset has to calculate these things repeatedly.
        (for TorusLi: concating the grids batch_size-1 more times)
        (for ns_contextual: the grid stuff, and expanding the 'f', 'mu' batch_size-1 more times)
        Note that The dataset should be 2D here.
    """
    def __init__(self, dataset, batch_size, shuffle=True, seperate_consts=False, num_workers=16, **kwargs):
        super(FieldPredLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        spacial_resolution = None
        self.collate_fn = default_collate
        for item in dataset:
            break
        spacial_resolution  = item['x'].shape[1:] # [(space)]
        dimension = len(spacial_resolution)

        if not seperate_consts:
            if dimension == 2: broadcast_function = lambda data: repeat(data, 'b c -> b c m n', m=spacial_resolution[0], n=spacial_resolution[1])
            elif dimension == 1: broadcast_function = lambda data: repeat(data, 'b c -> b c m', m=spacial_resolution[0])
            elif dimension == 3: broadcast_function = lambda data: repeat(data, 'b c -> b c m n p', m=spacial_resolution[0], n=spacial_resolution[1], p=spacial_resolution[2])
            def _collate_fn(batch):
                batch = default_collate(batch)
                new_batch = {'x': torch.cat([batch['x'], broadcast_function(batch['consts'])], dim=1),
                             'y': batch['y']
                             }
                return new_batch
            
            self.collate_fn = _collate_fn



def load_autoregressive_traintestsplit_dim(
                        data_path, 
                        n_train, n_tests,
                        batch_size, test_batch_size, 
                        train_subsample_rate, test_subsample_rates,
                        time_step,
                        time_skips=10,
                        test_data_paths=[''],
                        seperate_consts=False,
                        predict_features=['u'],
                        ):
    """Create train-test split from a single file
    containing any number of tensors. n_train or
    n_test can be zero. First n_train
    points are used for the training set and n_test of
    the remaining points are used for the test set.
    If subsampling or interpolation is used, all tensors 
    are assumed to be of the same dimension and the 
    operation will be applied to all.

    Parameters
    ----------
    n_train : int
    n_test : int
    batch_size : int
    test_batch_size : int
    labels: str list, default is 'x'
        tensor labels in the data file
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool list, default is True
    gaussian_norm : bool list, default is False
    norm_type : str, default is 'channel-wise'
    channel_dim : int list, default is 1
        where to put the channel dimension, defaults size is batch, channel, height, width

    Returns
    -------
    train_loader, test_loader

    train_loader : torch DataLoader None
    test_loader : torch DataLoader None
    """
    dataset_type = 'h5' if data_path.endswith('.h5') else ('pt' if data_path.endswith('.pt') else 'mat')
    if dataset_type == 'h5':
        data = h5py.File(data_path, 'r')
    elif dataset_type == 'pt':
        data = torch.load(data_path)
    else:
        try:
            data = scipy.io.loadmat(data_path)
            del data['__header__']
            del data['__version__']
            del data['__globals__']
            del data['a']
            del data['t']
        except:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_data = None
    if n_train > 0:
        train_data = {}
        for name in data:
            if len(data[name].shape) == 3:
                train_data[name] = data[name][:n_train, ::train_subsample_rate, ::train_subsample_rate]
            elif len(data[name].shape) == 4:
                train_data[name] = data[name][:n_train, ::train_subsample_rate, ::train_subsample_rate, ::time_skips]
            elif len(data[name].shape) == 5:
                train_data[name] = data[name][:n_train, :, ::train_subsample_rate, ::train_subsample_rate, ::time_skips]
            elif len(data[name].shape) > 5:
                train_data[name] = data[name][:n_train, :, ::train_subsample_rate, ::train_subsample_rate, ::time_skips, ...]
            else:
                train_data[name] = data[name][:n_train, ...]
    
            train_data[name] = torch.tensor(train_data[name]).type(torch.float32)

    del data
    if train_data is not None:
        train_db = FieldPredDataset(train_data, time_step=time_step, predict_features=predict_features)
        train_loader = FieldPredLoader(train_db, 
                                  batch_size=batch_size, shuffle=True, seperate_consts=seperate_consts,
                                  num_workers=71,)
    else:
        train_loader = None

    if type(n_tests) == type(1):
        n_tests = [n_tests]

    if type(test_subsample_rates) == type(1):
        test_subsample_rates = [test_subsample_rates]

    if type(test_data_paths) == type(''):
        test_data_paths = [test_data_paths]

    num_test_loaders = max(len(n_tests), len(test_subsample_rates), len(test_data_paths))

    if len(n_tests) == 1:
        n_tests = n_tests * num_test_loaders
    
    if len(test_subsample_rates) == 1:
        test_subsample_rates = test_subsample_rates * num_test_loaders
    
    if len(test_data_paths) == 1:
        test_data_paths = test_data_paths * num_test_loaders

    test_loaders = dict()
    idx=0
    for (n_test, test_data_path, test_subsample_rate) in zip(n_tests, test_data_paths, test_subsample_rates):
        if test_data_path == None or test_data_path == '':
            test_data_path = data_path
        
        if dataset_type == 'h5':
            data = h5py.File(test_data_path, 'r')
        elif dataset_type == 'pt':
            data = torch.load(test_data_path)
        else:
            try:
                data = scipy.io.loadmat(test_data_path)
                del data['__header__']
                del data['__version__']
                del data['__globals__']
                del data['a']
                del data['t']
            except:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        test_data = None
        if n_test > 0:
            test_data = {}
            for name in data:
                for name in data:
                    if len(data[name].shape) == 3:
                        test_data[name] = data[name][:n_test, ::test_subsample_rate, ::test_subsample_rate]
                    elif len(data[name].shape) == 4:
                        test_data[name] = data[name][:n_test, ::test_subsample_rate, ::test_subsample_rate, ::time_skips]
                    elif len(data[name].shape) == 5:
                        test_data[name] = data[name][:n_test, :, ::test_subsample_rate, ::test_subsample_rate, ::time_skips]
                    elif len(data[name].shape) > 5:
                        test_data[name] = data[name][:n_test, :, ::test_subsample_rate, ::test_subsample_rate, ::time_skips, ...]
                    else:
                        test_data[name] = data[name][:n_test, ...]

        del data

        if test_data is not None:
            test_db = FieldPredDataset(test_data, time_step=time_step, predict_features=predict_features)
            test_loader = FieldPredLoader(test_db,
                                  batch_size=test_batch_size, shuffle=False, seperate_consts=seperate_consts,
                                  num_workers=71,)
        else:
            test_loader = None

        test_loaders[f'id_{idx}_subsample_rate{test_subsample_rate}'] = test_loader
        idx+=1

    return train_loader, test_loaders

