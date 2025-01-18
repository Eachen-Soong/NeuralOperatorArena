import h5py
import torch
from torch.utils.data import Dataset
from ..utils import UnitGaussianNormalizer
from .transforms import PositionalEmbedding
from .dataloader import ns_contextual_loader
from torch.utils.data.dataloader import DataLoader
from h5py import File
import random
import scipy

class AutoregressiveDataset(Dataset):
    """ Note that this version of dataset just contains data x, 
        instead of preparing correspondent y in advance.
        Because the y is just the x after k time steps! 
        We needn't create a new dataset each time we alter k!
        the dataset can have different input attributes, e.g. for ns_contextual (F-FNO): u, f, mu

        Parameters
        ----------
        data: a dict-like object
        subsample_rate: subsample rate across the spatial dimension
        time_step: predict the state after which time step
        predict_feature: the feature to predict, default is 'u'.
        constant_dims: {feature: (constant_in_space, constant_in_time)}, 
                        default is None, automatically filled
        TODO: support multiple predict_features, 1D/3D datasets
    """
    def __init__(self, data, subsample_rate=1, time_step=1, preprocess_space_ssr=True, predict_feature='u', constant_dims=None):
        # self.data = data # a dict of array by default
        self.ssr = subsample_rate
        self.time_step = time_step
        self.preprocess_space_ssr = preprocess_space_ssr
        self.predict_feature = predict_feature

        assert (predict_feature=='x' or 'x' not in data.keys()) and 'y' not in data.keys(), "Name 'x' or 'y' may cause conflicts."
        # Each feature should be like [sample_index, space, time] or [sample_index, space] or [sample_index, time] or [sample_index]
        len_shapes = {data_name: len(data[data_name].shape) for data_name in data}
        max_dim_data_name = max(len_shapes, key=len_shapes.get)
        self.n_ticks = data[max_dim_data_name].shape[-1] - time_step
        self.n_samples = data[max_dim_data_name].shape[0]
        self.spatial_resolution = data[max_dim_data_name].shape[1: -1]
        self.dimension = len(self.spatial_resolution)

        shapes = {data_name: data[data_name].shape[1:] for data_name in data}

        # accross which dim is the data considered constant
        assert not (self.dimension == 1 and self.spatial_resolution[-1] == self.n_ticks+time_step) or constant_dims, "Can't tell space from time, please input constant_dims"
        def judge_constant_dims(shape):
            if len(shape) == self.dimension + 1:
                return (False, False)
            else:
                if self.dimension != 1:
                    if len(shape) == self.dimension:
                        return (False, True)
                    elif len(shape) == 1:
                        return (True, False)
                    elif len(shape) == 0:
                        return (True, True)
                else:
                    if len(shape) == 0:
                        return (True, True)
                    elif shape[-1] == self.spatial_resolution[-1]:
                        return (False, True)
                    else:
                        return (True, False)

        self.constant_dims = constant_dims if constant_dims is not None \
            else {name: judge_constant_dims(shapes[name]) for name in data}
        
        if preprocess_space_ssr:
            for name in data:
                if not self.constant_dims[name][0]:
                    if self.dimension == 2:
                        data[name] = data[name][:, ::self.ssr, ::self.ssr, ...]
                    elif self.dimension == 1:
                        data[name] = data[name][:, ::self.ssr, ...]
                    elif self.dimension == 3:
                        data[name] = data[name][:, ::self.ssr, ::self.ssr, ::self.ssr, ...]
            self.spatial_resolution = data[max_dim_data_name].shape[1: -1]
        self.data = data
        del data

        if self.dimension == 2:
            if preprocess_space_ssr:
                def getitem_2D(idx):
                    b = idx // self.n_ticks
                    t = idx % self.n_ticks
                    item = {'y': self.data[self.predict_feature][b, ..., t+self.time_step]}
                    
                    for name in self.data:
                        constant_dim = self.constant_dims[name]
                        if constant_dim[0]:
                            if constant_dim[1]:
                                item[name] = self.data[name][b]
                            else: item[name] = self.data[name][b, t]
                        else:
                            if constant_dim[1]:
                                item[name] = self.data[name][b, ...]
                            else: item[name] = self.data[name][b, ..., t]

                    item['x'] = item.pop(self.predict_feature)

                    return item
            else:
                def getitem_2D(idx, n_steps=1):
                    b = idx // self.n_ticks
                    t = idx % self.n_ticks

                    item = {'y': self.data[self.predict_feature][b, ::self.ssr, ::self.ssr, t+n_steps*self.time_step]}
                    for name in self.data:
                        constant_dim = self.constant_dims[name]
                        if constant_dim[0]:
                            if constant_dim[1]:
                                item[name] = self.data[name][b]
                            else: item[name] = self.data[name][b, t]
                        else:
                            if constant_dim[1]:
                                item[name] = self.data[name][b, ::self.ssr, ::self.ssr]
                            else: item[name] = self.data[name][b, ::self.ssr, ::self.ssr, t]
                    
                    item['x'] = item.pop(self.predict_feature)

                    return item
                
            self.get_item = getitem_2D

        else: assert self.dimension == 2, f"dimension = {self.dimension}, Currently only 2D supported"

    def __len__(self):
        return self.n_samples * self.n_ticks
    
    def __getitem__(self, index):
        # returns: {'x', 'y', other features}
        return self.get_item(index)

# From the initial implementations, we've cut out some functions such as different subsample rate across different dimensions, or to upsample the dataset.
# Note that we've taken the train_transform part (i.e. concatenating grid to model input), into the model itself.
# This is just like what is done in language models, we consider the positional encoding as a part of this model, 
# instead of some "input data".


def load_autoregressive_traintestsplit_v3(
                        data_path, 
                        n_train, n_tests,
                        batch_size, test_batch_size, 
                        train_subsample_rate, test_subsample_rates,
                        time_step,
                        test_data_paths=[''],
                        predict_feature='u',
                        append_positional_encoding=True,
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
            current_data = torch.tensor(data[name][0:n_train, ...]).type(torch.float32)
            train_data[name] = current_data.contiguous()

    del data

    if train_data is not None:
        train_db = AutoregressiveDataset(train_data, subsample_rate=train_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        train_loader = ns_contextual_loader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=71,
                                            append_positional_encoding=append_positional_encoding)
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
                current_data = torch.tensor(data[name][-n_test:, ...]).type(torch.float32)
                test_data[name] = current_data.contiguous()

        del data

        if test_data is not None:
            test_db = AutoregressiveDataset(test_data, subsample_rate=test_subsample_rate, time_step=time_step, predict_feature=predict_feature)
            test_loader = ns_contextual_loader(test_db,
                                            batch_size=test_batch_size, shuffle=False,
                                            num_workers=71,
                                            append_positional_encoding=append_positional_encoding)
        else:
            test_loader = None

        test_loaders[f'id_{idx}_subsample_rate{test_subsample_rate}'] = test_loader
        idx+=1

    return train_loader, test_loaders


def load_autoregressive_traintestsplit_v2(
                        data_path, 
                        n_train, n_test,
                        batch_size, test_batch_size, 
                        train_subsample_rate, test_subsample_rate,
                        time_step,
                        test_data_path='',
                        predict_feature='u',
                        append_positional_encoding=True,
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
    encoders : UnitGaussianNormalizer List[UnitGaussianNormalizer] None
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
            current_data = torch.tensor(data[name][0:n_train, ...]).type(torch.float32)
            train_data[name] = current_data.contiguous()

    del data

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
            current_data = torch.tensor(data[name][-n_test:, ...]).type(torch.float32)
            test_data[name] = current_data.contiguous()

    del data

    if train_data is not None:
        train_db = AutoregressiveDataset(train_data, subsample_rate=train_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        train_loader = ns_contextual_loader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=71,
                                            append_positional_encoding=append_positional_encoding)
    else:
        train_loader = None

    if test_data is not None:
        test_db = AutoregressiveDataset(test_data, subsample_rate=test_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        test_loader = ns_contextual_loader(test_db,
                                           batch_size=test_batch_size, shuffle=False,
                                           num_workers=71,
                                           append_positional_encoding=append_positional_encoding)
    else:
        test_loader = None

    return train_loader, test_loader


def load_autoregressive_traintestsplit_v1(data_path, 
                        n_train, n_test,
                        batch_size, test_batch_size, 
                        train_subsample_rate, test_subsample_rate,
                        time_step,
                        predict_feature='u',
                        append_positional_encoding=True,
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
    encoders : UnitGaussianNormalizer List[UnitGaussianNormalizer] None
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
            current_data = torch.tensor(data[name][0:n_train, ...]).type(torch.float32)
            train_data[name] = current_data.contiguous()

    test_data = None
    if n_test > 0:
        test_data = {}
        for name in data:
            current_data = torch.tensor(data[name][n_train:(n_train + n_test), ...]).type(torch.float32)
            test_data[name] = current_data.contiguous()

    del data

    if train_data is not None:
        train_db = AutoregressiveDataset(train_data, subsample_rate=train_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        train_loader = ns_contextual_loader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=71,
                                            append_positional_encoding=append_positional_encoding)
    else:
        train_loader = None

    if test_data is not None:
        test_db = AutoregressiveDataset(test_data, subsample_rate=test_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        test_loader = ns_contextual_loader(test_db,
                                           batch_size=test_batch_size, shuffle=False,
                                           num_workers=71,
                                           append_positional_encoding=append_positional_encoding)
    else:
        test_loader = None

    return train_loader, test_loader

def load_autoregressive_traintestsplit(data_path, 
                        n_train, n_test,
                        batch_size, test_batch_size, 
                        train_subsample_rate, test_subsample_rate,
                        time_step,
                        predict_feature='u',
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
    encoders : UnitGaussianNormalizer List[UnitGaussianNormalizer] None
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
            current_data = torch.tensor(data[name][0:n_train, ...]).type(torch.float32)
            train_data[name] = current_data.contiguous()

    test_data = None
    if n_test > 0:
        test_data = {}
        for name in data:
            current_data = torch.tensor(data[name][n_train:(n_train + n_test), ...]).type(torch.float32)
            test_data[name] = current_data.contiguous()

    del data

    if train_data is not None:
        train_db = AutoregressiveDataset(train_data, subsample_rate=train_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        train_loader = DataLoader(train_db,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=71, pin_memory=True, persistent_workers=False)
    else:
        train_loader = None

    if test_data is not None:
        test_db = AutoregressiveDataset(test_data, subsample_rate=test_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        test_loader = DataLoader(test_db,
                                                batch_size=test_batch_size, shuffle=False,
                                                num_workers=71, pin_memory=True, persistent_workers=False)
    else:
        test_loader = None

    return train_loader, test_loader


def load_autoregressive_multitask_mu_preordered(
                        data_path, 
                        splits:list,
                        batch_size, test_batch_size, 
                        train_subsample_rate, test_subsample_rate,
                        time_step,
                        n_data=-1,  
                        predict_feature='u',
                        append_positional_encoding=True,
                        seed=42
                        ):
    """Create a Multi-Task Learning dataset according to mu:
    the dataset is pre-ordered, so just simply load the dataset.

    Parameters
    ----------
    splits : list of (train_prop, test_prop), the splits of the datasets
        if train_prop or test_prop is 0, the counterpart would be None
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
    encoders : UnitGaussianNormalizer List[UnitGaussianNormalizer] None
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
    if type(test_subsample_rate) == type([]):
        test_subsample_rate = test_subsample_rate[0]

    # get part of the total dataset
    n_total = data[predict_feature].shape[0]
    if n_data != -1:
        new_data = {}
        shuffled_idx = list(range(n_total))
        random.shuffle(shuffled_idx)
        chosen_data = sorted(shuffled_idx)
        for name in data:
            new_data[name] = data[name][chosen_data]
        n_total = n_data
        del data
        data = new_data
    
    # normalize
    n_splits = len(splits)

    sum_splits = [sum(split) for split in splits]
    sum_sum_splits = sum(sum_splits)
    # the sturcture of an element of train_test_splits: (total number of the split, train split)
    new_splits = [(round(sum_splits[i] * n_total / sum_sum_splits), round(splits[i][0] * n_total / sum_sum_splits)) for i in range(n_splits-1)]
    new_splits.append((n_total - sum([split[0] for split in new_splits[:]]), round(splits[-1][0] * n_total / sum_sum_splits)))

    train_loaders = []
    test_loaders = []
    split_idx = 0
    for i in range(n_splits):
        new_split_idx = split_idx + new_splits[i][0]
        tmp_train_data = {}
        tmp_test_data  = {}
        shuffled_idx = list(range(split_idx, new_split_idx))
        random.shuffle(shuffled_idx)
        train_slice = sorted(shuffled_idx[:new_splits[i][1]])
        test_slice = sorted(shuffled_idx[new_splits[i][1]:])

        for name in data:
            if len(train_slice):
                tmp_train_data[name] = torch.tensor(data[name][train_slice, ...]).type(torch.float32).contiguous()
                
            else: tmp_trainloader = None
            
            if len(test_slice):
                tmp_test_data[name] = torch.tensor(data[name][test_slice, ...]).type(torch.float32).contiguous()
                
            else: tmp_testloader = None
        
        tmp_train_db = AutoregressiveDataset(tmp_train_data, subsample_rate=train_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        # tmp_trainloader = DataLoader(tmp_train_db,
        #                     batch_size=batch_size, shuffle=True,
        #                     num_workers=0, pin_memory=True, persistent_workers=False)
        tmp_trainloader = ns_contextual_loader(tmp_train_db,
                            batch_size=batch_size, shuffle=True,
                            num_workers=71,
                            append_positional_encoding=append_positional_encoding)
        
        tmp_test_db = AutoregressiveDataset(tmp_test_data, subsample_rate=test_subsample_rate, time_step=time_step, predict_feature=predict_feature)
        # tmp_testloader = DataLoader(tmp_test_db,
        #                     batch_size=batch_size, shuffle=True,
        #                     num_workers=0, pin_memory=True, persistent_workers=False)
        tmp_testloader = ns_contextual_loader(tmp_test_db,
                                           batch_size=test_batch_size, shuffle=False,
                                           num_workers=71,
                                           append_positional_encoding=append_positional_encoding)
        
        train_loaders.append(tmp_trainloader)
        test_loaders.append(tmp_testloader)
        split_idx = new_split_idx
    
    del data

    return train_loaders, test_loaders