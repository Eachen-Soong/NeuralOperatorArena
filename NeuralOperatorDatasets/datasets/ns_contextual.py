from pathlib import Path
import torch
from einops import repeat

from .autoregressive_dataset import load_autoregressive_traintestsplit

"""
NS-contextual Introduction:
This dataset is 2D vorticity NS dataset, with force carying with space (and time), and different viscosities.

The Pipeline:
Dataset: AutoregressiveDataset
        {data['u'][n, x, y, t], data['f'][n, x, y], data['mu'][n]} -> {batch['x'], ['y'], ['f'], ['mu']}
        Does dataset preprocessing: subsample rate on space

Callbacks: MultipleInputCallback
        on_batch_start: appends the f and mu (and grids for the old version)
        # this is not implemented in Dataset.__getitem__ so that we can build the features in batches instead of item by item

Model: positional_encoding: appends grids(or other) # not included in the old version
"""

def load_ns_contextual_toy(
    n_train,
    n_test,
    batch_size,
    test_batch_size,
    train_resolution=64,
    test_resolution=64,
    time_step=8,
):
    """Loads a Navier-Stokes toy dataset (with viscosity and force, invariant with time)
    Note that this original dataset is on x-y grid 256 * 256.
    if we need to test on a lower resolution, we just devide that by parameter ssr (sub-sample rate).
    
    For test dataset, we would make the model autoregressively predict the next n_steps outputs and calculate the total loss.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [16, 32],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : { 'x': self.u[b, ... , t],
                            'y': self.u[b, ... , t+self.time_step],
                            'mu': self.mu[b],
                            'f': f, }
    testing_dataloaders : { 'data': self.u[b, ... , ::self.k],
                            'mu': self.mu[b],
                            'f': f,
                            'times': self.times }
    """
    
    path = Path(__file__).resolve().parent.joinpath("data")
    data_path = str(path) + "/ns_random_forces_toy1.h5"
    dataset_resolution = 256
    train_subsample_rate = dataset_resolution // train_resolution
    test_subsample_rate = dataset_resolution // test_resolution
    assert not(dataset_resolution % train_resolution and dataset_resolution % test_resolution), f"train/test resolution must be a divisor of the dataset resolution {dataset_resolution}!"
    
    return load_autoregressive_traintestsplit(
    data_path,
    n_train, n_test,
    batch_size, test_batch_size, 
    train_subsample_rate, test_subsample_rate,
    time_step,
    predict_feature='u',
)


def load_ns_contextual_mini(
    n_train,
    n_test,
    batch_size,
    test_batch_size,
    train_resolution=64,
    test_resolution=64,
    time_step=8,
):
    """Loads a Navier-Stokes toy dataset (with viscosity and force, invariant with time)
    Note that this original dataset is on x-y grid 256 * 256.
    if we need to test on a lower resolution, we just devide that by parameter ssr (sub-sample rate).
    
    For test dataset, we would make the model autoregressively predict the next n_steps outputs and calculate the total loss.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [16, 32],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : { 'x': self.u[b, ... , t],
                            'y': self.u[b, ... , t+self.time_step],
                            'mu': self.mu[b],
                            'f': f, }
    testing_dataloaders : { 'data': self.u[b, ... , ::self.k],
                            'mu': self.mu[b],
                            'f': f,
                            'times': self.times }
    """
    
    path = Path(__file__).resolve().parent.joinpath("data")
    data_path = str(path) + "/ns_random_forces_1.h5"
    dataset_resolution = 256
    train_subsample_rate = dataset_resolution // train_resolution
    test_subsample_rate = dataset_resolution // test_resolution
    assert not(dataset_resolution % train_resolution and dataset_resolution % test_resolution), f"train/test resolution must be a divisor of the dataset resolution {dataset_resolution}!"
    
    return load_autoregressive_traintestsplit(
    data_path,
    n_train, n_test,
    batch_size, test_batch_size, 
    train_subsample_rate, test_subsample_rate,
    time_step,
    predict_feature='u',
)
