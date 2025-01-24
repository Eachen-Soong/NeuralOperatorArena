import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import default_collate # it's a function
from einops import repeat
from .positional_encoding import get_grid_positional_encoding
import h5py, os, json

def load_data(filename):
    """
    Load velocity fields and time step data from an HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        vel_time = f["velocities"][:]
        times = f["times"][:]
        mask = f["mask"][:]
    return vel_time, times, mask

class Cylinder2dDataset(Dataset):
    def __init__(self, config: dict, masks, times, vel_times, time_step=1):
        '''
        Initialize the Cylinder2D dataset.

        Args:
            config (dict): Configuration parameters for LBM simulation.
            masks (torch.Tensor): List of independent variables, each element is a 2D array of size config['nx'] * config['ny'].
            time_steps (torch.Tensor): List of time points.
            velocity_fields (torch.Tensor): List of dependent variables, each element is a time-series velocity field.
            time_step (int): The time step for autoregressive prediction, not the real timestep of simulation.
        '''
        assert masks.size(0) == vel_times.size(0), "Size mismatch between variables"
        self.config = config
        self.masks = masks  # List of independent variables
        self.times = times  # List of time points
        self.vel_times = vel_times  # List of dependent variables (time-series velocity fields)
        self.time_step = time_step
        self.n_ticks = times.size(0)-self.time_step

    def __len__(self):
        '''Returns the size of the dataset (number of samples).'''
        return self.masks.size(0) * self.n_ticks

    def __getitem__(self, index):
        '''
        Returns the index-th item.
        Args:
            - index (int): Index of the item.
        Returns:
            dict: A dictionary containing:
                - 'x' (torch.Tensor): Input velocity field at time t, size [2, config['nx'], config['ny']].
                - 'y' (torch.Tensor): Target velocity field at time t + time_step, size [2, config['nx'], config['ny']].
                - 'mask' (torch.Tensor): Mask for the sample, size [config['nx'], config['ny']].
        '''
        b = index // self.n_ticks # sample number
        t = index % self.n_ticks # time in a sample
        item = {}
        item['y'] = self.vel_times[b][t+self.time_step].permute(2, 0, 1) # Size [2, config['nx'], config['ny']]
        x = self.vel_times[b][t].permute(2, 0, 1) # Size [2, config['nx'], config['ny']]
        x = torch.concat([x, self.masks[b].unsqueeze(0)])
        item['x'] = x
        return item
    

def load_cylinder2d(
    data_dir='./',
    n_data=64,
    batch_size=16, 
    subsample_rate=1, 
    time_step=1,
    shuffle=False,
):
    """
    Load the Cylinder2D dataset.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        n_data (int): Number of samples to use.
        batch_size (int): Batch size for the  DataLoader.
        subsample_rate (int): Subsampling rate for data.
        time_step (int): Number of time steps in the dataset.
        shuffle (bool): Whether to shuffle the dataset

    Returns:
        loader: DataLoader for the dataset.
    """
    # Load data
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
    data_files = data_files[:n_data]
    with open(os.path.join(data_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    data_masks, data_vel_times = [], []
    data_times = None
    for file in data_files:
        vel_time, times, mask = load_data(file)
        data_masks.append(torch.from_numpy(mask))
        data_vel_times.append(torch.from_numpy(vel_time))
        if data_times == None:
            data_times = torch.from_numpy(times)
        else:
            assert (torch.equal(data_times, torch.from_numpy(times))), "'times' not the same!"
        
    data_masks = torch.stack(data_masks, dim=0)[::
    subsample_rate]
    data_vel_times = torch.stack(data_vel_times, dim=0)[::
    subsample_rate]

    dataset = Cylinder2dDataset(config=config, masks=data_masks, times=data_times, vel_times=data_vel_times, time_step=time_step)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def load_cylinder2d_traintestsplit(
    data_dir='./',
    # train_data_dir='./train_data/', 
    # test_data_dir='./test_data/',
    n_train=64, n_test=16,
    train_batch_size=16, test_batch_size=4, 
    train_subsample_rate=1, test_subsample_rate=1,
    time_step=1,
):
    """
    Load and split the Cylinder2D dataset for training and testing.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        n_train (int): Number of training samples to use.
        n_test (int): Number of testing samples to use.
        train_batch_size (int): Batch size for the training DataLoader.
        test_batch_size (int): Batch size for the testing DataLoader.
        train_subsample_rate (int): Subsampling rate for training data.
        test_subsample_rate (int): Subsampling rates for testing data.
        time_step (int): Number of time steps in the dataset.

    Returns:
        tuple: (train_loader, test_loaders)
            train_loader: DataLoader for training dataset.
            test_loader: DataLoader for each test dataset.
    """
    train_data_dir = os.path.join(data_dir, 'train_data/')
    test_data_dir = os.path.join(data_dir, 'test_data/')
    train_loader = load_cylinder2d(data_dir=train_data_dir, n_data=n_train, batch_size=train_batch_size, 
                                   subsample_rate=train_subsample_rate, time_step=time_step, shuffle=True)
    test_loader = load_cylinder2d(data_dir=test_data_dir, n_data=n_test, batch_size=test_batch_size, 
                                   subsample_rate=test_subsample_rate, time_step=time_step, shuffle=False)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Load training and testing data
    train_loader, test_loader = load_cylinder2d_traintestsplit(
        data_dir='/data/jmwang/DimOL/2D_cylinders/', 
    )

    # Print the size of the training and testing datasets
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")

    # Get a training batch and check its shape
    train_batch = next(iter(train_loader))
    print("\nTraining batch:")
    print(f"Input 'x' shape: {train_batch['x'].shape}")  # Should be [batch_size, nx, ny， 2]
    print(f"Target 'y' shape: {train_batch['y'].shape}")  # Should be [batch_size, nx, ny， 2]
    print(f"Mask shape: {train_batch['mask'].shape}")     # Should be [batch_size, nx, ny]

    # Get a testing batch and check its shape
    test_batch = next(iter(test_loader))
    print("\nTesting batch:")
    print(f"Input 'x' shape: {test_batch['x'].shape}")    # Should be [batch_size, nx, ny，2]
    print(f"Target 'y' shape: {test_batch['y'].shape}")   # Should be [batch_size, nx, ny， 2]
    print(f"Mask shape: {test_batch['mask'].shape}")      # Should be [batch_size, nx, ny]
    