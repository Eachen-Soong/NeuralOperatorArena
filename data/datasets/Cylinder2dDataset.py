import torch
from torch.utils.data import Dataset, DataLoader
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
        item['y'] = self.vel_times[b][t+self.time_step] # Size [config['nx'], config['ny'], 2]
        item['x'] = self.vel_times[b][t] # Size [config['nx'], config['ny'], 2]
        item['mask'] = self.masks[b]  # Size [config['nx'], config['ny']]

        return item
        
def load_cylinder2d_traintestsplit(
    train_data_dic='./train_data/', 
    test_data_dic='./test_data/',
    n_train=64, n_tests=16,
    train_batch_size=16, test_batch_size=4, 
    train_subsample_rate=1, test_subsample_rate=1,
    time_step=1,
):
    """
    Load and split the Cylinder2D dataset for training and testing.

    Args:
        train_data_dic (str): Path to the directory containing HDF5 files for training.
        test_data_dic (str): Path to the directory containing HDF5 files for testing.
        n_train (int): Number of training samples to use.
        n_tests (int): Number of testing samples to use.
        train_batch_size (int): Batch size for the training DataLoader.
        test_batch_size (int): Batch size for the testing DataLoader.
        train_subsample_rate (int): Subsampling rate for training data.
        test_subsample_rate (int): Subsampling rates for testing data.
        time_step (int): Number of time steps in the dataset.
        predict_feature (str): Feature to predict (e.g., 'u', 'v').

    Returns:
        tuple: (train_loader, test_loaders)
            train_loader: DataLoader for training dataset.
            test_loader: DataLoader for each test dataset.
    """

    # Load training data
    train_files = [os.path.join(train_data_dic, f) for f in os.listdir(train_data_dic) if f.endswith('.h5')]
    train_files = train_files[:n_train]
    with open(os.path.join(train_data_dic, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    train_masks, train_vel_times = [], []
    train_times = None
    for file in train_files:
        vel_time, times, mask = load_data(file)
        train_masks.append(torch.from_numpy(mask))
        train_vel_times.append(torch.from_numpy(vel_time))
        if train_times == None:
            train_times = torch.from_numpy(times)
        else:
            assert (torch.equal(train_times, torch.from_numpy(times))), "'times' not the same!"
        
    train_masks = torch.stack(train_masks, dim=0)[::train_subsample_rate]
    train_vel_times = torch.stack(train_vel_times, dim=0)[::train_subsample_rate]
    train_dataset = Cylinder2dDataset(config=config, masks=train_masks, times=train_times, vel_times=train_vel_times, time_step=time_step)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # Load test data
    test_files = [os.path.join(test_data_dic, f) for f in os.listdir(test_data_dic) if f.endswith('.h5')]
    test_files = test_files[:n_tests]

    test_masks, test_vel_times= [], []
    test_times = None
    for file in test_files:
        vel_time, times, mask = load_data(file)
        test_masks.append(torch.from_numpy(mask))
        test_vel_times.append(torch.from_numpy(vel_time))
        if test_times == None:
            test_times = torch.from_numpy(times)

    test_masks = torch.stack(test_masks, dim=0)[::test_subsample_rate]
    test_vel_times = torch.stack(test_vel_times, dim=0)[::test_subsample_rate]
    test_dataset = Cylinder2dDataset(config=config, masks=test_masks, times=test_times, vel_times=test_vel_times, time_step=time_step)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    # Load training and testing data
    train_loader, test_loader = load_cylinder2d_traintestsplit(
        train_data_dic='/data/jmwang/DimOL/2D_cylinders/train_data/', 
        test_data_dic='/data/jmwang/DimOL/2D_cylinders/test_data/'
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
    