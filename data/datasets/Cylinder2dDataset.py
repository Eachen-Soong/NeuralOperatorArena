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
    def __init__(self, config: dict, masks, times, vel_times):
        '''
        Args:
            - config (dict): Configuration parameters for LBM simulation.
            - masks: List of independent variables, each element is a 2D array of size config['nx'] * config['ny'].
            - vel_times: List of dependent variables, each element is a time-series velocity field.
            - times: List of time points.
        '''
        assert masks.size(0) == vel_times.size(0), "Size mismatch between variables"
        self.config = config
        self.masks = masks  # List of independent variables
        self.times = times  # List of time points
        self.vel_times = vel_times  # List of dependent variables (time-series velocity fields)

    def __len__(self):
        '''Returns the size of the dataset (number of samples).'''
        return self.masks.size(0)

    def __getitem__(self, index):
        '''
        Returns the index-th sample.
        Args:
            - index (int): Index of the sample.
        Returns:
            - mask (torch.Tensor): Independent variable, size [config['nx'], config['ny']].
            - vel_time (torch.Tensor): Dependent variable, size [T, 2, config['nx'], config['ny']], where T is the number of time steps.
            - time (torch.Tensor): List of time points, size [T].
        '''
        # Get the index-th independent variable (mask)
        mask = self.masks[index]  # Size [config['nx'], config['ny']]

        # Get the index-th dependent variable (time-series velocity field)
        vel_time = self.vel_times[index]  # Size [T, 2, config['nx'], config['ny']]

        # Return the sample
        return {
            'mask': mask,  # Independent variable
            'velocity': vel_time,  # Dependent variable (time-series velocity field)
            'time': self.times  # List of time points
        }
        
def load_cylinder2d_traintestsplit(
    train_data_dic='./train_data/', 
    test_data_dic='./test_data/',
    n_train=64, n_tests=16,
    train_batch_size=16, test_batch_size=4, 
    train_subsample_rate=1, test_subsample_rate=1,
    time_step=1,
    predict_feature='u',
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

    train_masks, train_vel_times, train_times = [], [], []
    for file in train_files:
        vel_time, times, mask = load_data(file)
        train_masks.append(torch.from_numpy(mask))
        train_vel_times.append(torch.from_numpy(vel_time))
        train_times.append(torch.from_numpy(times))


    train_masks = torch.stack(train_masks, dim=0)[::train_subsample_rate]
    train_vel_times = torch.stack(train_vel_times, dim=0)[::train_subsample_rate]
    train_times = train_times[0]  # Same for all files
    train_dataset = Cylinder2dDataset(config=config, masks=train_masks, times=train_times, vel_times=train_vel_times)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # Load test data
    test_files = [os.path.join(test_data_dic, f) for f in os.listdir(test_data_dic) if f.endswith('.h5')]
    test_files = test_files[:n_tests]

    test_masks, test_vel_times, test_times = [], [], []
    for file in test_files:
        vel_time, times, mask = load_data(file)
        test_masks.append(torch.from_numpy(mask))
        test_vel_times.append(torch.from_numpy(vel_time))
        test_times.append(torch.from_numpy(times))

    test_masks = torch.stack(test_masks, dim=0)[::test_subsample_rate]
    test_vel_times = torch.stack(test_vel_times, dim=0)[::test_subsample_rate]
    test_times = test_times[0]  # Same for all files

    test_dataset = Cylinder2dDataset(config=config, masks=test_masks, times=test_times, vel_times=test_vel_times)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader =load_cylinder2d_traintestsplit('./train_data/', './test_data/')