from pathlib import Path
import torch
import scipy
from .tensor_dataset import TensorDataset
from ..utils import UnitGaussianNormalizer
from .transforms import PositionalEmbedding

def load_burgers(
    data_path, n_train, n_test, batch_train=32, batch_test=100, time=1, grid=[0, 1]
):

    data_path = Path(data_path).joinpath("burgers.pt").as_posix()
    data = torch.load(data_path)

    x_train = data[0:n_train, :, 0]
    x_test = data[n_train : (n_train + n_test), :, 0]

    y_train = data[0:n_train, :, time]
    y_test = data[n_train : (n_train + n_test), :, time]

    s = x_train.size(-1)

    if grid is not None:
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1, -1)

        grid_train = grid.repeat(n_train, 1)
        grid_test = grid.repeat(n_test, 1)

        x_train = torch.cat((x_train.unsqueeze(1), grid_train.unsqueeze(1)), 1)
        x_test = torch.cat((x_test.unsqueeze(1), grid_test.unsqueeze(1)), 1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_train,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=batch_test,
        shuffle=False,
    )

    return train_loader, test_loader

def load_burgers_mat(data_path,
    n_train,
    n_test,
    batch_size,
    test_batch_size,
    train_ssr=4,
    test_ssrs=[4],
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=False,
    encoding="channel-wise",
    channel_dim=1,
    ):
    try: 
        data0 = scipy.io.loadmat(data_path)
        del data0['__header__']
        del data0['__version__']
        del data0['__globals__']
        del data0['a_smooth']
        del data0['a_smooth_x']
        del data0['a_x']
    except:
        raise ValueError(f"Unknown dataset type")
    
    data = {}

    data['x'] = torch.from_numpy(data0['a'])
    data['y'] = torch.from_numpy(data0['u'])

    
    del data0

    initial_resolution = data["x"].shape[1]

    train_resolution = int(((initial_resolution - 1)/train_ssr) + 1)


    x_train = (
        data["x"][0:n_train, ::train_ssr][:,:train_resolution].unsqueeze(channel_dim).type(torch.float32).clone()
    )
    y_train = data["y"][0:n_train, ::train_ssr][:,:train_resolution].unsqueeze(channel_dim).clone()

    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None
    
    train_db = TensorDataset(
        x_train,
        y_train,
        transform_x=PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {}
    
    if type(test_ssrs) != type([]):
        test_ssrs = [test_ssrs]

    for ssr in test_ssrs:
        res = int(((initial_resolution - 1)/ssr) + 1)
        print(
            f"Loading test db at resolution {res} with {n_test} samples "
            f"and batch-size={test_batch_size}"
        )
        x_test = (
            data["x"][:n_test, ::ssr][:,:res].unsqueeze(channel_dim).type(torch.float32).clone()
        )
        y_test = data["y"][:n_test, ::ssr][:,:res].unsqueeze(channel_dim).clone()
        del data
        if input_encoder is not None:
            x_test = input_encoder.encode(x_test)

        test_db = TensorDataset(
            x_test,
            y_test,
            transform_x=PositionalEmbedding(grid_boundaries, 0)
            if positional_encoding
            else None,
        )
        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        test_loaders[res] = test_loader

    return train_loader, test_loaders, output_encoder
