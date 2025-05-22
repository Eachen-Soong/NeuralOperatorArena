from .get_parser import BaseDataParser
from data.datasets import load_burgers_mat, load_darcy_mat, load_autoregressive_traintestsplit_dim, load_autoregressive_traintestsplit_v3, load_autoregressive_multitask_mu_preordered, load_autoregressive_traintestsplit_v1, load_cylinder2d_traintestsplit, load_cylinder2d
from data.datasets.dataloader import ns_contextual_loader
import torch
import numpy as np
import types
import copy
from typing import Union


class BurgersParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Burgers'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_train', type=int, default=-1)
        parser.add_argument('--n_test', type=int, default=-1)
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, nargs="+", default=1)
        return
    
    def get_data(self, args):
        train_loader, val_loader, _ = load_burgers_mat(
        data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
        train_ssr=args.train_subsample_rate, test_ssrs=args.test_subsample_rate, 
        positional_encoding=args.pos_encoding
        )
        return train_loader, val_loader


class DarcyParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Darcy'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_train', type=int, default=-1)
        parser.add_argument('--n_test', type=int, default=-1)
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, nargs="+", default=1)
        return
    
    def get_data(self, args):
        train_loader, val_loader, _ = load_darcy_mat(
            data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_ssr=args.train_subsample_rate, test_ssrs=args.test_subsample_rate, 
            positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        return train_loader, val_loader


class TorusLiParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'TorusLi'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_train', type=int, default=-1)
        parser.add_argument('--n_test', type=int, default=-1)
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, default=1)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, default='u')
        return
    
    def get_data(self, args):
        train_loader, val_loader = load_autoregressive_traintestsplit_v1(
            data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_subsample_rate=args.train_subsample_rate, test_subsample_rate=args.test_subsample_rate, time_step=args.time_step,
            predict_feature=args.predict_feature, append_positional_encoding=args.pos_encoding,
        )
        return train_loader, val_loader
    
def to_torch_tensor(data: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(data, list):
        return torch.tensor(np.array(data))
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError("Input type must be list, np.array or torch.tensor")

def gen_similar_dataloaders_dt_divided_p(origin_loader, scaling_ps, batch_size=0):

    def new_get_item(self, index):
        # returns: {'x', 'y', other features}
        p = self.scaling_p
        p_int = int(p)
        origin_item =  self.get_item(index, p_int)
        new_item = {}
        new_item['x'] = origin_item['x'] * p
        new_item['x'][:, 1, ...] = origin_item['x'][:, 1, ...] * p # forcing
        new_item['y'] = origin_item['y'] * p
        new_item['consts'] = origin_item['consts'] * p
        return new_item

    sim_loaders = {}
    n_scale_coeff = to_torch_tensor(scaling_ps).shape[0]
    # shallow copy to share the same raw data
    sim_dataset = copy.copy(origin_loader.dataset)
    sim_dataset.__getitem__ = types.MethodType(new_get_item, sim_dataset)
    sim_dataset.scaling_p = 1.
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


class TorusVisForceDimParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'TorusVisForceDim'
        self.var_field_pred_names = []
        self.var_field_nonpred_names = []
        self.const_field_names = []
        self.constants_names = []

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_train', type=int, default=-1)
        parser.add_argument('--n_test', type=int, default=-1)
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, nargs="+", default=1)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--time_skips', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, nargs='+', default=['u'])
        parser.add_argument('--simaug_coeff', type=int, nargs='+', default=0)
        return
    
    def get_data(self, args):
        seperate_consts = args.norm == "dim_norm"
        train_loader, val_loader = load_autoregressive_traintestsplit_dim(
            data_path=args.data_path, n_train=args.n_train, n_tests=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_subsample_rate=args.train_subsample_rate, test_subsample_rates=args.test_subsample_rate, time_step=args.time_step, time_skips=args.time_skips,
            predict_features=args.predict_feature, seperate_consts=seperate_consts
        )
        
        # if args.simaug_coeff != 0:
        #     sim_loaders = gen_similar_dataloaders_dt_divided_p(train_loader, scaling_ps=args.simaug_coeff, batch_size=train_loader.batch_size)
        #     val_loader.update(sim_loaders)
        # else: print("No simaug")

        return train_loader, val_loader

    def get_dim_aligner(self, args):
        """
        The dataset consists of u(vorticity), mu(viscosity), f(external force).
        By didmensional analysis, here we consider 3 dimensionless number:
        1; Reynolds = rho v d / mu ~~ v / mu; Froude = v / sqrt(f L) ~~ v / sqrt(f)
        """
        n_dim = args.n_dim
        normalization_dims = list(range(2, 2+n_dim))
        # def get_normalizer(x, consts, **kwargs):
        #     mu = torch.mean(x, dim=normalization_dims)
        #     mean = torch.sqrt(torch.mean(torch.square(x), dim=normalization_dims))
        #     std = torch.sqrt(mean**2 - mu**2)
        #     Re_std = std[:, 0:1] / consts[:, 0:]
        #     Fr_std = std[:, 0:1] / mean[:, 1:]
        #     ones = torch.ones_like(Re_std)
        #     Re_mu = mu[:, 0:1] / consts[:, 0:]
        #     Fr_mu = mu[:, 0:1] / mean[:, 1:]
        #     return mu, std, torch.cat([ones, Re_mu, Fr_mu], dim=1), torch.cat([ones, Re_std, Fr_std], dim=1)
        def get_normalizer(x, consts, **kwargs):
            mu = torch.mean(x, dim=normalization_dims)
            mean = torch.sqrt(torch.mean(torch.square(x), dim=normalization_dims))
            std = torch.sqrt(mean**2 - mu**2)
            St_inv_std = 1 / std[:, 0:1]
            Re_St_inv_std = consts[:, 0:] / std[:, 0:1]**2
            Fr_sq_St_inv_std = std[:, 1:] / std[:, 0:1]**3

            ones = torch.ones_like(Re_St_inv_std)
            # Re_mu = mu[:, 0:1] / consts[:, 0:]
            # Fr_mu = mu[:, 0:1] / mean[:, 1:]
            return mu, std, torch.cat([ones, ones, ones, ones], dim=1), torch.cat([ones, St_inv_std, Re_St_inv_std, Fr_sq_St_inv_std], dim=1)
        return get_normalizer

class TorusVisForceParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'TorusVisForce'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_train', type=int, default=-1)
        parser.add_argument('--n_test', type=int, default=-1)
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, nargs="+", default=1)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, default='u')
        parser.add_argument('--simaug_coeff', type=int, nargs='+', default=0)
        return
    
    def get_data(self, args):
        train_loader, val_loader = load_autoregressive_traintestsplit_v3(
            data_path=args.data_path, n_train=args.n_train, n_tests=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_subsample_rate=args.train_subsample_rate, test_subsample_rates=args.test_subsample_rate, time_step=args.time_step,
            append_positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        if args.simaug_coeff != 0:
            sim_loaders = gen_similar_dataloaders_dt_divided_p(train_loader, scaling_ps=args.simaug_coeff, batch_size=train_loader.batch_size)
            val_loader.update(sim_loaders)
        else: print("No simaug")

        return train_loader, val_loader


class MultiTaskTorusVisForceParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'MultiTaskTorusVisForce'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_data', type=int, default=-1)
        parser.add_argument('--splits', nargs='+', default=[(4, 1), (4, 1), (4, 1), (4, 1)], help='train_test splits')
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, nargs="+", default=1)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, default='u')
        return
    
    def get_data(self, args):
        train_loaders, val_loaders = load_autoregressive_multitask_mu_preordered(
            data_path=args.data_path, n_data=args.n_data, splits=args.splits, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_subsample_rate=args.train_subsample_rate, test_subsample_rate=args.test_subsample_rate, time_step=args.time_step,
            append_positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        
        return train_loaders, val_loaders


class MultiTaskTorusVisForceParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'MultiTaskTorusvisForce'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--data_path', type=str, default='', help="the path of data file")
        parser.add_argument('--n_data', type=int, default=-1)
        parser.add_argument('--splits', nargs='+', default=[(4, 1), (4, 1), (4, 1), (4, 1)], help='train_test splits')
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate', type=int, nargs="+", default=1)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, default='u')
        return
    
    def get_data(self, args):
        train_loaders, val_loaders = load_autoregressive_traintestsplit_v3(
            data_path=args.data_path, n_data=args.n_data, splits=args.splits, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_subsample_rate=args.train_subsample_rate, test_subsample_rate=args.test_subsample_rate, time_step=args.time_step,
            append_positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        
        return train_loaders, val_loaders
    

class MultiTaskCylinderFlowParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'MultiTaskCylinderFlow'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--train_path', type=str, nargs='+', default='./', help="the path of training data file")
        parser.add_argument('--test_path',  type=str, nargs='+', default='./', help="the path of test data file")
        parser.add_argument('--n_train', type=int, nargs='+', default=64)
        parser.add_argument('--n_test', type=int, nargs='+', default=16)
        parser.add_argument('--train_subsample_rate', type=int, default=1)
        parser.add_argument('--test_subsample_rate',  type=int, default=1)
        parser.add_argument('--time_step',  type=int, default=1, help='subsample rate of time')
        return
    
    def get_data(self, args):
        train_path = args.train_path
        test_path = args.test_path
        n_train = args.n_train
        n_test = args.n_test
        if len(n_train) == 1:
            n_train = n_train * len(train_path)
        if len(n_test) == 1:
            n_test = n_test * len(test_path)
        
        assert len(n_train) == len(train_path) and len(n_test) == len(test_path), f'Please check the number of train_path, n_train, test_path, n_test: {len(train_path) , len(n_train) , len(test_path) , len(n_test)}'
        
        train_loaders = []; val_loaders = []
        for i in range(len(n_train)):
            loader = load_cylinder2d(
                data_dir=train_path[i], n_data=n_train[i], batch_size=args.batch_size, 
                subsample_rate=args.train_subsample_rate, time_step=args.time_step, shuffle=True
            )
            train_loaders.append(loader)
        for i in range(len(n_test)):
            loader = load_cylinder2d(
                data_dir=test_path[i], n_data=n_test[i], batch_size=args.batch_size, 
                subsample_rate=args.test_subsample_rate, time_step=args.time_step, shuffle=False
            )
            val_loaders.append(loader)
            
        return train_loaders, val_loaders