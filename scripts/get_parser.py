import torch
import argparse
from typing import Optional, List
import matplotlib.pyplot as plt

"""
    To support other models, define the correspondent parser class as follows!
    Example Usage: train_burgers.py FNO --channel_mixing mlp
"""

class BaseModelParser():
    def __init__(self) -> None:
        self.name = 'Model'
    
    def add_parser_args(self, parser:argparse.ArgumentParser):
        pass

    def get_model(self, args):
        return torch.nn.Identity()

class BaseDataParser():
    def __init__(self) -> None:
        self.name = 'Data'
    
    def add_parser_args(self, parser:argparse.ArgumentParser):
        pass

    def get_data(self, args):
        raise NotImplementedError
    

def add_train_args(parser:argparse.ArgumentParser):
    # # # Data Loader Configs # # #
    parser.add_argument('--batch_size', type=int, default=32) #
    # # # Model Configs # # #
    parser.add_argument('--load_path', type=str, default='', help='load checkpoint')
    # # # Optimizer Configs # # #
    parser.add_argument('--lr', type=float, default=1e-3) #
    parser.add_argument('--weight_decay', type=float, default=1e-4) #
    parser.add_argument('--scheduler_steps', type=int, default=100) #
    parser.add_argument('--scheduler_gamma', type=float, default=0.5) #
    parser.add_argument('--train_loss', type=str, default='h1', help='h1 or l2 or l1') #
    parser.add_argument('--eval_loss', type=str, nargs='+', default=['h1', 'l2'], help='h1 or l2 or l1') #
    parser.add_argument('--loss_reduction', type=str, default='sum', help='sum or mean') #

    # # # Log and Save Configs # # #
    parser.add_argument('--save_dir', type=str, default='./runs')
    parser.add_argument('--version_of_time', type=int, default=1, help='whether to use program start time as suffix')
    parser.add_argument('--monitor', type=str, default='val_err_l2', help='train_test splits')
    # # # Trainer Configs # # #
    parser.add_argument('--epochs', type=int, default=501) #
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=0, help='whether to use random seed')
    parser.add_argument('--seed', type=int, default=0)
    return


def add_eval_args(parser:argparse.ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=32) #
    # # # Model Configs # # #
    parser.add_argument('--load_path', type=str, default='', help='load checkpoint')
    # # # Optimizer Configs # # #
    parser.add_argument('--eval_loss', type=str, nargs='+', default=['h1', 'l2'], help='h1 or l2 or l1') #
    parser.add_argument('--loss_reduction', type=str, default='sum', help='sum or mean') #

    # # # Log and Save Configs # # #
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--log_input', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--visualize_example', type=int, default=0, help='Whether to visualize the output')
    return


class Fetcher():
    def __init__(self, DataParsers:Optional[List[BaseDataParser]]=None, ModelParsers:Optional[List[BaseModelParser]]=None, mode='train') -> None:
        self.DataParsers = DataParsers
        self.ModelParsers = ModelParsers
        if self.DataParsers == None:
            self.DataParsers = BaseDataParser.__subclasses__()
        if self.ModelParsers == None:
            self.ModelParsers = BaseModelParser.__subclasses__()
        self.data_fetcher = {}
        self.model_fetcher = {}
        self.mode = mode

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser('NeuralOps', add_help=False)
        subparsers = parser.add_subparsers(dest='data', help='Choose a dataset')

        # Add data parsers
        for data_parser_class in self.DataParsers:
            data_parser_instance = data_parser_class()
            data_name = data_parser_instance.name
            subparser = subparsers.add_parser(data_name, help=f'Use {data_name}')
            self.data_fetcher[data_name] = data_parser_class
            
            # Add model subparsers for each data subcommand
            model_subparsers = subparser.add_subparsers(dest='model', help='Choose a model')
            for model_parser_class in self.ModelParsers:
                model_parser_instance = model_parser_class()
                model_name = model_parser_instance.name
                subsubparser = model_subparsers.add_parser(model_name, help=f'Train {model_name} on {data_name}')

                if self.mode == 'train':
                    add_train_args(subsubparser)
                elif self.mode == 'test' or self.mode == 'predict':
                    add_eval_args(subsubparser)
                data_parser_instance.add_parser_args(subsubparser)
                model_parser_instance.add_parser_args(subsubparser)
                self.model_fetcher[model_name] = model_parser_class

        return parser.parse_args(args=args)

    def get_data(self, args):
        data_name = args.data
        return self.data_fetcher[data_name]().get_data(args)

    def get_model(self, args)->torch.nn.Module:
        model_name = args.model
        return self.model_fetcher[model_name]().get_model(args)
    
    def get_visualizing_function(self, args):
        data_name = args.data
        return self.data_fetcher[data_name].visualize


if __name__ == "__main__":
    fetcher = Fetcher()
    args = fetcher.parse_args()
    
    data = fetcher.get_data(args)
    model = fetcher.get_model(args)
    
    print(args)
    print(f"Data: {data}")
    print(f"Model: {model}")

