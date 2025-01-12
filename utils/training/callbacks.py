"""
Callbacks store all non-essential logic
required to run specific training scripts. 

The callbacks in this module follow the form and 
logic of callbacks in Pytorch-Lightning (https://lightning.ai/docs/pytorch/stable)
"""

import sys
from typing import List, Union
from pathlib import Path

import torch
import wandb
from einops import repeat

from neuralop1.training.patching import MultigridPatching2D
from neuralop1.datasets.positional_encoding import get_grid_positional_encoding

class Callback(object):
    """
    Base callback class. Each abstract method is called in the trainer's
    training loop at the appropriate time. 

    Callbacks are stateful, meaning they keep track of a state and 
        update it throughout the lifetime of a Trainer class.
        Storing the state as a dict enables the Callback to keep track of
        references to underlying parts of the Trainer's process, such as 
        models, cost functions and output encoders
    """
    def __init__(self):
        self.state_dict = {}
    
    def _update_state_dict(self, **kwargs):
        self.state_dict.update(kwargs)

    def on_init_start(self, **kwargs):
        pass

    def on_init_end(self, *args, **kwargs):
        pass

    def on_before_train(self, *args, **kwargs):
        pass

    def on_train_start(self, *args, **kwargs):
        pass

    def on_epoch_start(self, *args, **kwargs):
        pass
    
    def on_batch_start(self, *args, **kwargs):
        pass

    def on_load_to_device(self, *args, **kwargs):
        pass
    
    def on_before_forward(self, *args, **kwargs):
        pass

    def on_before_loss(self, *args, **kwargs):
        pass
    
    def compute_training_loss(self, *args, **kwargs):
        raise NotImplementedError
    
    def on_batch_end(self, *args, **kwargs):
        pass
    
    def on_epoch_end(self, *args, **kwargs):
        pass
    
    def on_train_end(self, *args, **kwargs):
        pass

    def on_before_val(self, *args, **kwargs):
        pass

    def on_val_epoch_start(self, *args, **kwargs):
        pass
    
    def on_val_batch_start(self, *args, **kwargs):
        pass

    def on_before_val_loss(self, **kwargs):
        pass
    
    def compute_val_loss(self, *args, **kwargs):
        pass
    
    def on_val_batch_end(self, *args, **kwargs):
        pass

    def on_val_epoch_end(self, *args, **kwargs):
        pass
    
    def on_val_end(self, *args, **kwargs):
        pass


class PipelineCallback(Callback):
    
    def __init__(self, callbacks: List[Callback]):
        """
        PipelineCallback handles logic for the case in which
        a user passes more than one Callback to a trainer.

        Parameters
        ----------
        callbacks : List[Callback]
            list of Callbacks to use in Trainer
        """
        self.callbacks = callbacks

        overrides_device_load = ["on_load_to_device" in c.__class__.__dict__.keys() for c in callbacks]
        
        assert sum(overrides_device_load) < 2, "More than one callback cannot override device loading"
        if sum(overrides_device_load) == 1:
            self.device_load_callback_idx = overrides_device_load.index(True)
            print("using custom callback to load data to device.")
        else:
            self.device_load_callback_idx = None
            print("using standard method to load data to device.")

        # unless loss computation is overriden, call a basic loss function calculation
        overrides_loss = ["compute_training_loss" in c.__class__.__dict__.keys() for c in callbacks]

        if sum(overrides_loss) >= 1:
            self.overrides_loss = True
            print("using custom callback to compute loss.")
        else:
            self.overrides_loss = False
            print("using standard method to compute loss.")
        
    def _update_state_dict(self, **kwargs):
        for c in self.callbacks:
            c._update_state_dict(kwargs)

    def on_init_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_init_start(*args, **kwargs)

    def on_init_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_init_end(*args, **kwargs)

    def on_before_train(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_train(*args, **kwargs)

    def on_train_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_train_start(*args, **kwargs)

    def on_epoch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_start(*args, **kwargs)
    
    def on_batch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_start(*args, **kwargs)

    def on_load_to_device(self, *args, **kwargs):
        if self.device_load_callback_idx:
            self.callbacks[self.device_load_callback_idx].on_load_to_device(*args, *kwargs)
    
    def on_before_forward(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_forward(*args, **kwargs)

    def on_before_loss(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_loss(*args, **kwargs)
    
    def compute_training_loss(self, *args, **kwargs):
        if self.overrides_loss:
            for c in self.callbacks:
                c.compute_training_loss(*args, **kwargs)
        else:
            pass
    
    def on_batch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_end(*args, **kwargs)
    
    def on_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_end(*args, **kwargs)
    
    def on_train_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_train_end(*args, **kwargs)

    def on_before_val(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_val(*args, **kwargs)

    def on_val_epoch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_epoch_start(*args, **kwargs)
    
    def on_val_batch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_batch_start(*args, **kwargs)

    def on_before_val_loss(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_val_loss(*args, **kwargs)
    
    def compute_val_loss(self, *args, **kwargs):
        if self.overrides_loss:
            for c in self.callbacks:
                c.compute_val_loss(*args, **kwargs)
        else:
            pass
    
    def on_val_batch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_batch_end(*args, **kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_epoch_end(*args, **kwargs)
    
    def on_val_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_end(*args, **kwargs)

class SimpleWandBLoggerCallback(Callback):
    """
    Callback that implements simple logging functionality 
    expected when passing verbose to a Trainer
    """

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs:
            wandb.init(**kwargs)
    
    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_train_start(self, **kwargs):
        self._update_state_dict(**kwargs)

        train_loader = self.state_dict['train_loader']
        test_loaders = self.state_dict['test_loaders']
        verbose = self.state_dict['verbose']

        n_train = len(train_loader.dataset)
        self._update_state_dict(n_train=n_train)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        
    def on_epoch_start(self, epoch):
        self._update_state_dict(epoch=epoch)
    
    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)

    def on_before_loss(self, out, **kwargs):
        if self.state_dict['epoch'] == 0 and self.state_dict['idx'] == 0 \
            and self.state_dict['verbose']:
            print(f'Raw outputs of size {out.shape=}')
    
    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        # track training err and val losses to print at interval epochs
        msg = f'[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'
        values_to_log = dict(train_err=train_err / self.state_dict['n_train'], time=time, avg_loss=avg_loss)

        self._update_state_dict(msg=msg, values_to_log=values_to_log)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)
        
    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            self.state_dict['msg'] += f', {loss_name}={loss_value:.4f}'
            self.state_dict['values_to_log'][loss_name] = loss_value
    
    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get('regularizer', False):
            avg_lasso = self.state_dict.get('avg_lasso_loss', 0.)
            avg_lasso /= self.state_dict.get('n_epochs')
            self.state_dict['msg'] += f', avg_lasso={avg_lasso:.5f}'
        
        print(self.state_dict['msg'])
        sys.stdout.flush()

        if self.state_dict.get('wandb_log', False):
            for pg in self.state_dict['optimizer'].param_groups:
                lr = pg['lr']
                self.state_dict['values_to_log']['lr'] = lr
            wandb.log(self.state_dict['values_to_log'], step=self.state_dict['epoch'], commit=True)

class MGPatchingCallback(Callback):
    def __init__(self, levels: int, padding_fraction: float, stitching: float, encoder=None):
        """MGPatchingCallback implements multigrid patching functionality
        for datasets that require domain patching, stitching and/or padding.

        Parameters
        ----------
        levels : int
            mg_patching level parameter for MultigridPatching2D
        padding_fraction : float
            mg_padding_fraction parameter for MultigridPatching2D
        stitching : _type_
            mg_patching_stitching parameter for MultigridPatching2D
        encoder : neuralop.datasets.output_encoder.OutputEncoder, optional
            OutputEncoder to decode model outputs, by default None
        """
        super().__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.encoder = encoder
        
    def on_init_end(self, **kwargs):
        self._update_state_dict(**kwargs)
        self.patcher = MultigridPatching2D(model=self.state_dict['model'], levels=self.levels, 
                                      padding_fraction=self.padding_fraction,
                                      stitching=self.stitching)
    
    def on_batch_start(self, **kwargs):
        self._update_state_dict(**kwargs)
        self.state_dict['sample']['x'],self.state_dict['sample']['y'] =\
              self.patcher.patch(self.state_dict['sample']['x'],
                                 self.state_dict['sample']['y'],)
    
    def on_val_batch_start(self, *args, **kwargs):
        return self.on_batch_start(*args, **kwargs)
        
    def on_before_loss(self, out, **kwargs):
        
        evaluation = kwargs.get('eval', False)
        self._update_state_dict(out=out)
        self.state_dict['out'], self.state_dict['sample']['y'] = \
            self.patcher.unpatch(self.state_dict['out'],
                                 self.state_dict['sample']['y'],
                                 evaluation=evaluation)

        if self.encoder:
            self.state_dict['out'] = self.encoder.decode(self.state_dict['out'])
            self.state_dict['sample']['y'] = self.encoder.decode(self.state_dict['sample']['y'])
        
    
    def on_before_val_loss(self, **kwargs):
        return self.on_before_loss(**kwargs, evaluation=True)


class OutputEncoderCallback(Callback):
    
    def __init__(self, encoder):
        """
        Callback class for a training loop that involves
        an output normalizer but no MG patching.

        Parameters
        -----------
        encoder : neuralop.datasets.output_encoder.OutputEncoder
            module to normalize model inputs/outputs
        """
        super().__init__()
        self.encoder = encoder
    
    def on_batch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_before_loss(self, out):
        self.state_dict['out'] = self.encoder.decode(out)
        self.state_dict['sample']['y'] = self.encoder.decode(self.state_dict['sample']['y'])
    
    def on_before_val_loss(self, **kwargs):
        return self.on_before_loss(**kwargs)

class ModelCheckpointCallback(Callback):

    def __init__(self, checkpoint_dir: Union[Path, str] = Path('./checkpoints'), interval: int = 1):
        """
        Implements basic model checkpointing by saving a model every N epochs.
        
        Parameters
        ----------
        checkpoint_dir : str | pathlib.Path
            folder in which to save checkpoints
        interval : int
            interval at which to check metric
        """
        super().__init__()

        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_epoch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_epoch_end(self, *args, **kwargs):
        if self.state_dict['epoch'] % self.interval == 0:
            checkpoint_path = self.checkpoint_dir / f"ep_{self.state_dict['epoch']}.pt"
            torch.save(self.state_dict['model'].state_dict(), checkpoint_path)
        

class MonitorMetricCheckpointCallback(ModelCheckpointCallback):

    def __init__(self, loss_key: str, checkpoint_dir: str = './checkpoints'):
        """
        Implements model checkpointing with the addition of monitoring a chosen metric

        Parameters
        ----------
        monitor : str
            key name of validation metric to monitor
        checkpoint_path : str
            folder in which to save checkpoints
        """

        super().__init__()

        self.loss_key = loss_key
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
        self.checkpoint_dir = checkpoint_dir

    def on_train_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
        assert self.loss_key in self.state_dict['eval_losses'].keys(), \
            "Error: ModelCheckpointingCallback can only monitor metrics\
                tracked in eval_losses."

        self._update_state_dict(best_score=float('inf'))
    
    def on_val_epoch_end(self, errors):
        """
        save model if loss_key metric is lower than best
        """
        epoch = self.state_dict['epoch']
        if errors[self.loss_key] < self.state_dict['best_score']:
            model_save_path = f"{self.checkpoint_dir}/ep_{epoch}.pt"
            torch.save(self.state_dict['model'].state_dict(), model_save_path)
            print(f"Best value for {self.loss_key} found, saving to {model_save_path}")
        

class SimpleTensorBoardLoggerCallback(Callback):
    """
    Callback that implements simple logging functionality
    expected when passing verbose to a Trainer, using TensorBoard.
    """

    def __init__(self, log_dir='runs', **kwargs):
        super().__init__()
        import sys
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_train_start(self, **kwargs):
        self._update_state_dict(**kwargs)

        train_loader = self.state_dict['train_loader']
        test_loaders = self.state_dict['test_loaders']
        verbose = self.state_dict['verbose']

        n_train = len(train_loader.dataset)
        self._update_state_dict(n_train=n_train)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        
    def on_epoch_start(self, epoch):
        self._update_state_dict(epoch=epoch)
    
    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)

    def on_before_loss(self, out, **kwargs):
        if self.state_dict['epoch'] == 0 and self.state_dict['idx'] == 0 \
            and self.state_dict['verbose']:
            print(f'Raw outputs of size {out.shape=}')
    
    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        msg = f'[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'
        values_to_log = dict(train_err=train_err / self.state_dict['n_train'], time=time, avg_loss=avg_loss)

        self._update_state_dict(msg=msg, values_to_log=values_to_log)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)
        
    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            self.state_dict['msg'] += f', {loss_name}={loss_value:.4f}'
            self.state_dict['values_to_log'][loss_name] = loss_value
    
    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get('regularizer', False):
            avg_lasso = self.state_dict.get('avg_lasso_loss', 0.)
            avg_lasso /= self.state_dict.get('n_epochs')
            self.state_dict['msg'] += f', avg_lasso={avg_lasso:.5f}'
        
        print(self.state_dict['msg'])
        sys.stdout.flush()

        for key, value in self.state_dict['values_to_log'].items():
            self.writer.add_scalar(key, value, self.state_dict['epoch'])

        if self.state_dict.get('log_lr', False):
            for pg in self.state_dict['optimizer'].param_groups:
                lr = pg['lr']
                self.writer.add_scalar('learning_rate', lr, self.state_dict['epoch'])

    def __del__(self):
        self.writer.close()


class MultipleInputCallback(Callback):
    def __init__(self, append_positional_encoding=False, positional_encoding=None, grid_boundaries=[[0,1],[0,1]], channel_dim=1):
        super().__init__()
        # space dimension
        self.dimension = 2
        self.spacial_resolution = None
        self.dim_appenders = [] # [(name, broadcast function)]
        self.device = None
        self.append_positional_encoding = append_positional_encoding
        self.positional_encoding = positional_encoding
        self.grid_boundaries = grid_boundaries
        self.channel_dim = channel_dim

    def on_init_start(self, **kwargs):
        self.device = kwargs['device']
    
    def on_train_start(self, *args, **kwargs):
        # check and alter dim
        train_loader = kwargs['train_loader']
        for idx, batch in enumerate(train_loader):
            self.spacial_resolution  = batch['x'].shape[1:] # [b, (space)]
            self.dimension = len(self.spacial_resolution)
            # TODO: grids of 1D, 3D
            if self.positional_encoding is None and self.append_positional_encoding:
                assert self.dimension == 2, "Positional encoding currently only supported for 2D"
                grids = get_grid_positional_encoding(batch['x'], self.grid_boundaries, channel_dim=self.channel_dim)
                self.positional_encoding = torch.cat((grids[0], grids[1]), dim=1).to(self.device).permute(0,2,3,1).squeeze(0)

            for name in batch:
                if name == 'x' or name == 'y':
                    continue
                length = len(batch[name].shape)
                if length == self.dimension + 1: broadcast_function = lambda data: data.unsqueeze(-1)
                else: 
                    if self.dimension == 2: broadcast_function = lambda data: repeat(data, 'b -> b m n 1', m=self.spacial_resolution[0], n=self.spacial_resolution[1])
                    elif self.dimension == 1: broadcast_function = lambda data: repeat(data, 'b -> b m 1', m=self.spacial_resolution[0])
                    elif self.dimension == 3: broadcast_function = lambda data: repeat(data, 'b -> b m n p 1', m=self.spacial_resolution[0], n=self.spacial_resolution[1], p=self.spacial_resolution[2])
                    else: raise ValueError(f"Invalid dimension {self.dimension}")
                self.dim_appenders.append((name, broadcast_function))

            if self.append_positional_encoding:
                self.dim_appenders.append(('x', lambda data: repeat(self.positional_encoding, 'm n c -> b m n c', b=data.shape[0])))
            
            break
        
    def on_batch_start(self, idx, sample):
        sample = self.build_features(sample)
    
    def on_val_batch_start(self, idx, sample):
        sample = self.build_features(sample)

    def build_features(self, batch):
        x = batch['x']
        x = x.to(self.device)
        x.unsqueeze_(-1)
        for (name, appender) in self.dim_appenders:
            feature = appender(batch[name]).to(self.device)            
            x = torch.cat([x, feature], dim=-1)
        x = x.permute(0, 3 ,1, 2)

        batch['x'] = x
        # If the dataset only has one predictable target
        batch['y'].unsqueeze_(self.channel_dim)
        # return x

        
