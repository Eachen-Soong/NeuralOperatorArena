import torch
from torch.cuda import amp
from timeit import default_timer
import sys 
import wandb
import pathlib

import neuralop1.mpu.comm as comm

from .losses import LpLoss
from .callbacks import PipelineCallback

from typing import Union, Dict

import time

class Trainer:
    def __init__(self, *, 
                 model, 
                 n_epochs, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False, 
                 callbacks = None,
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 checkpoint_to_load: pathlib.Path=None,
                 verbose=True):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        """

        if callbacks:
            assert type(callbacks) == list, "Callbacks must be a list of Callback objects"
            self.callbacks = PipelineCallback(callbacks=callbacks)
            self.override_load_to_device = (self.callbacks.device_load_callback_idx is not None)
            self.overrides_loss = self.callbacks.overrides_loss
        else:
            self.callbacks = []
            self.override_load_to_device = False
            self.overrides_loss = False
        
        if verbose:
            print(f"{self.override_load_to_device=}")
            print(f"{self.overrides_loss=}")

        if self.callbacks:
            self.callbacks.on_init_start(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)

        if checkpoint_to_load:
            self.model.load_state_dict(torch.load(checkpoint_to_load))

        self.model = model
        self.n_epochs = n_epochs

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast

        if self.callbacks:
            self.callbacks.on_init_end(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)
        
        
    def train(self, 
              train_loader:Union[Dict[str, torch.utils.data.DataLoader], torch.utils.data.DataLoader], 
              test_loaders, optimizer, scheduler, regularizer,
              training_loss=None, eval_losses=None):
        
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader or dict[torch.utils.data.DataLoader]
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: function to use 
        """
        if isinstance(train_loader, torch.utils.data.DataLoader):
            # This is the case where train_loaders is a single DataLoader instance.
            train_loader_list = {'default': train_loader}
        elif isinstance(train_loader, dict):
            # This is the case where train_loaders is a dictionary of DataLoader instances.
            train_loader_list = train_loader
        else:
            raise ValueError("train_loaders must be either a DataLoader or a dictionary of DataLoader instances.")
        
        n_loader = len(train_loader_list.keys())

        if self.callbacks:
            self.callbacks.on_train_start(train_loader=train_loader, test_loaders=test_loaders,
                                    optimizer=optimizer, scheduler=scheduler, 
                                    regularizer=regularizer, training_loss=training_loss, 
                                    eval_losses=eval_losses)

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 
        
        for epoch in range(self.n_epochs):
            
            train_err = 0.0
            avg_loss = 0
            avg_lasso_loss = 0
            for loader_name, train_loader in train_loader_list.items():
                if self.callbacks:
                    self.callbacks.on_epoch_start(epoch=epoch)

                
                self.model.train()
                t1 = default_timer()
                

                for idx, sample in enumerate(train_loader):
                    if self.callbacks:
                        self.callbacks.on_batch_start(idx=idx, sample=sample)

                    # Decide what to do about logging later when we decide on batch naming conventions
                    '''if epoch == 0 and idx == 0 and self.verbose and is_logger:
                        print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')'''

                    # y = sample['y']

                    # load everything from the batch onto self.device if 
                    # no callback overrides default load to device
                    
                    if self.override_load_to_device:
                        self.callbacks.on_load_to_device(sample=sample)
                    else:
                        for k,v in sample.items():
                            if hasattr(v, 'to'):
                                sample[k] = v.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    if regularizer:
                        regularizer.reset()

                    if self.amp_autocast:
                        with amp.autocast(enabled=True):
                            out = self.model(**sample)
                    else:
                        out = self.model(**sample)

                    if self.callbacks:
                        self.callbacks.on_before_loss(out=out)

                    loss = 0.

                    if self.overrides_loss:
                        if isinstance(out, torch.Tensor):
                                loss += self.callbacks.compute_training_loss(out=out.float(), **sample, amp_autocast=self.amp_autocast)
                        elif isinstance(out, dict):
                            loss += self.callbacks.compute_training_loss(**out, **sample, amp_autocast=self.amp_autocast)
                    else:
                        if self.amp_autocast:
                            with amp.autocast(enabled=True):
                                if isinstance(out, torch.Tensor):
                                    loss = training_loss(out.float(), **sample)
                                elif isinstance(out, dict):
                                    loss += training_loss(**out, **sample)
                        else:
                            if isinstance(out, torch.Tensor):
                                loss = training_loss(out.float(), **sample)
                            elif isinstance(out, dict):
                                loss += training_loss(**out, **sample)
                    del out

                    if regularizer:
                        loss += regularizer.loss

                    loss.backward()
                    
                    optimizer.step()

                train_err += loss.item()
            
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

                if self.callbacks:
                    self.callbacks.on_batch_end()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1

            train_err /= len(train_loader.dataset)
            train_err /= n_loader
            avg_loss  /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 

                if self.callbacks:
                    self.callbacks.on_before_val(epoch=epoch, train_err=train_err, time=epoch_train_time, \
                                           avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss)
                
                for loader_name, loader in test_loaders.items():
                    _ = self.evaluate(eval_losses, loader, log_prefix=loader_name)

                if self.callbacks:
                    self.callbacks.on_val_end()
            
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch=epoch, train_err=train_err, avg_loss=avg_loss)

    def predict_step(self, sample, idx=0):
        """Predicts one step with the model
        
        Parameters
        ----------
        data_loader : data_loader to evaluate on

        Returns
        -------
        y_hat : the output of the model
        """

        self.model.eval()

        with torch.no_grad():
            
            if self.callbacks:
                self.callbacks.on_val_batch_start(idx=idx, sample=sample)
            
            # load everything from the batch onto self.device if 
            # no callback overrides default load to device
            
            if self.override_load_to_device:
                self.callbacks.on_load_to_device(sample=sample)
            else:
                for k,v in sample.items():
                    if hasattr(v, 'to'):
                        sample[k] = v.to(self.device)
            
            out = self.model(**sample)

        return out

    def evaluate(self, loss_dict, data_loader,
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        if self.callbacks:
            self.callbacks.on_val_epoch_start(loss_dict = loss_dict, data_loader=data_loader)

        self.model.eval()

        errors = {f'{log_prefix}_{loss_name}': 0. for loss_name in loss_dict.keys()}

        n_samples = 0
        eval_start_time = time.time()
        evaluation_time = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                batch_start_time = time.time()
                y = sample['y']
                n_samples += y.size(0)

                out = self.predict_step(idx=idx, sample=sample)

                if self.callbacks:
                    self.callbacks.on_before_val_loss(out=out)
                
                for loss_name, loss in loss_dict.items():
                    val_loss = 0.
                    if self.overrides_loss:
                        if isinstance(out, torch.Tensor):
                            val_loss = self.callbacks.compute_training_loss(out, **sample)
                        elif isinstance(out, dict):
                            val_loss = self.callbacks.compute_training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            val_loss = loss(out, **sample).item()
                        elif isinstance(out, dict):
                            val_loss = loss(**out, **sample).item()

                    errors[f'{log_prefix}_{loss_name}'] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
                evaluation_time += time.time() - batch_start_time
        
        preprocess_time = time.time() - eval_start_time - evaluation_time

        del y, out

        for key in errors.keys():
            errors[key] /= n_samples
        
        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors)
            
        # if self.verbose: print("test_set: ", log_prefix, preprocess_time, errors)
        return errors