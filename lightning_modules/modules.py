from typing import Any
import lightning as L
import torch

class MultiMetricModule(L.LightningModule):
    """
    Basic Pytorch-Lightning module for AI for Science Tasks.
    Defines the train/val/test/predict steps, the major features are:
        1. The prediction target for each batch is batch['y']
        2. The 
    TODO: add single metric
    """
    def __init__(self, model, optimizer, scheduler, train_loss, metric_dict:dict) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = train_loss
        self.metric_dict = metric_dict

    def configure_optimizers(self):
        return {'optimizer':self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.train_loss(self.model(**batch), batch['y'])
        self.log('train_err', loss)
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss_dict[key] = self.metric_dict[key](pred, batch['y'])
        self.log_dict(loss_dict)
    
    def test_step(self, batch, batch_idx, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss_dict[key] = self.metric_dict[key](pred, batch['y'])
        self.log_dict(loss_dict)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        pred = self.model(**batch)
        self.log('predict_value', pred)
        return pred
    

class MultiTaskModule(L.LightningModule):
    """
    Multi-Task Learning Pytorch-Lightning module for AI for Science Tasks.
    Defines the train/val/test/predict steps, the major features are:
        1. The prediction target for each batch is batch['y']

    Must be cocupled with the MultiTask DataModule!
    TODO: add single metric
    """
    def __init__(self, model, optimizer, scheduler, train_loss, metric_dict:dict, n_tasks, n_val_tasks=-1, train_data_names:list=None, val_data_names:list=None, log_on_epoch=True) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.optimizer = optimizer
        if type(train_loss) != type([]):
            train_loss = [train_loss for _ in range(n_tasks)]
        assert len(train_loss) == n_tasks, f'len(train_loss) != n_tasks: {train_loss, n_tasks}'
        self.train_loss = train_loss
        self.metric_dict = metric_dict
        self.scheduler = scheduler
        self.log_on_epoch = log_on_epoch
        
        if n_val_tasks == -1:
            n_val_tasks = n_tasks
        
        self.n_tasks = n_tasks
        self.n_val_tasks = n_val_tasks

        self.train_data_names = train_data_names
        if train_data_names == None:
            self.train_data_names = [f'train_err_{i}' for i in range(n_tasks)]

        self.val_data_names = val_data_names
        if val_data_names == None:
            self.val_data_names = [f'val_err_{i}' for i in range(n_val_tasks)]
        
        self.train_loss = train_loss
        self.metric_dict = metric_dict

    def configure_optimizers(self):
        return {'optimizer':self.optimizer, 'lr_scheduler': self.scheduler}
    
    def training_step(self, batch, *args, **kwargs):
        # train loader is a CombinedLoader(dataloaders, "max_size_cycle")
        # batch: [sampled(datasets[i]) for i in range(num_datasets)]
        assert len(batch) == self.n_tasks, f'len(batch) != self.n_tasks, {len(batch), self.n_tasks}'
        train_err_batch = 0
        for i in range(self.n_tasks):
            self.optimizer.zero_grad()
            loss = self.train_loss[i](self.model(**batch[i]), batch[i]['y'])
            self.log(self.train_data_names[i] + '_batch', loss, on_epoch=False, on_step=True)
            self.log(self.train_data_names[i], loss, on_epoch=True, on_step=False)
            loss.backward()
            self.optimizer.step()
            train_err_batch += loss.detach()
        self.log('train_err_batch', train_err_batch, on_epoch=False, on_step=True)
        self.log('train_err', train_err_batch, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, *args, **kwargs):
        # val loader is a CombinedLoader(dataloaders, "sequential")
        assert len(batch) == self.n_val_tasks, f'len(batch) != self.n_val_tasks, {len(batch), self.n_val_tasks}'
        pred = self.model(**batch)
        loss_dict = dict()
        # data_name = self.val_data_names[dataloader_idx]
        for key in self.metric_dict.keys():
            loss_dict[key] = self.metric_dict[key](pred, batch['y'])
            # loss_dict[key+"/"+data_name] = self.metric_dict[key](pred, batch['y'])
        self.log_dict(loss_dict)
    
    def test_step(self, batch, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss_dict[key] = self.metric_dict[key](pred, batch['y'])
        self.log_dict(loss_dict)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        pred = self.model(**batch)
        self.log('predict_value', pred)
        return pred
    
    
class MultiTaskMoEModule(L.LightningModule):
    """
    Multi-Task Learning Pytorch-Lightning module for AI for Science Tasks.
    Defines the train/val/test/predict steps, the major features are:
        1. The prediction target for each batch is batch['y']

    Must be cocupled with the MultiTask DataModule!
    TODO: add single metric
    """
    pass
