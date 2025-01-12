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
    def __init__(self, model, optimizer, train_loss, metric_dict:dict) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.metric_dict = metric_dict

    def configure_optimizers(self):
        return self.optimizer

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
    def __init__(self, model, optimizer, train_loss, n_tasks, metric_dict:dict) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        # self.train_loss = 
        if type(train_loss) != type([]):
            train_loss = [train_loss for _ in range(n_tasks)]
        assert len(train_loss) == n_tasks, f'len(train_loss) != n_tasks: {train_loss, n_tasks}'
        self.train_loss = train_loss
        self.metric_dict = metric_dict

    def configure_optimizers(self):
        return self.optimizer

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