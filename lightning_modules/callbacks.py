import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional
import psutil
import string
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.trainer import Trainer

"""
TODO: 
0. log model infos into run file on init start
1. CKPT auto-save
2. testing
3. prediction: 2D case visualization

"""

class MemoryMonitoringCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.process = psutil.Process()

    def log_memory_usage(self, stage):
        memory_info = self.process.memory_info()
        memory_used = memory_info.rss / (1024 ** 3)  # 转换为 GB

        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为 GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # 转换为 GB
        else:
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0

        print(f"[{stage}] Python Process Memory: {memory_used:.2f} GB used")
        if torch.cuda.is_available():
            print(f"[{stage}] GPU Memory: {gpu_memory_allocated:.2f} GB allocated / {gpu_memory_reserved:.2f} GB reserved")

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_memory_usage('Train Start')

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_memory_usage('Train End')

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_memory_usage('Validation Start')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_memory_usage('Validation End')

    def on_test_epoch_start(self, trainer, pl_module):
        self.log_memory_usage('Test Start')

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_memory_usage('Test End')


class AggregateMetricCallback(L.Callback):
    """
        Aggregates all metrics with a certain prefix into a metric named by the prefix.
        E.g. 'l2/dataloader_idx_0' 'l2/dataloader_idx_1' -> 'l2'
    """

    def __init__(self, prefixes_to_sum:list, method='sum'):
        super().__init__()
        self.prefixes_to_sum = prefixes_to_sum
        self.metric_groups = {}
        self.method = method

    def group_metrics(self, metrics):
        metric_groups = {}
        for prefix in self.prefixes_to_sum:
            metric_groups[prefix] = []
        for name in metrics:
            for prefix in self.prefixes_to_sum:
                if name.startswith(prefix) and len(name) > len(prefix):
                    if name[len(prefix)] == '/':
                        metric_groups[prefix].append(name)
                        break
        return metric_groups

    def on_train_start(self, trainer:Trainer, pl_module:L.LightningModule):
        metric_names = trainer.callback_metrics.keys()
        self.metric_groups = self.group_metrics(metric_names)
        return super().on_train_start(trainer, pl_module)
    
    def aggregate_metrics(self, logs):
        new_logs = {}
        for key, metric_list in self.metric_groups.items():
            new_logs[key] = 0.
            for metric in metric_list:
                new_logs[key] += logs[metric]
            if self.method == 'mean':
                if len(metric_list):
                    new_logs[key] /= len(metric_list)
        return new_logs

    def on_train_epoch_end(self, trainer:Trainer, pl_module:L.LightningModule):
        logs = trainer.callback_metrics
        pl_module.log_dict(self.aggregate_metrics(logs))
        return super().on_train_epoch_end(trainer, pl_module)
        

# class LossNomorlizationCallback(L.Callback):
#     def __init__(self, std:Optional[float]=None) -> None:
#         super().__init__()
#         self.std = std
#         epsilon = 1e-6
#         if self.std is not None:
#             self.coeff = 1 / (self.std**2 + epsilon)
#         else:
#             self.coeff = 1

#     def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: torch.Tensor) -> None:
#         return super().on_before_backward(trainer, pl_module, loss*self.coeff)

