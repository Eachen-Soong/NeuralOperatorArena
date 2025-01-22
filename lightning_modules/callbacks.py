import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional
import psutil
import string
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.trainer import Trainer
import os

"""
    Multitask Learning module is somehow not compatible with the ModelCheckpoint Callback 
    of Lightning, so we implemented a version of that on our own.

    TODO: 
    1. testing
    1. prediction: 2D case visualization

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


class CustomModelCheckpoint(L.Callback):
    def __init__(self, dirpath, monitor='val_loss', mode='min', save_top_k=1):
        """
            ModelCheckpoint Callback
        """
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_scores = []
        self.best_paths = []

        if mode not in ['min', 'max']:
            raise ValueError("mode should be 'min' or 'max'")

        self.compare_op = min if mode == 'min' else max
        self.best_score = float('inf') if mode == 'min' else float('-inf')

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            pl_module.print(f"Metric {self.monitor} not found. Skipping checkpointing.")
            return

        current_score = current_score.item()

        is_best = self.compare_op(current_score, self.best_score) == current_score

        if is_best:
            self.best_score = current_score
            checkpoint_path = os.path.join(
                self.dirpath,
                f"best_model-{self.monitor}={current_score:.4f}.ckpt"
            )
            self._save_checkpoint(trainer, pl_module, checkpoint_path)

    def _save_checkpoint(self, trainer, pl_module, checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)
        pl_module.print(f"Saved best model to {checkpoint_path}")

        self.best_paths.append(checkpoint_path)
        self.best_scores.append(self.best_score)

        if len(self.best_paths) > self.save_top_k:
            idx_to_remove = self.best_scores.index(self.compare_op(self.best_scores))
            path_to_remove = self.best_paths.pop(idx_to_remove)
            self.best_scores.pop(idx_to_remove)

            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                pl_module.print(f"Removed old model checkpoint: {path_to_remove}")



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
    
    def aggregate_metrics(self, logs):
        new_logs = {}
        for key, metric_list in self.metric_groups.items():
            new_logs[key] = 0.
            for metric in metric_list:
                new_logs[key] += logs[metric]
                # print(f'{metric}: {logs[metric]}')
            if self.method == 'mean':
                if len(metric_list):
                    new_logs[key] /= len(metric_list)
            print(f'{key}: {new_logs[key]}')
        return new_logs
        
    def on_validation_epoch_end(self, trainer:Trainer, pl_module:L.LightningModule):
        logs = trainer.callback_metrics
        total_metric_len = sum([len(value) for _, value in self.metric_groups.items()])
        if not total_metric_len:
            metric_names = trainer.callback_metrics.keys()
            metric_groups = self.group_metrics(metric_names)
            total_metric_len = sum([len(value) for _, value in metric_groups.items()])
            if total_metric_len:
                self.metric_groups = metric_groups
                print("self.metric_groups set: ", metric_names, self.metric_groups)
        pl_module.log_dict(self.aggregate_metrics(logs))
        return super().on_validation_epoch_end(trainer, pl_module)


class FooCallback(L.Callback):
    """
        Aggregates all metrics with a certain prefix into a metric named by the prefix.
        E.g. 'l2/dataloader_idx_0' 'l2/dataloader_idx_1' -> 'l2'
    """

    def __init__(self, suffix='_'):
        super().__init__()
        self.suffix = suffix
        self.metric_groups = {}

    def group_metrics(self, metrics):
        metric_groups = {}
        for name in metrics:
            metric_groups[name] = name+self.suffix
        return metric_groups

    def on_train_epoch_end(self, trainer:Trainer, pl_module:L.LightningModule):
        logs = trainer.callback_metrics
        if not len(self.metric_groups):
            metric_names = trainer.callback_metrics.keys()
            self.metric_groups = self.group_metrics(metric_names)
        new_logs = {self.metric_groups[key]: logs[key] for key in self.metric_groups.keys()}
        pl_module.log_dict(new_logs)
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

