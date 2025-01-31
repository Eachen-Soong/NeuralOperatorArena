import matplotlib.pyplot as plt
import yaml
from types import SimpleNamespace

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import time

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9504))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary

import lightning as L
from lightning.pytorch import seed_everything
from lightning_modules import MultiTaskModule, PredictionWriter
from utils.losses import LpLoss, H1Loss

from scripts.get_parser import Fetcher
from scripts.models import FNOParser, LSMParser, CNOParser, FNO_OriginalParser
from scripts.datasets import MultiTaskTorusVisForceParser, MultiTaskCylinderFlowParser

ModelParsers = [FNOParser, LSMParser, CNOParser, FNO_OriginalParser]
DataParsers = [MultiTaskTorusVisForceParser, MultiTaskCylinderFlowParser]

"""
    Here we suppose all models are saved in such form:
    model_dir
    |- ___.ckpt
    |- hparams.yaml
    | ...

"""

def find_files_by_suffix(directory, suffix):
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                result.append(os.path.join(root, file))
    return result


def run(raw_args=None):
    fetcher = Fetcher(DataParsers=DataParsers, ModelParsers=ModelParsers, mode='test')

    args = fetcher.parse_args(raw_args)

    verbose = args.verbose

    
    # # # Data Preparation # # #
    _, test_loader = fetcher.get_data(args)

    
    # # # Create Lightning Module # # #
    # 1. Model Definition
    assert args.load_path != '', "Please provide the parent path of the checkpoint file!"
    hparams_path = os.path.join(args.load_path, 'hparams.yaml')
    if os.path.exists(hparams_path):
        with open(hparams_path) as stream:
            try:
                hparams = SimpleNamespace(**yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
    else:
        hparams = args
    
    print(hparams)
    model = fetcher.get_model(hparams)

    ckpt_list = find_files_by_suffix(args.load_path, '.ckpt')
    assert len(ckpt_list) == 1, f"Number of files ended with .ckpt should be 1, instead of {len(ckpt_list)}."
    
    ckpt = torch.load(ckpt_list[0])
    prefix = "model."
    state_dict = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}

    model.load_state_dict(state_dict)

    # 2. Visualizer
    # visualizer = fetcher.get_visualizing_function()

    del fetcher
    

    # 3. Loss Definition
    loss_dict = {'h1': H1Loss(d=2, reductions=args.loss_reduction), 'l2': LpLoss(d=2, p=2, reductions=args.loss_reduction) , 'l1': LpLoss(d=2, p=1, reductions=args.loss_reduction)}

    try:
        eval_loss_names = args.eval_loss
        if type(eval_loss_names) == type(''):
            eval_loss_names = [eval_loss_names]
        eval_losses = {key: loss_dict[key] for key in eval_loss_names}
    except: print(f"Unsupported eval loss! {args.eval_loss}")

    if verbose:
        print('\n### MODEL ###\n', model)
        print('\n### LOSSES ###')
        print(f'\n * Evaluation: {eval_losses}')
        sys.stdout.flush()

    n_val_tasks = len(test_loader)
    use_sum_reduction = (args.loss_reduction == 'sum')
    module = MultiTaskModule(model=model, optimizer=None, scheduler=None, train_loss=None, metric_dict=None, n_tasks=0, n_val_tasks=n_val_tasks, average_over_batch=use_sum_reduction, prediction_output_x=args.log_input)

    save_dir = args.save_dir if len(args.save_dir) else args.load_path
    # # # Predicting or Testing # # #
    trainer = L.Trainer(
        callbacks=[
            PredictionWriter(output_dir=save_dir, write_interval='epoch')
        ]
    )
    trainer.predict(module, test_loader)

if __name__ == '__main__':
    run()