import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import time

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary

import lightning as L
from lightning.pytorch import seed_everything
from lightning_modules import MultiMetricModule
from utils.losses import LpLoss, H1Loss

from scripts.get_parser import Fetcher
from scripts.models import FNOParser, LSMParser
from scripts.datasets import BurgersParser, DarcyParser, TorusLiParser, TorusVisForceParser

ModelParsers = [FNOParser, LSMParser]
DataParsers = [BurgersParser, DarcyParser, TorusLiParser, TorusVisForceParser]
loss_dict = {'h1': H1Loss(d=2), 'l2': LpLoss(d=2, p=2)}

def run(raw_args=None):
    fetcher = Fetcher(DataParsers=DataParsers, ModelParsers=ModelParsers)

    args = fetcher.parse_args(raw_args)
    verbose = args.verbose
    # # # Seed # # #
    if args.random_seed:
        seed_everything()
    else:
        seed_everything(args.seed)
    
    # # # Data Preparation # # #
    train_loader, val_loader = fetcher.get_data(args)

    
    # # # Create Lightning Module # # #
    # 1. Model Definition
    model = fetcher.get_model(args)
    del fetcher

    if args.load_path != '':
        model.load_state_dict(torch.load(args.load_path))
    
    # 2. Optimizer Definition
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_steps, gamma=args.scheduler_gamma)

    # 3. Loss Definition
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    try: train_loss = loss_dict[args.train_loss]
    except: print("Unsupported training loss!")
    eval_losses={'h1': h1loss, 'l2': l2loss}

    if verbose:
        print('\n### MODEL ###\n', model)
        print('\n### OPTIMIZER ###\n', optimizer)
        print('\n### SCHEDULER ###\n', scheduler)
        print('\n### LOSSES ###')
        print(f'\n * Train: {train_loss}')
        print(f'\n * Test: {eval_losses}')
        sys.stdout.flush()

    module = MultiMetricModule(model=model, optimizer=optimizer, train_loss=train_loss, metric_dict=loss_dict)

    # # # Logs # # #
    save_dir = args.save_dir + '/' + args.data + '/' + args.model + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.version_of_time:
        localtime = time.localtime(time.time())
        time_name = f"{localtime.tm_mon}-{localtime.tm_mday}-{localtime.tm_hour}-{localtime.tm_min}"
        name = 'exp_'+time_name
    else:
        name = None

    logger = TensorBoardLogger(save_dir=save_dir, name=name)
    logger.log_hyperparams(args)
    log_path = logger.log_dir

    with open(log_path + '/model_summary.txt', 'w+') as file:
        file.write(ModelSummary(module).__str__())

    # # # Training # # #
    trainer = L.Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath=log_path, 
                monitor='l2', save_top_k=1
                ),
            EarlyStopping(monitor='l2', min_delta=1e-6, patience=100),
            Timer(),
        ], 
        max_epochs=args.epochs,
        logger=logger,
        )
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    run()