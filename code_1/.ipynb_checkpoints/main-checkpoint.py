import os
import random
import numpy as np
import pandas as pd
import torch
from modules.utils import make_directory, get_logger, save_yaml, accuracy
from datetime import datetime, timezone, timedelta
from modules.pose import get_pose_datasets
from easydict import EasyDict
import yaml
from model import get_pose_net
from modules.trainer import CustomTrainer
from torch.optim import lr_scheduler
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from losses import HeatmapMSELoss,KeypointLoss
import torch.optim as optim
import argparse

def model_define(yaml_path, train=True):
    with open(yaml_path) as f:
        cfg = yaml.load(f)
    # cfg['MODEL']['PRETRAINED']=False
    model = get_pose_net(cfg, train)
    return model

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


DEBUG = False
PROJECT_DIR = "./"
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'cow_{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)

if __name__ == '__main__':
    # wandb.init(entity='jo_member', project='vqa', name='original_large')
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    make_directory(PERFORMANCE_RECORD_DIR)
    system_logger = get_logger(name='train', file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    args = args.parse_args()

    with open(args.config, "r") as f:
        cfg = EasyDict(yaml.load(f, yaml.FullLoader))

    yaml_name = "heatmap_train2.yaml"
    PERFORMANCE_RECORD_COLUMN_NAME_LIST = cfg['PERFORMANCE_RECORD']['column_list']
    model = model_define(os.path.join('../', 'Config', yaml_name))
    model = model.to(device)
    system_logger.info('===== Review Model Architecture =====')
    system_logger.info(f'{model} \n')
    metric_fn = accuracy
    if cfg.train.loss_type == "ce":
        criterion = KeypointLoss(True)
    elif cfg.train.loss_type == "hmse":
        criterion = HeatmapMSELoss(True)
    else:
        raise NotImplementedError()
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.train.scheduler.params)
    dl_train, dl_valid = get_pose_datasets(cfg)

    trainer = CustomTrainer(model, device, criterion, metric_fn, optimizer, scheduler, logger=system_logger)
    early_stopper = LossEarlyStopper(patience=10, verbose=False, logger=system_logger)
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        "HRNet",
        "AdamW",
        "CE",
        "Acc",
        10,
        cfg.dataset.batch_size,
        cfg.train.max_epochs,
        cfg.train.lr,
        "",
        cfg.seed]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=None)
    
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), cfg)
    #wandb.watch(model)
    EPOCHS = cfg.train.max_epochs
    for epoch in range(EPOCHS):
        trainer.train_epoch(dl_train, epoch_index=epoch)
        trainer.validate_epoch(dl_valid, epoch_index=epoch)
        performance_recorder.add_row(epoch_index=epoch,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)
        #wandb.log({"train_loss": trainer.train_loss_mean})
        #wandb.log({"valid_loss": trainer.validation_loss_mean})
        #wandb.log({"train_acc": trainer.train_score})
        #wandb.log({"valid_acc": trainer.validation_score})
        #wandb.log({"lr" : get_lr(optimizer)})
        scheduler.step(trainer.validation_loss_mean)
        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)
        if early_stopper.stop:
            break

        trainer.clear_history()
    performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'last.pt')
    performance_recorder.save_weight()