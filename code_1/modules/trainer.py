from modules.recorders import PerformanceRecorder
from modules.utils import save_yaml, get_logger, make_directory, get_max_preds, calc_coord_loss, get_final_preds
from losses import HeatmapMSELoss, KeypointLoss
from tqdm import tqdm
import yaml
import torch
import numpy as np

class CustomTrainer():

    """ CustomTrainer
        epoch에 대한 학습 및 검증 절차 정의

    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
        logger (`logger`)
    """

    def __init__(self, model, device, loss_fn, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.logger = logger
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()


        # Output
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        self.train_score = 0

        self.validation_loss_mean = 0
        self.validation_loss_sum = 0
        self.validation_score = 0

        
    def train_epoch(self, dataloader, epoch_index=0, verbose=True, logging_interval=1):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
        dataloader (`dataloader`)
        epoch_index (int)
        verbose (boolean)
        logging_interval (int)
        """
        self.model.train()

        for batch_index, sample in enumerate(tqdm(dataloader)):
            images, targ_coords = sample['image'].type(torch.FloatTensor).to(self.device), sample['keypoints'].type(torch.FloatTensor).to(self.device)
            target, target_weight = sample['target'].type(torch.FloatTensor).to(self.device), sample['target_weight'].type(torch.FloatTensor).to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.loss_fn(preds, target, target_weight)

            heatmap_height = preds.shape[2]
            heatmap_width = preds.shape[3]
            pred_coords, _ = get_max_preds(preds.detach().cpu().numpy())
            pred_coords[:, :, 0] = pred_coords[:, :, 0] / (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
            pred_coords[:, :, 1] = pred_coords[:, :, 1] / (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)

            pred_coords = torch.tensor(pred_coords).float().to(self.device)
            coord_loss  = calc_coord_loss(pred_coords, targ_coords)
            _, pck, cnt, pred, _, _ = self.metric_fn(preds.detach().cpu().numpy(), target.detach().cpu().numpy(),0.35,0.5, None)
            avg_acc = pck[0]
            batch_loss_sum = loss.item() * dataloader.batch_size
            self.train_batch_loss_mean_list.append(loss.item())
            self.train_loss_sum += batch_loss_sum

            # Metric
            self.train_batch_score_list.append(avg_acc)
            loss.backward()
            self.optimizer.step()


            # Log verbose
            if verbose & (batch_index % logging_interval == 0):
                msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {loss} score: {avg_acc}"
                print(msg)
                self.logger.info(msg) if self.logger else print(msg)

        self.train_loss_mean = self.train_loss_sum / len(dataloader)
        self.train_score = np.mean(self.train_batch_score_list)
        msg = f'Epoch {epoch_index}, Train, Mean loss: {self.train_loss_mean}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)

    def validate_epoch(self, dataloader, epoch_index=0, verbose=True, logging_interval=1):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
            verbose (boolean)
            logging_interval (int)
        """
        self.model.eval()

        with torch.no_grad():
            for batch_index, sample in enumerate(tqdm(dataloader)):
                images, targ_coords = sample['image'].type(torch.FloatTensor).to(self.device), sample['keypoints'].type(torch.FloatTensor).to(self.device)
                target, target_weight = sample['target'].type(torch.FloatTensor).to(self.device), sample['target_weight'].type(torch.FloatTensor).to(self.device)

                preds = self.model(images)
                loss = self.loss_fn(preds, target, target_weight)

                pred_coords = get_final_preds(preds.detach().cpu().numpy())
                pred_coords = torch.tensor(pred_coords).float().to(self.device)
                coord_loss  = calc_coord_loss(pred_coords, targ_coords)
                _, pck, cnt, pred, _, _ = self.metric_fn(preds.detach().cpu().numpy(), target.detach().cpu().numpy(),0.35,0.5, None)
                avg_acc = pck[0]
                self.validation_batch_loss_mean_list.append(loss.item())
                batch_loss_sum = loss.item() * dataloader.batch_size
                self.validation_loss_sum += batch_loss_sum
                self.validation_batch_score_list.append(avg_acc)
                


                # Log verbose
                if verbose & (batch_index % logging_interval == 0):
                    msg = f"Epoch {epoch_index} validation batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {loss} score: {avg_acc}"
                    print(msg)
                    self.logger.info(msg) if self.logger else print(msg)

            self.validation_loss_mean = self.validation_loss_sum / len(dataloader)
            self.validation_score = np.mean(self.validation_batch_score_list)
            msg = f'Epoch {epoch_index}, Validation, Mean loss: {self.validation_loss_mean}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)


    def clear_history(self):
        """ 한 epoch 종료 후 history 초기화
            Examples:
                >>for epoch_index in tqdm(range(EPOCH)):
                >>    trainer.train_epoch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.validate_epoch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.clear_history()
        """

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # Output
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        self.train_score = 0

        self.validation_loss_mean = 0
        self.validation_loss_sum = 0
        self.validation_score = 0