import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from loss import *
from dataset import *
from utils import *
from hjs_model import *
from predict_tools import *


def train(cfg, train_tfms=None, valid_tfms=None):
    # for reporduction
    seed = cfg.seed
    torch.cuda.empty_cache()
    seed_everything(seed)

    # device type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define model
    if cfg.target_type == 'offset':
        yaml_name = "offset_train.yaml"
    elif cfg.target_type == 'gaussian':
        yaml_name = "heatmap_train2.yaml"

    yaml_path = os.path.join('/Result', 'Config', yaml_name)
#   model = model_define(yaml_path, cfg.init_training)
    model = model_define(os.path.join('/Result', 'Config', yaml_name))
    model = model.to(device)

    # define criterions
    if cfg.target_type == "offset":
        main_criterion = OffsetMSELoss(True)
    elif cfg.target_type == "gaussian":
        if cfg.loss_type == "MSE":
            main_criterion = HeatmapMSELoss(True)
        elif cfg.loss_type == "OHKMMSE":
            main_criterion = HeatmapOHKMMSELoss(True)
    rmse_criterion = JointsRMSELoss()

    # define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    transform_train = A.Compose([
        A.Resize(384, 288, p=1),
        #    A.HorizontalFlip(p=0.5),
        #    ToTensorV2(p=1),
    ], keypoint_params=A.KeypointParams(format='xy'))

    transform_val = A.Compose([
        A.Resize(384, 288, p=1),
        #    ToTensorV2(p=1),
    ], keypoint_params=A.KeypointParams(format='xy'))

    # augmentations only contains methods about color
    augmentations = A.Compose([
        A.GaussianBlur(p=0.5),
        A.ChannelShuffle(p=0.5),
        #   A.MotionBlur(p=0.5),
        A.RGBShift(p=0.5)
    ], keypoint_params=A.KeypointParams(format='xy'))

    train_dataset = CowData(
        cfg=cfg, transform=transform_train, augmentations=augmentations)
    val_dataset = CowData(cfg=cfg, transform=transform_val, mode='val')
    train_dl = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dl = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False)

    print("Train Transformation:\n", train_tfms, "\n")
    print("Valid Transformation:\n", valid_tfms, "\n")

    best_loss = float('INF')
    best_acc = 0
    for epoch in range(cfg.epochs):
        ################
        #    Train     #
        ################
        with tqdm(train_dl, total=train_dl.__len__(), unit="batch") as train_bar:
            train_acc_list = []
            train_rmse_list = []
            train_heatmap_list = []
            train_coord_list = []
            train_offset_list = []
            train_total_list = []

            for sample in train_bar:
                train_bar.set_description(f"Train Epoch {epoch+1}")

                optimizer.zero_grad()
                images, targ_coords = sample['image'].type(torch.FloatTensor).to(
                    device), sample['keypoints'].type(torch.FloatTensor).to(device)
                target, target_weight = sample['target'].type(torch.FloatTensor).to(
                    device), sample['target_weight'].type(torch.FloatTensor).to(device)
              #   print(target.shape)
                model.train()
                with torch.set_grad_enabled(True):
                    preds = model(images)

                    if cfg.target_type == "offset":
                        loss_hm, loss_os = main_criterion(
                            preds, target, target_weight)
                        loss = loss_hm + loss_os
                    elif cfg.target_type == "gaussian":
                        loss = main_criterion(preds, target, target_weight)

                    if cfg.target_type == "offset":
                        heatmap_height = preds.shape[2]
                        heatmap_width = preds.shape[3]
                        pred_coords = get_final_preds(
                            cfg, preds.detach().cpu().numpy())
                    elif cfg.target_type == 'gaussian':
                        heatmap_height = preds.shape[2]
                        heatmap_width = preds.shape[3]
                        pred_coords, _ = get_max_preds(
                            preds.detach().cpu().numpy())
                        pred_coords[:, :, 0] = pred_coords[:, :, 0] / \
                            (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
                        pred_coords[:, :, 1] = pred_coords[:, :, 1] / \
                            (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)

                    pred_coords = torch.tensor(pred_coords).float().to(device)
                    coord_loss = calc_coord_loss(pred_coords, targ_coords)

                    rmse_loss = rmse_criterion(pred_coords, targ_coords)
                  #   _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                  #                                    target.detach().cpu().numpy()[:, ::3, :, :])
                    avg_acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(preds.detach().cpu().numpy(),
                                                                              target.detach().cpu().numpy(), 0.35, 0.5, None)

                    loss.backward()
                    optimizer.step()

                    if cfg.target_type == "offset":
                        train_heatmap_list.append(loss_hm.item())
                        train_offset_list.append(loss_os.item())
                    train_rmse_list.append(rmse_loss.item())
                    train_total_list.append(loss.item())
                    train_coord_list.append(coord_loss.item())
                    train_acc_list.append(acc_PCK[0])
                train_acc = np.mean(train_acc_list)
                train_rmse = np.mean(train_rmse_list)
                train_coord = np.mean(train_coord_list)
                train_total = np.mean(train_total_list)

                if cfg.target_type == "offset":
                    train_offset = np.mean(train_offset_list)
                    train_heatmap = np.mean(train_heatmap_list)
                    train_bar.set_postfix(heatmap_loss=train_heatmap,
                                          coord_loss=train_coord,
                                          offset_loss=train_offset,
                                          rmse_loss=train_rmse,
                                          total_loss=train_total,
                                          train_acc=train_acc)
                else:
                    train_bar.set_postfix(coord_loss=train_coord,
                                          rmse_loss=train_rmse,
                                          total_loss=train_total,
                                          train_acc=train_acc)

        ################
        #    Valid     #
        ################
        with tqdm(valid_dl, total=valid_dl.__len__(), unit="batch") as valid_bar:
            valid_acc_list = []
            valid_rmse_list = []
            valid_heatmap_list = []
            valid_coord_list = []
            valid_offset_list = []
            valid_total_list = []
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch {epoch+1}")

                images, targ_coords = sample['image'].type(torch.FloatTensor).to(
                    device), sample['keypoints'].type(torch.FloatTensor).to(device)
                target, target_weight = sample['target'].type(torch.FloatTensor).to(
                    device), sample['target_weight'].type(torch.FloatTensor).to(device)

                model.eval()
                with torch.no_grad():
                    preds = model(images)
                    if cfg.target_type == "offset":
                        loss_hm, loss_os = main_criterion(
                            preds, target, target_weight)
                        loss = loss_hm + loss_os
                    elif cfg.target_type == "gaussian":
                        loss = main_criterion(preds, target, target_weight)

                    pred_coords = get_final_preds(
                        cfg, preds.detach().cpu().numpy())
                    pred_coords = torch.tensor(pred_coords).float().to(device)
                    coord_loss = calc_coord_loss(pred_coords, targ_coords)

                    rmse_loss = rmse_criterion(pred_coords, targ_coords)
                    #   _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                    #                                    target.detach().cpu().numpy()[:, ::3, :, :])
                    avg_acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(preds.detach().cpu().numpy(),
                                                                              target.detach().cpu().numpy(), 0.35, 0.5, None)
                    if cfg.target_type == "offset":
                        valid_heatmap_list.append(loss_hm.item())
                        valid_offset_list.append(loss_os.item())
                    valid_rmse_list.append(rmse_loss.item())
                    valid_total_list.append(loss.item())
                    valid_coord_list.append(coord_loss.item())
                    valid_acc_list.append(acc_PCK[0])
                valid_acc = np.mean(valid_acc_list)
                valid_rmse = np.mean(valid_rmse_list)
                valid_coord = np.mean(valid_coord_list)
                valid_total = np.mean(valid_total_list)
                if cfg.target_type == "offset":
                    valid_offset = np.mean(valid_offset_list)
                    valid_heatmap = np.mean(valid_heatmap_list)
                    valid_bar.set_postfix(heatmap_loss=valid_heatmap,
                                          coord_loss=valid_coord,
                                          offset_loss=valid_offset,
                                          rmse_loss=valid_rmse,
                                          total_loss=valid_total,
                                          valid_acc=valid_acc)
                else:
                    valid_bar.set_postfix(coord_loss=valid_coord,
                                          rmse_loss=valid_rmse,
                                          total_loss=valid_total,
                                          valid_acc=valid_acc)

        #   if best_loss > valid_total:
        #       best_model = model
        #       save_dir = os.path.join(cfg.main_dir, cfg.save_folder)
        #       save_name = f'best_model_{valid_total}.pth'
        #       torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        #       print(f"Valid Loss: {valid_total:.8f}\nBest Model saved.")
        #       best_loss = valid_total
        # if best_acc < valid_acc:
        best_model = model
        # save_dir = os.path.join(cfg.main_dir, cfg.save_folder)
        save_name = f'second_model.pth'
        torch.save(model.state_dict(), os.path.join(
            '/Result/weights/', save_name))
        print(f"Valid Acc: {valid_acc:.8f}\nBest Model saved.")
        best_acc = valid_acc

    return best_model


if __name__ == "__main__":
    cfg = SingleModelConfig()
    best_model = train(cfg)
