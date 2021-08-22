import os
import random
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm   
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class AID:
    """AID: Augmentation by Informantion Dropping.
    Paper ref: Huang et al. AID: Pushing the Performance Boundary of Human Pose
    Estimation with Information Dropping Augmentation ( arXiv:2008.07139 2020).
    Args:
        transforms (list): ToTensor & Normalize
        prob_cutout (float): Probability of performing cutout.
        radius_factor (float): Size factor of cutout area.
        num_patch (float): Number of patches to be cutout.
        prob_has (float): Probability of performing hide-and-seek.
        prob_has_hide (float): Probability of hiding patches.
    """

    def __init__(self,
                 prob_cutout=0.3,
                 radius_factor=0.1,
                 num_patch=1,
                 prob_has=0.3,
                 prob_has_hide=0.3):

        self.prob_cutout = prob_cutout
        self.radius_factor = radius_factor
        self.num_patch = num_patch
        self.prob_has = prob_has
        self.prob_has_hide = prob_has_hide
        assert (self.prob_cutout + self.prob_has) > 0

    def _hide_and_seek(self, img):
        # get width and height of the image
        ht, wd, _ = img.shape
        # possible grid size, 0 means no hiding
        grid_sizes = [0, 16, 32, 44, 56]

        # randomly choose one grid size
        grid_size = grid_sizes[np.random.randint(0, len(grid_sizes) - 1)]

        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if np.random.rand() <= self.prob_has_hide:
                        img[y:y_end, x:x_end, :] = 0
        return img

    def _cutout(self, img):
        height, width, _ = img.shape
        img = img.reshape((height * width, -1))
        feat_x_int = np.arange(0, width)
        feat_y_int = np.arange(0, height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.reshape((-1, ))
        feat_y_int = feat_y_int.reshape((-1, ))
        for _ in range(self.num_patch):
            center = [np.random.rand() * width, np.random.rand() * height]
            radius = self.radius_factor * (1 + np.random.rand(2)) * width
            x_offset = (center[0] - feat_x_int) / radius[0]
            y_offset = (center[1] - feat_y_int) / radius[1]
            dis = x_offset**2 + y_offset**2
            keep_pos = np.where((dis <= 1) >= 0)[0]
            img[keep_pos, :] = 0
        img = img.reshape((height, width, -1))
        return img

    def __call__(self, img):
        if np.random.rand() < self.prob_cutout:
            img = self._cutout(img)
        if np.random.rand() < self.prob_has:
            img = self._hide_and_seek(img)
        return img

    
class CowData(Dataset):
    def __init__(self, cfg, files, keypoints, mode, augmentations=False)->None:
        super().__init__()
        self.files = files
        self.keypoints = keypoints
        self.kpd = cfg.kpd
        self.mode = mode
        self.image_size = np.array(cfg.dataset.input_size)
        self.output_size = self.image_size//4
        if cfg.aid and self.mode=="train":
            self.aid = AID()
        else:
            self.aid = None
        T = []
        T.append(A.Resize(cfg.dataset.input_size[0],cfg.dataset.input_size[1]))
        if augmentations:
            # 중간에 기구로 잘리는 경우를 가장
            T_ = []
            T_.append(A.Cutout(num_holes=2, max_h_size=30, max_w_size=30, fill_value=0, p=1))
            T_.append(A.Cutout(num_holes=2, max_h_size=30, max_w_size=30, fill_value=255, p=1))
            T_.append(A.Cutout(num_holes=2, max_h_size=30, max_w_size=30, fill_value=128, p=1))
            T_.append(A.Cutout(num_holes=2, max_h_size=30, max_w_size=30, fill_value=192, p=1))
            T_.append(A.Cutout(num_holes=2, max_h_size=30, max_w_size=30, fill_value=64, p=1))
            T.append(A.OneOf(T_))

            '''
            T_ = []
            T_.append(A.RandomGamma(p=1))
            T_.append(A.RandomBrightness(p=1))
            T_.append(A.RandomContrast(p=1))
            T.append(A.OneOf(T_))
            '''
            T.append(A.ChannelShuffle(p=0.5))
            T_ = []
            T_.append(A.MotionBlur(p=1))
            T_.append(A.GaussNoise(p=1))
            T.append(A.OneOf(T_))
        '''
        if cfg.dataset.normalize:
            if cfg.dataset.mean is not None and cfg.dataset.std is not None:
                T.append(A.Normalize(cfg.dataset.mean, cfg.dataset.std))
            else:
                T.append(A.Normalize())
        else:
            T.append(A.Normalize((0, 0, 0), (1, 1, 1)))
        '''
        T.append(ToTensorV2(p=1.0))

        self.transform = A.Compose(
            transforms=T,
            keypoint_params=A.KeypointParams(format="xy")
        )

        self.num_joints = cfg.dataset.num_joints
        self.target_type = cfg.dataset.target_type
        self.sigma= cfg.dataset.sigma
        self.heatmap_size = self.output_size
        self.use_different_joints_weight = cfg.use_different_joints_weight


    def __getitem__(self, idx):
        file = str(self.files[idx])
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.
        # 취향에 따라 주석처리하고 Normalize로 교체하셔도 됩니다.
        ### transform ###
        keypoint = self.keypoints[idx]
        a = self.transform(image=image, keypoints=keypoint)
        image = a["image"]
        keypoints = a['keypoints']
        # print(keypoints)
        keypoints = np.array(keypoints)
        keypoints = np.concatenate([keypoints, np.ones((17, 1))], axis=1) # reshape and concatenate to generate heatmap
        ################
        target, target_weight = self.generate_target(keypoints[:,:2], keypoints[:, 2])
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        if self.aid is not None and self.mode=="train":
            image = self.aid(image)

        sample = {
            'image':image,
            'keypoints':keypoints,
            'target':target.float(),
            'target_weight':target_weight
        }
        return sample
    # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/dataset/JointsDataset.py

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                            self.heatmap_size[0],
                            self.heatmap_size[1]),
                            dtype=np.float32)
            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[1] or ul[1] >= self.heatmap_size[0] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                '''
                mu_x_ac = joints[joint_id][0] / feat_stride[0] 
                mu_y_ac = joints[joint_id][1] / feat_stride[1] 
                x0 += mu_x_ac-mu_x 
                y0 += mu_y_ac-mu_y
                '''
                
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[1]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[0]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[1])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[0])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif self.target_type == 'offset':
            target = np.zeros((self.num_joints,
                            3,
                            self.heatmap_size[0]*
                            self.heatmap_size[1]),
                            dtype=np.float32)
            feat_width = self.heatmap_size[1]
            feat_height = self.heatmap_size[0]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.reshape((-1,))
            feat_y_int = feat_y_int.reshape((-1,))
            kps_pos_distance_x = self.kpd
            kps_pos_distance_y = self.kpd
            feat_stride = (self.image_size - 1.0) / (self.heatmap_size - 1.0)
            for joint_id in range(self.num_joints):
                mu_x = joints[joint_id][0] / feat_stride[0]
                mu_y = joints[joint_id][1] / feat_stride[1]

                x_offset = (mu_x - feat_x_int) / kps_pos_distance_x
                y_offset = (mu_y - feat_y_int) / kps_pos_distance_y

                dis = x_offset ** 2 + y_offset ** 2
                keep_pos = np.where((dis <= 1) >= 0)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target=target.reshape((self.num_joints*3,self.heatmap_size[0],self.heatmap_size[1]))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def __len__(self):
        return len(self.files)
    
def extract_keypoints(keypoints, img_info):
    new_keypoints = []
    temp = []
    for i in range(len(keypoints)):
        if i % 3 == 2:
            continue
        elif i % 3 == 0:
            if keypoints[i] == img_info['width']:
                temp.append(keypoints[i] - 1e-09)
            elif keypoints[i] == 0:
                temp.append(keypoints[i] + 1e-09)
            else:
                temp.append(keypoints[i])
        elif i % 3  == 1:
            if keypoints[i] == img_info['height']:
                temp.append(keypoints[i] - 1e-09)
            elif keypoints[i] == 0:
                temp.append(keypoints[i] + 1e-09)
            else:
                temp.append(keypoints[i])
            new_keypoints.append(tuple(temp))
            temp = []
    return new_keypoints # [(x_1, y_1), (x_2, y_2), ...]

def get_pose_datasets(cfg):
    total_imgs = np.array(sorted(list(glob.glob("/DATA/train/images/*.jpg"))))
    annots = np.array(sorted(list(glob.glob("/DATA/train/annotations/*.json"))))
    keypoints = []
    for t in annots:
        with open(t, 'r') as f:
            data = json.loads(f.read())
            annot = data['label_info']['annotations']
            img_info = data['label_info']['image']
            keypoints.append(extract_keypoints(annot[0]["keypoints"],img_info))
    total_keypoints = np.array(keypoints)
    img_train, img_test, key_train, key_test = train_test_split(total_imgs,total_keypoints, test_size=0.2,random_state = 42)

    # 데이터셋 생성
    ds_train = CowData(
        cfg,
        img_train,
        key_train,
        mode = "train",
        augmentations=True,
    )
    ds_valid = CowData(
        cfg,
        img_test,
        key_test,
        mode="valid",
        augmentations=False,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_cpus,
        shuffle=True,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_cpus,
        shuffle=False,
        pin_memory=True,
    )

    return dl_train, dl_valid