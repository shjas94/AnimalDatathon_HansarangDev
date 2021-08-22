import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from AID import AID
from utils import SingleModelConfig

'''
parent_dir : parent path which contains imgs and annots
imgs, annots : list of paths which contains images and annotations (already splitted)
mode : whether dataset is train_set or val_set or test_set
'''
# def collate_fn(batch):
#     return tuple(zip(*batch))


class CowData(Dataset):
    def __init__(self, cfg: SingleModelConfig, transform=None, augmentations=None, mode='train', val_ratio=0.2) -> None:
        super().__init__()
        self.parent_dir = cfg.main_dir
        # self.drive_dir = cfg.drive_dir
        self.transform = transform
        self.mode = mode
        image_dir = os.path.join(self.parent_dir, 'images', '*.jpg')
        annotation_dir = os.path.join(self.parent_dir, 'annotations', '*.json')
        images = sorted(glob.glob(image_dir), key=lambda x: int(
            x.split('/')[3].split('_')[3].split('.')[0]))
        annotations = sorted(glob.glob(annotation_dir), key=lambda x: int(
            x.split('/')[3].split('_')[3].split('.')[0]))
        if mode == 'train':
            self.imgs = sorted(glob.glob(os.path.join(self.parent_dir, 'images', '*.jpg')),  # path만 수정
                               key=lambda x: int(x.split('/')[3].split('_')[3].split('.')[0]))[int(9000*val_ratio):]
            self.annots = sorted(glob.glob(os.path.join(self.parent_dir, 'annotations', '*.json')),
                                 key=lambda x: int(x.split('/')[3].split('_')[3].split('.')[0]))[int(9000*val_ratio):]
        elif mode == 'val':
            self.imgs = sorted(glob.glob(os.path.join(self.parent_dir, 'images', '*.jpg')),
                               key=lambda x: int(x.split('/')[3].split('_')[3].split('.')[0]))[:int(9000*val_ratio)]
            self.annots = sorted(glob.glob(os.path.join(self.parent_dir, 'annotations', '*.json')),
                                 key=lambda x: int(x.split('/')[3].split('_')[3].split('.')[0]))[:int(9000*val_ratio)]
        # self.imgs = imgs
        # self.annots = annots
        if cfg.aid and mode == 'train':
            self.aid = AID()
        else:
            self.aid = None
        self.augmentations = augmentations

        self.num_joints = cfg.num_joints
        self.target_type = cfg.target_type
        self.sigma = cfg.sigma
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.output_size
        self.use_different_joints_weight = cfg.use_different_joints_weight
        self.totensor = A.Compose([
            ToTensorV2(p=1.0)
        ], keypoint_params=A.KeypointParams(format='xy'))

    def read_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        annot = data['label_info']['annotations']
        category = data['label_info']['categories'][annot[0]['category_id']-1]
        img = data['label_info']['image']
        return annot, category, img

    def extract_keypoints(self, keypoints, img_info):
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
            elif i % 3 == 1:
                if keypoints[i] == img_info['height']:
                    temp.append(keypoints[i] - 1e-09)
                elif keypoints[i] == 0:
                    temp.append(keypoints[i] + 1e-09)
                else:
                    temp.append(keypoints[i])
                new_keypoints.append(tuple(temp))
                temp = []
        return new_keypoints  # [(x_1, y_1), (x_2, y_2), ...]

    def __getitem__(self, index: int):
        annot_path = self.annots[index]
        annot, category, img_info = self.read_data(annot_path)
        img = cv2.imread(os.path.join(
            self.parent_dir, 'images', img_info['file_name']))

        # 취향에 따라 주석처리하고 Normalize로 교체하셔도 됩니다.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        keypoints = self.extract_keypoints(annot[0]['keypoints'], img_info)
        ### transform ###
        if self.transform:
            transformed = self.transform(image=img_rgb, keypoints=keypoints)
            img_rgb = transformed['image']
            keypoints = transformed['keypoints']
        if self.augmentations and self.mode == 'train':
            augmented = self.augmentations(image=img_rgb, keypoints=keypoints)
            img_rgb = augmented['image']
            keypoints = augmented['keypoints']

        tensor = self.totensor(image=img_rgb, keypoints=keypoints)
        image = tensor['image']
        keypoints = tensor['keypoints']
        # print(keypoints)
        keypoints = np.array(keypoints).flatten().reshape(17, 2)
        # reshape and concatenate to generate heatmap
        keypoints = np.concatenate([keypoints, np.ones((17, 1))], axis=1)
        ################
        target, target_weight = self.generate_target(
            keypoints[:, :2], keypoints[:, 2])
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        if self.aid is not None and self.mode == "train":
            image = self.aid(image)

        sample = {
            'image': image,
            'keypoints': keypoints,
            'target': target.float(),
            'target_weight': target_weight
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
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                           (2 * self.sigma ** 2))

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
                               self.heatmap_size[0] *
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
            target = target.reshape(
                (self.num_joints*3, self.heatmap_size[0], self.heatmap_size[1]))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def __len__(self):
        return len(self.imgs)
