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
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)


from typing import List

def get_kpts(cfg, pred_coords, img_h, img_w):
    kpts = []
    for coord in pred_coords: # (2) in (17,2)
        #print("resized_coord, [width, height]:", coord)
        w, h = coord
        y = min(int(h * img_h / cfg.input_size[0]), img_h)
        x = min(int(w * img_w / cfg.input_size[1]), img_w)
        y = int(y)
        x = int(x)
        #print("original_coord, [width, height]", [x, y])
        kpts.append([x,y])
    return kpts

class SingleModelConfig:
    def __init__(self,
                 input_size: List[int] = [384, 288],
                 kpd: float = 4.0,
                 epochs: int = 150,
                 sigma: float = 3.0,
                 num_joints: int = 17,
                 batch_size: int = 16,
                 random_seed: int = 42,
                 test_ratio: float = 0.001,
                 learning_rate: float = 1e-4,
                 main_dir: str = '/DATA',
                 drive_dir: str = '/Result',
                 save_folder: str = '/Result/weights',  # 추후 추가

                 loss_type: str = "MSE",
                 target_type: str = "gaussian",
                 post_processing: str = "dark",

                 aid: bool = True,
                 yoga: bool = False,
                 debug: bool = False,
                 shift: bool = False,
                 startify: bool = False,
                 init_training: bool = False,
                 startify_with_dir: bool = False,
                 use_different_joints_weight: bool = False,
                 ):
        self.input_size = input_size
        self.main_dir = main_dir
        self.drive_dir = os.path.join(main_dir, drive_dir)
        self.epochs = epochs
        self.seed = random_seed
        self.lr = learning_rate
        self.loss_type = loss_type
        self.num_joints = num_joints
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.init_training = init_training
        self.aid = aid
        self.kpd = kpd
        self.sigma = sigma
        self.shift = shift
        self.debug = debug
        self.startify = startify
        self.target_type = target_type
        self.image_size = np.array(input_size)
        self.output_size = self.image_size // 4
        self.post_processing = post_processing
        self.startify_with_dir = startify_with_dir
        self.use_different_joints_weight = use_different_joints_weight
        self.save_folder = os.path.join(drive_dir, save_folder)

        self.joints_name = {
            1: 'fore_head',
            2: 'neck',
            3: 'fore_spine',
            4: 'fore_right_shoulder',
            5: 'fore_right_knee',
            6: 'fore_right_foot',
            7: 'fore_left_shoulder',
            8: 'fore_left_knee',
            9: 'fore_left_foot',
            10: 'rear_spine',
            11: 'rear_right_shoulder',
            12: 'rear_right_knee',
            13: 'rear_right_foot',
            14: 'rear_left_shoulder',
            15: 'rear_left_knee',
            16: 'rear_left_foot',
            17: 'hip'
        }

        self.joint_pair = [
            (1, 2), (2, 3), (3, 4), (3, 7), (3, 10), (4, 5),
            (5, 6), (7, 8), (8, 9), (10, 11), (10, 14),
            (11, 12), (12, 13), (14, 15), (15, 16), (10, 17),
        ]

        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, self.num_joints + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        self.joint_colors = {k: colors[k] for k in range(self.num_joints)}

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


'''
parent_dir : parent path which contains imgs and annots
imgs, annots : list of paths which contains images and annotations (already splitted)
mode : whether dataset is train_set or val_set or test_set
'''


# def collate_fn(batch):
#     return tuple(zip(*batch))
class CowData(Dataset):
    def __init__(self, cfg:SingleModelConfig, transform=None, augmentations=None, mode='train', val_ratio=0.12)->None:
        super().__init__()
        self.parent_dir= cfg.main_dir #
        self.transform=transform
        self.mode=mode
        
        if self.mode == "train" or self.mode == "val": #/DATA/train/images/
            image_dir = os.path.join(self.parent_dir, 'train','images', '*.jpg')
            annotation_dir = os.path.join(self.parent_dir, 'train', 'annotations', '*.json')
            images = sorted(glob.glob(image_dir), key=lambda x: int(x.split('/')[4].split('_')[3].split('.')[0]))
            annotations = sorted(glob.glob(annotation_dir), key=lambda x: int(x.split('/')[4].split('_')[3].split('.')[0]))
        elif self.mode == "test": #/DATA/test/ADK2021_TestDataSet/images/
            image_dir = os.path.join(self.parent_dir, 'test', 'ADK2021_TestDataSet', 'images', '*.jpg')
            #print(image_dir.split('/')[5].split('_')[4])
            images = sorted(glob.glob(image_dir), key=lambda x: int(x.split('/')[5].split('_')[4].split('.')[0]))
        elif self.mode == "vis": #/USER/kw/vis/images/
            image_dir = os.path.join(cfg.code_dir, 'kw', 'vis', 'images', '*.jpg')
            annotation_dir = os.path.join(cfg.code_dir, 'kw', 'vis', 'annotations', '*.json')
            images = sorted(glob.glob(image_dir), key=lambda x: int(x.split('/')[5].split('_')[3].split('.')[0]))
            annotations = sorted(glob.glob(annotation_dir), key=lambda x: int(x.split('/')[5].split('_')[3].split('.')[0]))
        
        
        if mode == 'train':
            self.imgs = sorted(images, key=lambda x: int(x.split('/')[4].split('_')[3].split('.')[0]))[int(len(images)*val_ratio):] 
            self.annots = sorted(annotations, key=lambda x: int(x.split('/')[4].split('_')[3].split('.')[0]))[int(len(annotations)*val_ratio):]
        elif mode == 'val':
            self.imgs = sorted(images, key=lambda x: int(x.split('/')[4].split('_')[3].split('.')[0]))[:int(len(images)*val_ratio)] 
            self.annots = sorted(annotations, key=lambda x: int(x.split('/')[4].split('_')[3].split('.')[0]))[:int(len(annotations)*val_ratio)]
        elif mode == 'test':
            self.imgs = images
        elif mode == 'vis':
            self.imgs = images
            self.annots = annotations
            
        self.aid = cfg.aid
        self.augmentations = augmentations
        self.num_joints = cfg.num_joints
        self.target_type = cfg.target_type
        self.sigma= cfg.sigma
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.output_size
        self.use_different_joints_weight = cfg.use_different_joints_weight
        self.totensor = A.Compose([ToTensorV2(p=1.0)])
        
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

    def __getitem__(self, index: int):
        if self.mode != "test":
            annot_path = self.annots[index]
            annot, category, img_info = self.read_data(annot_path)
            img_path = os.path.join(self.parent_dir, 'train', 'images', img_info['file_name'])
        else : #test
            img = cv2.imread(self.imgs[index])
            img_path = self.imgs[index]
                                    
                                    
        if self.mode == "train" or self.mode == "val":
            img = cv2.imread(os.path.join(self.parent_dir, 'train', 'images', img_info['file_name']))
        elif self.mode == "test":
            None
        elif self.mode == "vis":
            img = cv2.imread(os.path.join(cfg.code_dir, 'kw', 'vis', 'images', img_info['file_name']))
            img_path = os.path.join(self.code_dir, 'train', 'images', img_info['file_name'])
                                    
        height, width, _ = img.shape
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        if self.mode != "test" :
            keypoints = self.extract_keypoints(annot[0]['keypoints'], img_info)
        else :
            keypoints = None
            target = None
            target_weight = None
            
        ### transform ###
        if self.transform:
            if self.mode != "test":
                transformed = self.transform(image=img_rgb, keypoints=keypoints)
            elif self.mode == "test":
                transformed = self.transform(image=img_rgb)
            img_rgb = transformed['image']
            if self.mode != "test":
                keypoints = transformed['keypoints']
            
        if self.augmentations and self.mode=='train':
            if self.mode != "test":
                augmented = self.augmentations(image=img_rgb, keypoints=keypoints)
            img_rgb = augmented['image']
            if self.mode != "test":
                keypoints = augmented['keypoints']
        
        ### aid augmentation ###
        if self.aid is not None and self.mode == "train":
            img_rgb = self.aid(img_rgb)
        
        tensor = self.totensor(image=img_rgb)
        image = tensor['image']
        
        if self.mode != "test" :
            keypoints = tensor['keypoints']
            keypoints = np.array(keypoints)
            keypoints = np.concatenate([keypoints, np.ones((17, 1))], axis=1) # reshape and concatenate to generate heatmap
            ################
            target, target_weight = self.generate_target(keypoints[:,:2], keypoints[:, 2])
            target = torch.from_numpy(target).float()
            target_weight = torch.from_numpy(target_weight)

        sample = {
            'image':image,
            'keypoints':keypoints,
            'target':target,
            'target_weight':target_weight,
            'image_shape':tuple([height, width]),
            'image_path':img_path
        }
        if self.mode == "test" :
            sample = {
            'image':image,
            'image_shape':tuple([height, width]),
            'image_path':img_path
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

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def __len__(self):
        return len(self.imgs)


# https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/ba50a82dce412df97f088c572d86d7977753bf74/lib/core/inference.py#L18:5
# https://www.dacon.io/competitions/official/235701/codeshare/2478?page=1&dtype=recent

from numpy.linalg import LinAlgError


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


# def get_max_preds(batch_heatmaps):
# 	batch_size = batch_heatmaps.shape[0]
# 	num_joints = batch_heatmaps.shape[1]
# 	width      = batch_heatmaps.shape[3]

# 	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
# 	idx               = np.argmax(heatmaps_reshaped, 2)
# 	maxvals           = np.amax(heatmaps_reshaped, 2)

# 	maxvals = maxvals.reshape((batch_size, num_joints, 1))
# 	idx     = idx.reshape((batch_size, num_joints, 1))

# 	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

# 	preds[:,:,0] = (preds[:,:,0]) % width
# 	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

# 	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
# 	pred_mask    = pred_mask.astype(np.float32)

# 	preds *= pred_mask
# 	return preds, maxvals
def dark_post_processing(coords, batch_heatmaps):
    '''
    DARK post-pocessing
    :param coords: batchsize*num_kps*2
    :param batch_heatmaps:batchsize*num_kps*high*width
    :return:
    '''

    shape_pad = list(batch_heatmaps.shape)
    shape_pad[2] = shape_pad[2] + 2
    shape_pad[3] = shape_pad[3] + 2

    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            mapij = batch_heatmaps[i, j, :, :]
            maxori = np.max(mapij)
            mapij = cv2.GaussianBlur(mapij, (7, 7), 0)
            max = np.max(mapij)
            min = np.min(mapij)
            mapij = (mapij - min) / (max - min) * maxori
            batch_heatmaps[i, j, :, :] = mapij
    batch_heatmaps = np.clip(batch_heatmaps, 0.001, 50)
    batch_heatmaps = np.log(batch_heatmaps)
    batch_heatmaps_pad = np.zeros(shape_pad, dtype=float)
    batch_heatmaps_pad[:, :, 1:-1, 1:-1] = batch_heatmaps
    batch_heatmaps_pad[:, :, 1:-1, -1] = batch_heatmaps[:, :, :, -1]
    batch_heatmaps_pad[:, :, -1, 1:-1] = batch_heatmaps[:, :, -1, :]
    batch_heatmaps_pad[:, :, 1:-1, 0] = batch_heatmaps[:, :, :, 0]
    batch_heatmaps_pad[:, :, 0, 1:-1] = batch_heatmaps[:, :, 0, :]
    batch_heatmaps_pad[:, :, -1, -1] = batch_heatmaps[:, :, -1, -1]
    batch_heatmaps_pad[:, :, 0, 0] = batch_heatmaps[:, :, 0, 0]
    batch_heatmaps_pad[:, :, 0, -1] = batch_heatmaps[:, :, 0, -1]
    batch_heatmaps_pad[:, :, -1, 0] = batch_heatmaps[:, :, -1, 0]
    I = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1 = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1y1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_y1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1_ = np.zeros((shape_pad[0], shape_pad[1]))
    coords = coords.astype(np.int32)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            I[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 1, coords[i, j, 0] + 1]
            Ix1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 1, coords[i, j, 0] + 2]
            Ix1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 1, coords[i, j, 0]]
            Iy1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0] + 1]
            Iy1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1], coords[i, j, 0] + 1]
            Ix1y1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0] + 2]
            Ix1_y1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1], coords[i, j, 0]]
    dx = 0.5 * (Ix1 - Ix1_)
    dy = 0.5 * (Iy1 - Iy1_)
    D = np.zeros((shape_pad[0], shape_pad[1], 2))
    D[:, :, 0] = dx
    D[:, :, 1] = dy
    D.reshape((shape_pad[0], shape_pad[1], 2, 1))
    dxx = Ix1 - 2 * I + Ix1_
    dyy = Iy1 - 2 * I + Iy1_
    dxy = 0.5 * (Ix1y1 - Ix1 - Iy1 + I + I - Ix1_ - Iy1_ + Ix1_y1_)
    hessian = np.zeros((shape_pad[0], shape_pad[1], 2, 2))
    hessian[:, :, 0, 0] = dxx
    hessian[:, :, 1, 0] = dxy
    hessian[:, :, 0, 1] = dxy
    hessian[:, :, 1, 1] = dyy
    inv_hessian = np.zeros(hessian.shape)
    # hessian_test = np.zeros(hessian.shape)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            hessian_tmp = hessian[i, j, :, :]
            try:
                inv_hessian[i, j, :, :] = np.linalg.inv(hessian_tmp)
            except LinAlgError:
                inv_hessian[i, j, :, :] = np.zeros((2, 2))
            # hessian_test[i,j,:,:] = np.matmul(hessian[i,j,:,:],inv_hessian[i,j,:,:])
            # print( hessian_test[i,j,:,:])
    res = np.zeros(coords.shape)
    coords = coords.astype(np.float)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            D_tmp = D[i, j, :]
            D_tmp = D_tmp[:, np.newaxis]
            shift = np.matmul(inv_hessian[i, j, :, :], D_tmp)
            # print(shift.shape)
            res_tmp = coords[i, j, :] - shift.reshape((-1))
            res[i, j, :] = res_tmp
    return res


def get_final_preds(cfg, batch_heatmaps):
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    if cfg.target_type == 'gaussian':
        coords, maxvals = get_max_preds(batch_heatmaps)
        if cfg.post_processing == "dark":
            coords = dark_post_processing(coords, batch_heatmaps)
    elif cfg.target_type == 'offset':
        net_output = batch_heatmaps.copy()
        kps_pos_distance_x = cfg.kpd
        kps_pos_distance_y = cfg.kpd
        batch_heatmaps = net_output[:, ::3, :]
        offset_x = net_output[:, 1::3, :] * kps_pos_distance_x
        offset_y = net_output[:, 2::3, :] * kps_pos_distance_y
        for i in range(batch_heatmaps.shape[0]):
            for j in range(batch_heatmaps.shape[1]):
                batch_heatmaps[i, j, :, :] = cv2.GaussianBlur(batch_heatmaps[i, j, :, :], (15, 15), 0)
                offset_x[i, j, :, :] = cv2.GaussianBlur(offset_x[i, j, :, :], (7, 7), 0)
                offset_y[i, j, :, :] = cv2.GaussianBlur(offset_y[i, j, :, :], (7, 7), 0)
        coords, maxvals = get_max_preds(batch_heatmaps)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                px = int(coords[n][p][0])
                py = int(coords[n][p][1])
                coords[n][p][0] += offset_x[n, p, py, px]
                coords[n][p][1] += offset_y[n, p, py, px]

    preds = coords.copy()
    preds[:, :, 0] = preds[:, :, 0] / (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
    preds[:, :, 1] = preds[:, :, 1] / (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)

    return preds


# https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/ba50a82dce412df97f088c572d86d7977753bf74/lib/core/evaluate.py#L41
# https://www.dacon.io/competitions/official/235701/codeshare/2478?page=1&dtype=recent
# 수정 필요
def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy_heatmap(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def accuracy(output, target, thr_PCK, thr_PCKh, dataset, hm_type='gaussian', threshold=0.5):
    idx = list(range(output.shape[1]))
    norm = 1.0

    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)

        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx)))
    avg_acc = 0
    cnt = 0
    visible = np.zeros((len(idx)))

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]])
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1
            visible[i] = 1
        else:
            acc[i] = 0

    avg_acc = avg_acc / cnt if cnt != 0 else 0

    if cnt != 0:
        acc[0] = avg_acc

    # PCKh
    PCKh = np.zeros((len(idx)))
    avg_PCKh = 0
    headLength = np.linalg.norm(target[0, 1, :] - target[0, 2, :])

    for i in range(len(idx)):
        PCKh[i] = dist_acc(dists[idx[i]], thr_PCKh * headLength)
        if PCKh[i] >= 0:
            avg_PCKh = avg_PCKh + PCKh[i]
        else:
            PCKh[i] = 0

    avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

    if cnt != 0:
        PCKh[0] = avg_PCKh

    # PCK
    PCK = np.zeros((len(idx)))
    avg_PCK = 0

    torso = np.linalg.norm(target[0, 3, :] - target[0, 10, :])  # 여기 0,3,:, 0,10,: 이라고 되어있었음

    for i in range(len(idx)):
        PCK[i] = dist_acc(dists[idx[i]], thr_PCK * torso)

        if PCK[i] >= 0:
            avg_PCK = avg_PCK + PCK[i]
        else:
            PCK[i] = 0

    avg_PCK = avg_PCK / cnt if cnt != 0 else 0

    if cnt != 0:
        PCK[0] = avg_PCK
        # print("*****")
        # print(PCK)
        # print("*****")

    return acc, PCK, PCKh, cnt, pred, visible


# https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/core/loss.py#L15:7
class JointsRMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsRMSELoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        target_coord = target[:, :, :2]
        target_weight = target[:, :, 2].unsqueeze(-1)

        loss = self.criterion(pred, target_coord)
        if self.use_target_weight:
            loss *= target_weight

        loss = torch.sqrt(torch.mean(torch.mean(loss, dim=0)))
        return loss


class OffsetMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(OffsetMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss_hm = 0
        loss_offset = 0
        num_joints = output.size(1) // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()

            if self.use_target_weight:
                loss_hm += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_x_pred,
                    heatmap_gt * offset_x_gt
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_y_pred,
                    heatmap_gt * offset_y_gt
                )

        return loss_hm / num_joints, loss_offset / num_joints


class OffsetL1Loss(nn.Module):
    def __init__(self, use_target_weight, reduction='mean'):
        super(OffsetL1Loss, self).__init__()
        self.criterion = nn.SmoothL1Loss(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.reduction = reduction

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss_hm = 0
        loss_offset = 0
        num_joints = output.size(1) // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()

            if self.use_target_weight:
                loss_hm += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_x_pred,
                    heatmap_gt * offset_x_gt
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_y_pred,
                    heatmap_gt * offset_y_gt
                )
        if self.reduction == 'mean':
            return loss_hm / num_joints, loss_offset / num_joints
        else:
            return loss_hm, loss_offset


class HeatmapMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class HeatmapOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=8):
        super(HeatmapOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.sum(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

# https://github.com/DeLightCMU/PSA
# Polarized Self-Attention
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out


# https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/models/pose_hrnet.py
# HRNET

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.PSA = PSA_s(planes, planes)  # Added PSA module
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.PSA(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)  # output -> 256 channel, resolution은 그대로

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
            # expansion -> basic : 1, bottleneck : 4
        ]  # 48, 96
        self.transition1 = self._make_transition_layer([256], num_channels)  # 여기서 branch가 갈라짐
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)
        if cfg["MODEL"]["TARGET_TYPE"] == "offset":
            factor = 3
        else:
            factor = 1

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'] * factor,
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):  # stage2 : [256], [48, 96]
        num_branches_cur = len(num_channels_cur_layer)  # 2
        num_branches_pre = len(num_channels_pre_layer)  # 1

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],  # 256
                                num_channels_cur_layer[i],  # 48
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            print('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])
    else:
        model.load_state_dict(torch.load(cfg['MODEL']['PRETRAINED']))

    return model

# 모델 yaml 파일 필요
import yaml
def model_define(yaml_path, train=True):
  with open(yaml_path) as f:
    cfg = yaml.load(f)
    # cfg['MODEL']['PRETRAINED']=False
  model = get_pose_net(cfg, train)
  return model

KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)


# 수정 필요
def calc_coord_loss(pred, gt):
    batch_size = gt.size(0)
    valid_mask = gt[:, :, -1].view(batch_size, -1, 1)
    gt = gt[:, :, :2]
    return torch.mean(torch.sum(torch.abs(pred - gt) * valid_mask, dim=-1))


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

    yaml_path = os.path.join(cfg.main_dir, 'Result/Config', yaml_name)
    #   model = model_define(yaml_path, cfg.init_training)
    model = model_define(os.path.join(cfg.drive_dir, 'Config', yaml_name))
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
        #  A.ShiftScaleRotate(),
        A.Resize(384, 288, p=1),
        #    A.HorizontalFlip(p=0.5),
        #    ToTensorV2(p=1),
    ], keypoint_params=A.KeypointParams(format='xy'))

    transform_val = A.Compose([
        # A.Rotate(),
        A.Resize(384, 288, p=1),
        #    ToTensorV2(p=1),
    ], keypoint_params=A.KeypointParams(format='xy'))

    # augmentations only contains methods about color
    augmentations = A.Compose([
        # A.Rotate(),
        A.GaussianBlur(p=0.5),
        A.ChannelShuffle(p=0.5),
        #   A.MotionBlur(p=0.5),
        A.RGBShift(p=0.5)
    ], keypoint_params=A.KeypointParams(format='xy'))

    train_dataset = CowData(cfg=cfg, transform=transform_train, augmentations=augmentations)
    val_dataset = CowData(cfg=cfg, transform=transform_val, mode='val')
    train_dl = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dl = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

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
                train_bar.set_description(f"Train Epoch {epoch + 1}")

                optimizer.zero_grad()
                images, targ_coords = sample['image'].type(torch.FloatTensor).to(device), sample['keypoints'].type(
                    torch.FloatTensor).to(device)
                target, target_weight = sample['target'].type(torch.FloatTensor).to(device), sample[
                    'target_weight'].type(torch.FloatTensor).to(device)
                #   print(target.shape)
                model.train()
                with torch.set_grad_enabled(True):
                    preds = model(images)

                    if cfg.target_type == "offset":
                        loss_hm, loss_os = main_criterion(preds, target, target_weight)
                        loss = loss_hm + loss_os
                    elif cfg.target_type == "gaussian":
                        loss = main_criterion(preds, target, target_weight)

                    if cfg.target_type == "offset":
                        heatmap_height = preds.shape[2]
                        heatmap_width = preds.shape[3]
                        pred_coords = get_final_preds(cfg, preds.detach().cpu().numpy())
                    elif cfg.target_type == 'gaussian':
                        heatmap_height = preds.shape[2]
                        heatmap_width = preds.shape[3]
                        pred_coords, _ = get_max_preds(preds.detach().cpu().numpy())
                        pred_coords[:, :, 0] = pred_coords[:, :, 0] / (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
                        pred_coords[:, :, 1] = pred_coords[:, :, 1] / (heatmap_height - 1.0) * (
                                    4 * heatmap_height - 1.0)

                    pred_coords = torch.tensor(pred_coords).float().to(device)
                    coord_loss = calc_coord_loss(pred_coords, targ_coords)

                    rmse_loss = rmse_criterion(pred_coords, targ_coords)
                    #   _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                    #                                    target.detach().cpu().numpy()[:, ::3, :, :])
                    avg_acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(preds.detach().cpu().numpy(),
                                                                              target.detach().cpu().numpy(), 0.35, 0.5,
                                                                              None)

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
                valid_bar.set_description(f"Valid Epoch {epoch + 1}")

                images, targ_coords = sample['image'].type(torch.FloatTensor).to(device), sample['keypoints'].type(
                    torch.FloatTensor).to(device)
                target, target_weight = sample['target'].type(torch.FloatTensor).to(device), sample[
                    'target_weight'].type(torch.FloatTensor).to(device)

                model.eval()
                with torch.no_grad():
                    preds = model(images)
                    if cfg.target_type == "offset":
                        loss_hm, loss_os = main_criterion(preds, target, target_weight)
                        loss = loss_hm + loss_os
                    elif cfg.target_type == "gaussian":
                        loss = main_criterion(preds, target, target_weight)

                    pred_coords = get_final_preds(cfg, preds.detach().cpu().numpy())
                    pred_coords = torch.tensor(pred_coords).float().to(device)
                    coord_loss = calc_coord_loss(pred_coords, targ_coords)

                    rmse_loss = rmse_criterion(pred_coords, targ_coords)
                    #   _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                    #                                    target.detach().cpu().numpy()[:, ::3, :, :])
                    avg_acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(preds.detach().cpu().numpy(),
                                                                              target.detach().cpu().numpy(), 0.35, 0.5,
                                                                              None)
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


            best_model = model
            save_dir = os.path.join(cfg.main_dir, cfg.save_folder)
            save_name = f'third_model.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            print(f"Valid Loss: {valid_total:.8f}\nBest Model saved.")
            best_loss = valid_total
        # if best_acc <= valid_acc:
        #     best_model = model
        #     save_dir = os.path.join(cfg.main_dir, cfg.save_folder)
        #     save_name = f'best_model2_{valid_acc}.pth'
        #     torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        #     print(f"Valid Acc: {valid_acc:.8f}\nBest Model saved.")
        #     best_acc = valid_acc

    return best_model


def test(cfg):
    # for reporduction
    seed = cfg.seed
    torch.cuda.empty_cache()
    seed_everything(seed)

    # device type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define model
    if cfg.target_type == 'gaussian':
        yaml_name = "heatmap_train2.yaml"

    yaml_path = '/Result/Config/heatmap_train2.yaml'
    model = model_define(yaml_path)
    model = model.to(device)
    model.load_state_dict(torch.load(
        '/Result/weights/cy_model.pth', map_location=device))

    transform_test = A.Compose([
        A.Resize(384, 288),
        # ToTensorV2(p=1),
    ])

    test_ds = CowData(cfg, transform=transform_test, mode='test')
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    annot_list = []

    start_ts = time.time()
    model.eval()
    i = 0
    for sample in tqdm(test_dl):
        images = sample['image'].type(torch.FloatTensor).to(device)
        img_org_heights, img_org_widths = sample['image_shape']
        img_path = sample['image_path']
        # print(img_path)
        # print(img_org_heights, img_org_widths)
        # print(img_path)

        preds = model(images)

        pred_coords = get_final_preds(cfg, preds.detach().cpu().numpy())
        pred_coords = pred_coords.astype(np.float32)
        pred_coords = pred_coords.squeeze(0)
        pred_kpts = get_kpts(cfg, pred_coords, img_org_heights, img_org_widths)
        # print(pred_kpts)

        img_fn = img_path[0].split('/')[-1]
        annot = {
            "ID": i + 1,
            "img_path": img_fn,
            "joint_self": pred_kpts
        }

        annot_list.append(annot)
        i += 1

    latency = time.time() - start_ts
    submission_dict = {
        "latency": latency,
        "annotations": annot_list
    }

    with open('/Result/submission_3.json', 'w') as f:
        json.dump(submission_dict, f)

    return 1


if __name__ == "__main__":
    cfg = SingleModelConfig()
    test(cfg)
