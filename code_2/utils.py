import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List


class SingleModelConfig:
    def __init__(self,
                 input_size: List[int] = [384, 288],
                 kpd: float = 4.0,
                 epochs: int = 15,
                 sigma: float = 3.0,
                 num_joints: int = 17,
                 batch_size: int = 16,
                 random_seed: int = 42,
                 test_ratio: float = 0.2,
                 learning_rate: float = 1e-4,
                 main_dir: str = '/content',
                 drive_dir: str = 'drive/MyDrive/AnimalDatathon',
                 save_folder: str = 'Model',  # 추후 추가

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
        self.output_size = self.image_size//4
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
        colors = [cmap(i) for i in np.linspace(0, 1,  self.num_joints + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        self.joint_colors = {k: colors[k] for k in range(self.num_joints)}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
