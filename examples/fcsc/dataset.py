#!/usr/bin/env python

import glob
import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc
import skimage.io
from sklearn.model_selection import train_test_split

import fcn


class FCSCDataset(chainer.dataset.DatasetMixin):

    class_names = [
        'background',# 0
        'ball', # 1
        'circle_box', # 2
        'rectangle_box', # 3
        'dumbbell_green', # 4
        'dumbbell_red', # 5
        'hammer', # 6
        'misumi_frame', # 7
        'scissors',# 8
        'screw_driver',# 9
        'towel',# 10
    ]

    def __init__(self, split):
        assert split in ('train', 'val')

        dataset_dir = '/home/leus/haraduka/20180307_imagedataset/raw/dataset' # you should change it

        sub_dirs = []
        for sub_dir in os.listdir(dataset_dir):
            sub_dir = osp.join(dataset_dir, sub_dir)
            sub_dirs.append(sub_dir)

        seed = np.random.RandomState(1234)
        sub_dirs_train, sub_dirs_val = train_test_split(
            sub_dirs, test_size=0.2, random_state=seed)
        self._sub_dirs = sub_dirs_train if split == 'train' else sub_dirs_val

    def __len__(self):
        return len(self._sub_dirs)

    def get_example(self, i):
        sub_dir = self._sub_dirs[i]

        img_file = osp.join(sub_dir, 'image.jpg')
        lbl_file = osp.join(sub_dir, 'label.npz')

        img = skimage.io.imread(img_file)
        lbl = np.load(lbl_file)['arr_0']

        return img, lbl


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import six
    import cv2
    dataset = FCSCDataset('train')
    for i in six.moves.range(len(dataset)):
        img, lbl = dataset.get_example(i)
        viz = fcn.utils.label2rgb(lbl, img, label_names=dataset.class_names)
        cv2.imshow(__file__, viz[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break
