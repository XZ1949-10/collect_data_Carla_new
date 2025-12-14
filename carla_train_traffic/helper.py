#!/usr/bin/env python
# coding=utf-8
'''
辅助函数
'''

import os
import random
import shutil

import torch


class TransWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img):
        return self.seq.augment_image(img)


class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, id_, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join("save_models", "{}_best.pth".format(id_))
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        shutil.copyfile(filename, best_path)
