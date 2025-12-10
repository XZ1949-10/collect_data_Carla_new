#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu Sep 20 10:48:02 2018
Info:
'''

import os

import numpy as np
import matplotlib.pyplot as plt

DataLoaderTest = True

if __name__ == '__main__':
    if DataLoaderTest:
        print("======CarlaH5Data test start======")
        from carla_loader import CarlaH5Data
        from torchvision.utils import make_grid

        base_path = r"D:\carla_cil_pytorch\dataset\AgentHuman"
        data = CarlaH5Data(
            train_folder=os.path.join(base_path, "SeqTrain"),
            eval_folder=os.path.join(base_path, "SeqVal"),
            batch_size=128,
            num_workers=10)

        train_loader = data.loaders["train"]
        eval_loader = data.loaders["eval"]

        print(len(train_loader))
        print(len(eval_loader))

        for i, (img, speed, _, _) in enumerate(train_loader):
            show_img = make_grid(img)
            plt.imshow((np.transpose(
                show_img.numpy(),
                (1, 2, 0))*255).astype(np.uint8))
            plt.show()
            #print(one_hot)
            if i == 60:
                break

        # for i, (img, speed, command, predict) in enumerate(eval_loader):
        #     print(img.size())
        #     if i == 15:
        #         break
