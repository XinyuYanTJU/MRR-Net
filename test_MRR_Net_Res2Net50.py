#!/usr/bin/env python 
#-- coding: utf-8 -- 
#@Time : 2020/10/10 下午8:48 
#@Author : YXY 
#@File : test.py

import os
import sys
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from MRR_Net_Res2Net50 import MRR_Net_Res2Net50
from thop import profile
from data_util import Config,Data

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

if __name__=='__main__':
    model_pth = 'pths/MRR-Net-Res2Net-50.pth'
    test_paths = ['data/CAMO/','data/NC4K/','data/COD10K/']
    for path in test_paths:
        cfg = Config(datapath=path, snapshot=model_pth, mode='test')
        data = Data(cfg)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)
        net = MRR_Net_Res2Net50(cfg)

        net.train(False)
        net.cuda()
        dataset_name = path.split('/')[-2]
        with torch.no_grad():
            test_time = AverageMeter()
            input = torch.randn(1, 3, 384, 384).cuda()
            flops, params = profile(net, inputs=([input]))
            print(str(flops / 1e9) + 'G')
            print(str(params / 1e6) + 'M')
            for image, mask, shape, name in loader:
                image = image.cuda().float()
                begin = time.time()
                pred = net(image, shape)
                test_time.update(time.time() - begin)
                pred = (torch.sigmoid(pred[0, 0]) * 255).cpu().numpy()
                directory = 'results/' + model_pth.split('/')[-1].replace('.pth','') +'/'+ dataset_name
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(directory + '/' + name[0] + '.png', np.round(pred))
            fps = 1 / test_time.avg
            print("count:", test_time.count, "test_time:", test_time.avg, "fps:", fps)
