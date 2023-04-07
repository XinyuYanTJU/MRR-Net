#!/usr/bin/env python 
# -- coding: utf-8 --
# @Time : 2020/10/10 下午8:18
# @Author : YXY
# @File : train.py
import os
import sys
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
parentdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
from MRR_Net_ResNet50 import MRR_Net_ResNet50
from tensorboardX import SummaryWriter
from data_util import Config, Data
from apex import amp

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss_without_weit(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == '__main__':
    nets_name = 'MRR_Net_ResNet50'
    cfg = Config(datapath='/home/ubuntu/DeepLearning/YXY/Dataset/COD/COD10K_CAMO_CombinedTrainingDataset',
                 savepath='../pths/', mode='train',
                 batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=100, lr_decay_gamma=0.1)
    data = Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)

    net = MRR_Net_ResNet50(cfg)
    save_tensorboard_dir = '../tensorboard_server/'+nets_name+'/'
    if not os.path.exists(save_tensorboard_dir):
        os.makedirs(save_tensorboard_dir)
    save_pth_dir = '../pths/'+nets_name+'/'
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)
    net.train(True)
    net.cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(save_tensorboard_dir)
    global_step = 0
    for epoch in range(cfg.epoch):

        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask) in enumerate(loader):

            image, mask = image.cuda().float(), mask.cuda().float()
            pred, VFMRM1_out, VFMRM2_out, VFMRM3_out, VFMRM4_out = net(image)

            loss_pred = structure_loss(pred, mask)
            loss_VFMRM1_out = structure_loss(VFMRM1_out, mask)
            loss_VFMRM2_out = structure_loss(VFMRM2_out, mask)
            loss_VFMRM3_out = structure_loss(VFMRM3_out, mask)
            loss_VFMRM4_out = structure_loss(VFMRM4_out, mask)

            loss = loss_pred + loss_VFMRM1_out / 2 + loss_VFMRM2_out / 4 + loss_VFMRM3_out / 8 + loss_VFMRM4_out / 16

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            global_step += 1

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss_pred': loss_pred.item(), 'loss_VFMRM1_out': loss_VFMRM1_out.item(),
                                    'loss_VFMRM2_out': loss_VFMRM2_out.item(),
                                    'loss_VFMRM3_out': loss_VFMRM3_out.item(),
                                    'loss_VFMRM4_out': loss_VFMRM4_out.item()}, global_step=global_step)

            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss.item()))
            del loss, scale_loss, loss_pred, loss_VFMRM1_out, loss_VFMRM2_out, loss_VFMRM3_out, loss_VFMRM4_out, pred, VFMRM1_out, VFMRM2_out, VFMRM3_out, VFMRM4_out

        if epoch > cfg.epoch * 4 / 5:
            torch.save(net.state_dict(),save_pth_dir+ 'MRR_Net_ResNet50_' + str(epoch + 1) + '.pth')