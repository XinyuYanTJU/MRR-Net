#!/usr/bin/env python 
#-- coding: utf-8 -- 
#@Time : 2020/10/10 下午7:49 
#@Author : YXY
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
from math import sqrt
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Unfold):
            pass
        elif isinstance(m, GELU):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

    def initialize(self):
        weight_init(self)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('pths/resnet50_a1h2_176-001a1197.pth'), strict=False)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class dilatedComConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(dilatedComConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0],dilation=1)
        self.conv2_2 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1],dilation=2)
        self.conv2_3 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2],dilation=3)
        self.conv2_4 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3],dilation=4)

    def forward(self, x):
        conv2_1 = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        conv2_3 = self.conv2_3(x)
        conv2_4 = self.conv2_4(x)
        return torch.cat((conv2_1, conv2_2, conv2_3, conv2_4), dim=1)

    def initialize(self):
        weight_init(self)

class LSCE(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(LSCE, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            dilatedComConv4(inplanes // reduction1, inplanes // reduction1),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)

    def initialize(self):
        weight_init(self)

LayerNorm = partial(nn.InstanceNorm2d, affine = True)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.reduction_ratio = reduction_ratio

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:] # h:64,w:64
        heads, r = self.heads, self.reduction_ratio # heads:1 r:8
        # to_qkv:Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) # x:[1,32,64,64]->[1,96,64,64]->3*[1,32,64,64]
        # k, v = map(lambda t: reduce(t, 'b c (h r1) (w r2) -> b c h w', 'mean', r1 = r, r2 = r), (k, v))
        # k,v : [1,32,64,64] -> [1,32,8,8]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
        # 分为multi-head 此时head数为1
        # q:[1,32,64,64] -> [1,4096,32]
        # k:[1,32,8,8] -> [1,64,32]
        # v:[1,32,8,8] -> [1,64,32]
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # q*k: [1,4096,32]*[1,64,32]->[1,4096,64]
        attn = sim.softmax(dim = -1)
        # attention:[1,4096,64]
        out = einsum('b i j, b j d -> b i d', attn, v)
        # attn*v: [1,4096,64]*[1,64,32]->[1,4096,32]
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        # out: [1,4096,32]->[1,32,64,64]
        # to_out:Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # output:[1,32,64,64]->[1,32,64,64]
        return self.to_out(out)
    def initialize(self):
        weight_init(self)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            GELU(),
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)
        )

        # MixFeedForward(
        #  Sequential(
        # (0): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (2): GELU()
        # (3): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        # )
        # )

    def forward(self, x):
        return self.net(x)
    def initialize(self):
        weight_init(self)

class GSSE4(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(GSSE4, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(4608, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class GSSE3(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(GSSE3, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(2304, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class GSSE2(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(GSSE2, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(1152, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class GSSE1(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(GSSE1, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(576, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        # x = self.SelfAttention(self.LN(x)) + x
        # x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

class MergeGlobal(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergeGlobal, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self,  global_context,global_context_max):
        x = torch.cat((global_context,global_context_max), dim=1)
        x = self.features(x)
        return x

    # def forward(self, local_context, global_context):
    #     x = torch.cat((local_context, global_context), dim=1)
    #     x = self.features(x)
    #     return x

    def initialize(self):
        weight_init(self)

class MergeLocalGlobal(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergeLocalGlobal, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x

    # def forward(self, local_context, global_context):
    #     x = torch.cat((local_context, global_context), dim=1)
    #     x = self.features(x)
    #     return x

    def initialize(self):
        weight_init(self)

class VFMRM4(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(VFMRM4, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.LSCE = LSCE(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.GSSE = GSSE4(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.LSCE(x),  self.GSSE(x))
        return x

    def initialize(self):
        weight_init(self)

class VFMRM3(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(VFMRM3, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.LSCE = LSCE(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.GSSE = GSSE3(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.LSCE(x),  self.GSSE(x))
        return x

    def initialize(self):
        weight_init(self)

class VFMRM2(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(VFMRM2, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.LSCE = LSCE(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.GSSE = GSSE2(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.LSCE(x),  self.GSSE(x))
        return x

    def initialize(self):
        weight_init(self)

class VFMRM1(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(VFMRM1, self).__init__()

        out_size_local_context = int(inplanes/4)

        out_size_global_max_context = int(inplanes/4)

        self.LSCE = LSCE(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.GSSE = GSSE1(inplanes, out_size_local_context, BatchNorm, reduction1=4)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_max_context, planes, BatchNorm)
    def forward(self, x):

        x = self.merge_context(self.LSCE(x),  self.GSSE(x))
        return x

    def initialize(self):
        weight_init(self)

class CRB_SWRM3(nn.Module):
    def __init__(self):
        super(CRB_SWRM3, self).__init__()


        self.above_conv2 = nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1)
        self.above_bn2 = nn.BatchNorm2d(64)

        self.right_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.right_bn1   = nn.BatchNorm2d(64)

        self.fuse_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fuse_bn   = nn.BatchNorm2d(64)


    def forward(self, backboneabove, right):

        above_2 = F.relu(self.above_bn2(self.above_conv2(backboneabove)), inplace=True)

        right = F.interpolate(right, size=backboneabove.size()[2:], mode='bilinear')
        right_1 = F.relu(self.right_bn1(self.right_conv1(right )), inplace=True)


        fuse2 = above_2 * right_1

        fuse = F.relu(self.fuse_bn(self.fuse_conv(fuse2 )), inplace=True)+right_1

        return  fuse

    def initialize(self):
        weight_init(self)

class CRB_SWRM2(nn.Module):
    def __init__(self):
        super(CRB_SWRM2, self).__init__()

        self.above_conv2 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.above_bn2 = nn.BatchNorm2d(64)

        self.right_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.right_bn1 = nn.BatchNorm2d(64)

        self.fuse_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fuse_bn = nn.BatchNorm2d(64)

    def forward(self, backboneabove, right):
        above_2 = F.relu(self.above_bn2(self.above_conv2(backboneabove)), inplace=True)

        right = F.interpolate(right, size=backboneabove.size()[2:], mode='bilinear')
        right_1 = F.relu(self.right_bn1(self.right_conv1(right)), inplace=True)

        fuse2 = above_2 * right_1

        fuse = F.relu(self.fuse_bn(self.fuse_conv(fuse2)), inplace=True) + right_1

        return fuse

    def initialize(self):
        weight_init(self)

class CRB_SWRM1(nn.Module):
    def __init__(self):
        super(CRB_SWRM1, self).__init__()

        self.above_conv2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.above_bn2 = nn.BatchNorm2d(64)

        self.right_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.right_bn1 = nn.BatchNorm2d(64)

        self.fuse_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fuse_bn = nn.BatchNorm2d(64)

    def forward(self, backboneabove, right):
        above_2 = F.relu(self.above_bn2(self.above_conv2(backboneabove)), inplace=True)

        right = F.interpolate(right, size=backboneabove.size()[2:], mode='bilinear')
        right_1 = F.relu(self.right_bn1(self.right_conv1(right)), inplace=True)

        fuse2 = above_2 * right_1

        fuse = F.relu(self.fuse_bn(self.fuse_conv(fuse2)), inplace=True) + right_1

        return fuse

    def initialize(self):
        weight_init(self)

class MRR_Net_ResNet50(nn.Module):
    def __init__(self, cfg):

        super(MRR_Net_ResNet50, self).__init__()
        self.cfg      = cfg
        self.bkbone = ResNet()
        self.VFMRM4 = VFMRM4(2048, 64, nn.BatchNorm2d)
        self.CRB_SWRM3 = CRB_SWRM3()
        self.CRB_SWRM2 = CRB_SWRM2()
        self.CRB_SWRM1 = CRB_SWRM1()
        self.linearrpred = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        if cfg.mode == 'train':
            self.VFMRM3 = VFMRM3(1024, 64, nn.BatchNorm2d)
            self.VFMRM2 = VFMRM2(512, 64, nn.BatchNorm2d)
            self.VFMRM1 = VFMRM1(256, 64, nn.BatchNorm2d)
            self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
            self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
            self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
            self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()


    def forward(self, x, shape=None):
        if self.cfg.mode == 'train':
            shape = x.size()[2:] if shape is None else shape
            out1_bk, out2_bk, out3_bk, out4_bk = self.bkbone(x)
            VFMRM4_feature = self.VFMRM4(out4_bk)
            VFMRM3_feature = self.VFMRM3(out3_bk)
            VFMRM2_feature = self.VFMRM2(out2_bk)
            VFMRM1_feature = self.VFMRM1(out1_bk)
            SWRM3_out = self.CRB_SWRM3(out3_bk,VFMRM4_feature)
            SWRM2_out = self.CRB_SWRM2(out2_bk,SWRM3_out)
            SWRM1_out = self.CRB_SWRM1(out1_bk,SWRM2_out)
            VFMRM1_out = F.interpolate(self.linearr1(VFMRM1_feature), size=shape, mode='bilinear')
            VFMRM2_out = F.interpolate(self.linearr2(VFMRM2_feature), size=shape, mode='bilinear')
            VFMRM3_out = F.interpolate(self.linearr3(VFMRM3_feature), size=shape, mode='bilinear')
            VFMRM4_out = F.interpolate(self.linearr4(VFMRM4_feature), size=shape, mode='bilinear')
            pred = F.interpolate(self.linearrpred(SWRM1_out), size=shape, mode='bilinear')
            return pred, VFMRM1_out, VFMRM2_out, VFMRM3_out, VFMRM4_out
        else:
            shape = x.size()[2:] if shape is None else shape
            out1_bk, out2_bk, out3_bk, out4_bk = self.bkbone(x)
            VFMRM4_feature = self.VFMRM4(out4_bk)
            SWRM3_out = self.CRB_SWRM3(out3_bk, VFMRM4_feature)
            SWRM2_out = self.CRB_SWRM2(out2_bk, SWRM3_out)
            SWRM1_out = self.CRB_SWRM1(out1_bk, SWRM2_out)
            pred = F.interpolate(self.linearrpred(SWRM1_out), size=shape, mode='bilinear')
            return pred

    def initialize(self):
        if self.cfg.mode == 'test':
            param = {}
            model_dict = self.state_dict()
            checkpoint = torch.load(self.cfg.snapshot, map_location='cpu')
            for k, v in checkpoint.items():
                if k in model_dict:
                    param[k] = v
            model_dict.update(param)
            self.load_state_dict(model_dict)
        else:
            weight_init(self)
