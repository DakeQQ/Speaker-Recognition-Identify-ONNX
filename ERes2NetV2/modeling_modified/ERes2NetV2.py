# Copyright (c) Alibaba, Inc. and its affiliates.
"""
    To further improve the short-duration feature extraction capability of ERes2Net,
    we expand the channel dimension within each stage. However, this modification also
    increases the number of model parameters and computational complexity.
    To alleviate this problem, we propose an improved ERes2NetV2 by pruning redundant structures,
    ultimately reducing both the model parameters and its computational cost.
"""

import math
import os
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

import modelscope.models.audio.sv.pooling_layers as pooling_layers
from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.fusion import AFF
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class BasicBlockERes2NetV2(nn.Module):

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 baseWidth=26,
                 scale=2,
                 expansion=2):
        super(BasicBlockERes2NetV2, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.width = width
        self.conv1 = nn.Conv2d(
            in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2NetV2AFF(nn.Module):

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 baseWidth=26,
                 scale=2,
                 expansion=2):
        super(BasicBlockERes2NetV2AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.width = width
        self.conv1 = nn.Conv2d(
            in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width, r=4))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2NetV2(nn.Module):

    def __init__(self,
                 block=BasicBlockERes2NetV2,
                 block_fuse=BasicBlockERes2NetV2AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=64,
                 feat_dim=80,
                 embed_dim=192,
                 baseWidth=26,
                 scale=2,
                 expansion=2,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(ERes2NetV2, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.baseWidth = baseWidth
        self.scale = scale
        self.expansion = expansion

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block_fuse, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            block_fuse, m_channels * 8, num_blocks[3], stride=2)

        # Downsampling module
        self.layer3_ds = nn.Conv2d(
            m_channels * 4 * self.expansion,
            m_channels * 8 * self.expansion,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False)

        # Bottom-up fusion module
        self.fuse34 = AFF(channels=m_channels * 8 * self.expansion, r=4)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == 'TSDP' else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * self.expansion)
        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                    expansion=self.expansion))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        fuse_out34 = self.fuse34(out4, out3_ds)
        stats = self.pool(fuse_out34)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


@MODELS.register_module(
    Tasks.speaker_verification, module_name=Models.eres2netv2_sv)
class SpeakerVerificationERes2NetV2(TorchModel):
    r"""ERes2NetV2 architecture with local and global feature fusion. ERes2NetV2 is mainly composed
    of Bottom-up Dual-stage Feature Fusion (BDFF) and Bottleneck-like Local Feature Fusion (BLFF).
    BDFF fuses multi-scale feature maps in bottom-up pathway to obtain global information.
    The BLFF extracts localization-preserved speaker features and strengthen the local information interaction.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.embed_dim = self.model_config['embed_dim']
        self.baseWidth = self.model_config['baseWidth']
        self.scale = self.model_config['scale']
        self.expansion = self.model_config['expansion']
        self.other_config = kwargs
        self.feature_dim = 80
        self.device = create_device(self.other_config['device'])

        self.embedding_model = ERes2NetV2(
            embed_dim=self.embed_dim,
            baseWidth=self.baseWidth,
            scale=self.scale,
            expansion=self.expansion)

        pretrained_model_name = kwargs['pretrained_model']
        self.__load_check_point(pretrained_model_name)

        self.embedding_model.to(self.device)
        self.embedding_model.eval()

    def forward(self, audio):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        assert len(
            audio.shape
        ) == 2, 'modelscope error: the shape of input audio to model needs to be [N, T]'
        # audio shape: [N, T]
        feature = self.__extract_feature(audio)
        embedding = self.embedding_model(feature.to(self.device))

        return embedding.detach().cpu()

    def __extract_feature(self, audio):
        feature = Kaldi.fbank(audio, num_mel_bins=self.feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        return feature

    def __load_check_point(self, pretrained_model_name, device=None):
        if not device:
            device = torch.device('cpu')
        self.embedding_model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_model_name),
                map_location=device),
            strict=True)
