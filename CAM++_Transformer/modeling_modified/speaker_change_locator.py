# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.DTDNN import CAMPPlus
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, n_units, h=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.n_units = n_units
        self.d_k = n_units // h
        self.scale = float(1.0 / math.sqrt(self.d_k))
        self.h = h

    def forward(self, x, batch_size):
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1)) * self.scale
        x = torch.matmul(F.softmax(scores, dim=3), v.transpose(1, 2))
        return self.linearO(x.transpose(1, 2).reshape(-1, self.n_units))


class PositionwiseFeedForward(nn.Module):

    def __init__(self, n_units, d_units, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(n_units, d_units)
        self.linear2 = nn.Linear(d_units, n_units)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class PosEncoding(nn.Module):

    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array([[
            pos / np.power(10000, 2.0 * (j // 2) / d_word_vec)
            for j in range(d_word_vec)
        ] for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        self.pos_enc = torch.nn.Embedding(max_seq_len + 1, d_word_vec)
        self.pos_enc.weight = torch.nn.Parameter(
            torch.from_numpy(pos_enc), requires_grad=False)
        self.arrange = torch.arange(1, self.pos_enc.num_embeddings, 1, dtype=torch.int16).repeat(2, 1)

    def forward(self, input_len):
        input_pos = self.arrange[:, :input_len].int()
        return self.pos_enc(input_pos)


class TransformerEncoder(nn.Module):

    def __init__(self,
                 idim,
                 n_units=256,
                 n_layers=2,
                 e_units=512,
                 h=4,
                 dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format('lnorm1_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('self_att_', i),
                    MultiHeadSelfAttention(n_units, h))
            setattr(self, '{}{:d}'.format('lnorm2_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('ff_', i),
                    PositionwiseFeedForward(n_units, e_units, dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x, num_frames):
        num, _, dim = x.size()
        e = self.linear_in(x.reshape(-1, dim))
        for i in range(self.n_layers):
            e = getattr(self, '{}{:d}'.format('lnorm1_', i))(e)
            s = getattr(self, '{}{:d}'.format('self_att_', i))(e, num)
            e += s
            e = getattr(self, '{}{:d}'.format('lnorm2_', i))(e)
            s = getattr(self, '{}{:d}'.format('ff_', i))(e)
            e += s
        return self.lnorm_out(e).reshape(num, num_frames, -1)


class TransformerEncoder_out(nn.Module):

    def __init__(self,
                 idim,
                 n_units=256,
                 n_layers=2,
                 e_units=512,
                 h=4,
                 dropout=0.1):
        super(TransformerEncoder_out, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format('lnorm1_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('self_att_', i),
                    MultiHeadSelfAttention(n_units, h))
            setattr(self, '{}{:d}'.format('lnorm2_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('ff_', i),
                    PositionwiseFeedForward(n_units, e_units, dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x, num_frames):
        e = self.linear_in(x)
        for i in range(self.n_layers):
            e = getattr(self, '{}{:d}'.format('lnorm1_', i))(e)
            s = getattr(self, '{}{:d}'.format('self_att_', i))(e, x.shape[0])
            e += s
            e = getattr(self, '{}{:d}'.format('lnorm2_', i))(e)
            s = getattr(self, '{}{:d}'.format('ff_', i))(e)
            e += s
        return self.lnorm_out(e).reshape(num_frames, 2, -1)


class OutLayer(nn.Module):

    def __init__(self, n_units=256, num_anchors=2):
        super(OutLayer, self).__init__()
        self.combine = TransformerEncoder_out(num_anchors * n_units, n_units)
        self.out_linear = nn.Linear(n_units // num_anchors, 1)

    def forward(self, input, num_frames):
        return self.out_linear(self.combine(input.transpose(0, 1).reshape(1, num_frames, -1), num_frames))


class TransformerDetector(nn.Module):

    def __init__(self,
                 frame_dim=512,
                 anchor_dim=192,
                 hidden_dim=256,
                 max_seq_len=1000):
        super(TransformerDetector, self).__init__()
        self.detection = TransformerEncoder(
            idim=frame_dim + anchor_dim, n_units=hidden_dim)
        self.output = OutLayer(n_units=hidden_dim)
        self.pos_enc = PosEncoding(max_seq_len, hidden_dim)

    def forward(self, feats, anchors, num_frames):
        feats = feats.repeat(2, 1, 1)
        anchors = anchors.repeat(1, num_frames, 1)
        sd_in = torch.cat((feats, anchors), dim=-1)
        return self.output(self.detection(sd_in, num_frames) + self.pos_enc_plus(num_frames), num_frames)


@MODELS.register_module(Tasks.speaker_diarization, module_name=Models.scl_sd)
class SpeakerChangeLocatorTransformer(TorchModel):
    r"""A speaekr change locator using the transformer architecture as the backbone.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config

        self.feature_dim = self.model_config['fbank_dim']
        frame_size = self.model_config['frame_size']
        anchor_size = self.model_config['anchor_size']
        self.device = create_device(kwargs['device'])

        self.encoder = CAMPPlus(self.feature_dim, output_level='frame')
        self.backend = TransformerDetector(
            frame_dim=frame_size, anchor_dim=anchor_size)

        pretrained_encoder = kwargs['pretrained_encoder']
        pretrained_backend = kwargs['pretrained_backend']

        self.__load_check_point(pretrained_encoder, pretrained_backend)

        self.encoder.to(self.device)
        self.backend.to(self.device)
        self.encoder.eval()
        self.backend.eval()

    def forward(self, feature, anchors):
        frame_state = self.encoder(feature)
        output = self.backend(frame_state, anchors, frame_state.shape[1])
        output = output.sigmoid().squeeze(-1)
        output = output.repeat_interleave(2, dim=0)
        return output

    def __extract_feature(self, audio):
        feature = Kaldi.fbank(audio, num_mel_bins=self.feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        return feature

    def __load_check_point(
        self,
        pretrained_encoder,
        pretrained_backend,
    ):
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_encoder),
                map_location=torch.device('cpu')))

        self.backend.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_backend),
                map_location=torch.device('cpu')))
