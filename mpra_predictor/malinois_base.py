# -----------------------------------------------------------------------------
# This code contains parts of the original boda2 code (Modified)
# Modified by Joseph Finkelberg (Boston University), 2025
#
# This file contains a modified version of code from the boda2 repository,
# originally developed by Sagar Gosai, Rodrigo Castro, and contributors.
#
# Copyright (c) 2025 Sagar Gosai, Rodrigo Castro
#
# Licensed under the MIT License. See the full license in the boda2 repository:
# https://github.com/sjgosai/boda2/blob/main/LICENSE
# -----------------------------------------------------------------------------
from pathlib import Path
import math
import torch
from torch import nn


def load_pretrained_weights(
    model_path: Path,
    model_cls: nn.Module,
    freeze: bool = True,
    unfreeze_layers: list[str] = None,
    device: str = 'cpu',
):
    checkpoint = torch.load(model_path / 'torch_checkpoint.pt', weights_only=False)
    model = model_cls(**vars(checkpoint['model_hparams']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    if unfreeze_layers:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
    model.eval()
    return model


def get_padding(k):
    left = (k - 1) // 2
    right = k - 1 - left
    return [left, right]


class Conv1dNorm(nn.Module):
    def __init__(self, in_c, out_c, k, bn=True, wn=False):
        super().__init__()
        conv = nn.Conv1d(in_c, out_c, k)
        self.conv = nn.utils.weight_norm(conv) if wn else conv
        self.bn_layer = nn.BatchNorm1d(out_c) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn_layer(x) if self.bn_layer else x


class LinearNorm(nn.Module):
    def __init__(self, in_f, out_f, bn=True, wn=False):
        super().__init__()
        lin = nn.Linear(in_f, out_f)
        self.linear = nn.utils.weight_norm(lin) if wn else lin
        self.bn_layer = nn.BatchNorm1d(out_f) if bn else None

    def forward(self, x):
        x = self.linear(x)
        return self.bn_layer(x) if self.bn_layer else x


class GroupedLinear(nn.Module):
    def __init__(self, in_group, out_group, groups):
        super().__init__()
        self.in_group_size = in_group
        self.out_group_size = out_group
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group, out_group))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        x = torch.bmm(x, self.weight) + self.bias
        x = x.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)
        return x


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(self, in_f, hid_f, out_f, branches, layers, act='ReLU', dropout=0.5):
        super().__init__()
        self.intake = RepeatLayer(1, branches)
        self.n_layers = layers
        self.nonlin = getattr(nn, act)()
        self.dropout = nn.Dropout(dropout)

        for i in range(layers):
            inp = in_f if i == 0 else hid_f
            outp = out_f if i == layers - 1 else hid_f
            setattr(self, f'branched_layer_{i+1}', GroupedLinear(inp, outp, branches))

    def forward(self, x):
        x = self.intake(x)
        for i in range(self.n_layers - 1):
            x = self.dropout(self.nonlin(getattr(self, f'branched_layer_{i+1}')(x)))
        return getattr(self, f'branched_layer_{self.n_layers}')(x)


class BassetBranched(nn.Module):
    def __init__(
        self, input_len=600, conv1_channels=300, conv1_kernel_size=19,
        conv2_channels=200, conv2_kernel_size=11, conv3_channels=200,
        conv3_kernel_size=7, n_linear_layers=1, linear_channels=1000,
        linear_activation='ReLU', linear_dropout_p=0.11625456877954289,
        n_branched_layers=3, branched_channels=140, branched_activation='ReLU',
        branched_dropout_p=0.5757068086404574, n_outputs=3, use_batch_norm=True,
        use_weight_norm=False, loss_criterion='L1KLmixed', loss_args=None
    ):
        super().__init__()
        self.input_len = input_len
        self.n_linear_layers = n_linear_layers
        self.n_outputs = n_outputs
        self.linear_activation = linear_activation

        self.pad1 = nn.ConstantPad1d(get_padding(conv1_kernel_size), 0)
        self.conv1 = Conv1dNorm(4, conv1_channels, conv1_kernel_size, use_batch_norm, use_weight_norm)

        self.pad2 = nn.ConstantPad1d(get_padding(conv2_kernel_size), 0)
        self.conv2 = Conv1dNorm(conv1_channels, conv2_channels, conv2_kernel_size, use_batch_norm, use_weight_norm)

        self.pad3 = nn.ConstantPad1d(get_padding(conv3_kernel_size), 0)
        self.conv3 = Conv1dNorm(conv2_channels, conv3_channels, conv3_kernel_size, use_batch_norm, use_weight_norm)

        self.pad4 = nn.ConstantPad1d((1, 1), 0)
        self.maxpool_3 = nn.MaxPool1d(3)
        self.maxpool_4 = nn.MaxPool1d(4)

        self.flat_factor = self._compute_flatten_factor(input_len)
        in_linear = conv3_channels * self.flat_factor

        for i in range(n_linear_layers):
            setattr(self, f'linear{i+1}', LinearNorm(in_linear, linear_channels, use_batch_norm, use_weight_norm))
            in_linear = linear_channels

        self.nonlin = getattr(nn, linear_activation)()
        self.dropout = nn.Dropout(linear_dropout_p)

        self.branched = BranchedLinear(
            in_linear, branched_channels, branched_channels,
            branches=n_outputs, layers=n_branched_layers,
            act=branched_activation, dropout=branched_dropout_p
        )

        self.output = GroupedLinear(branched_channels, 1, n_outputs)

    def _compute_flatten_factor(self, L):
        return ((L // 3) // 4 + 2) // 4

    def encode(self, x):
        x = self.nonlin(self.conv1(self.pad1(x)))
        x = self.maxpool_3(x)
        x = self.nonlin(self.conv2(self.pad2(x)))
        x = self.maxpool_4(x)
        x = self.nonlin(self.conv3(self.pad3(x)))
        x = self.maxpool_4(self.pad4(x))
        x = torch.flatten(x, start_dim=1)
        return x

    def decode(self, x):
        for lin_idx in range(self.n_linear_layers):
            lin = getattr(self, f'linear{lin_idx + 1}')
            x = self.dropout(self.nonlin(lin(x)))
        return self.branched(x)

    def classify(self, x):
        return self.output(x)

    def forward(self, x):
        return self.classify(self.decode(self.encode(x)))


class MalinoisEncoder(BassetBranched):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = 200
        self.out_length = 13

    def __str__(self):
        return f"MalBase:{self.out_channels}-{self.out_length}"

    def forward(self, x):
        x = self.nonlin(self.conv1(self.pad1(x)))
        x = self.maxpool_3(x)
        x = self.nonlin(self.conv2(self.pad2(x)))
        x = self.maxpool_4(x)
        x = self.nonlin(self.conv3(self.pad3(x)))
        x = self.maxpool_4(self.pad4(x))
        return x  # (200, 13)
