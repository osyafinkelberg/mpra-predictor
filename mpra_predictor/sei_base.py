# -----------------------------------------------------------------------------
# This code contains parts of the original SEI Model code (Modified)
# Modified by Joseph Finkelberg (Boston University), 2025
#
# This file contains a modified version of the SEI model architecture,
# originally developed by The Trustees of Princeton University, The Simons Foundation, Inc.,
# and The University of Texas Southwestern Medical Center.
#
# Copyright (c) 2021 The Trustees of Princeton University, The Simons Foundation, Inc.,
# and The University of Texas Southwestern Medical Center. All rights reserved.
#
# Licensed for academic and research use only, per the original SEI license.
# See the full license in the SEI repository for details:
# https://github.com/FunctionLab/sei-framework/blob/main/LICENSE.txt
# -----------------------------------------------------------------------------
from pathlib import Path
import numpy as np
from scipy.interpolate import splev
import torch
import torch.nn as nn


def load_pretrained_weights(
    weights_path: Path,
    model_cls: nn.Module,
    freeze: bool = True,
    unfreeze_layers: list[str] = None,
    device: str = 'cpu',
) -> nn.Module:
    state_dict = torch.load(weights_path, map_location=torch.device(device))
    new_state_dict = {k.replace("module.model.", ""): v for k, v in state_dict.items()}
    model = model_cls()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    if unfreeze_layers:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
    model.eval()
    return model.to(device)


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.
    """
    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * (2 * (dist > 0) - 1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True
        )).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)
        ).float()


class BSplineTransformation(nn.Module):
    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        return torch.matmul(input, self._spline_tr)


class SeiWithoutClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 960
        self.out_length = 16

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, stride=1, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, stride=1, padding=4)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True)
        )
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, stride=1, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, stride=1, padding=4)
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True)
        )
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, stride=1, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, stride=1, padding=4)
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True)
        )
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, stride=1, padding=8, dilation=4),
            nn.ReLU(inplace=True)
        )
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, stride=1, padding=16, dilation=8),
            nn.ReLU(inplace=True)
        )
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, stride=1, padding=32, dilation=16),
            nn.ReLU(inplace=True)
        )
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, stride=1, padding=50, dilation=25),
            nn.ReLU(inplace=True)
        )
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(16, scaled=False)
        )

    def __str__(self):
        return f"SeiBase:{self.out_channels}-{self.out_length}"

    def forward(self, x):
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)
        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)
        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        spline_out = self.spline_tr(out)
        return spline_out


class SeiWithoutSpline(SeiWithoutClassifier):
    def __init__(self):
        super().__init__()
        self.flank_size = 75
        self.out_channels = 960
        self.out_length = 256 - 2 * self.flank_size

    def forward(self, x):
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)
        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)
        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        out = cat_out3 + dconv_out4
        return out[..., self.flank_size: self.flank_size + self.out_length]
