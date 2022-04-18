import cv2

from neko_2020nocr.dan.common.common import flatten_label_idx
from torch.nn import functional as trnf
from torch_scatter import scatter_mean
import numpy as np
# mk5 CF branch dropped predict-sample-predict support.
# A GP branch will be added if it's ever to be supported
# force breaking the context by random masking with temporal masks
import torch
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_2020nocr.dan.utils import Loss_counter

def debugva(nim):
    a=(nim.permute(1,2,0).reshape(32,128,3).detach().cpu().numpy()*255).astype(np.uint8);
    return a;

def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask
# we cannot afford messing around with inaccurate localization.
