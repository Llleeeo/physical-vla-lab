"""Normalize temporal loss weights over valid labels; no channel reweighting."""
import numpy as np


def temporal_weights(padding, factor, prefix=4):
    pad=np.asarray(padding)
    if pad.ndim!=2 or pad.dtype!=np.bool_ or pad.shape[1]!=20:
        raise ValueError('Expected boolean Bx20 padding')
    if type(prefix) is not int or not 1<=prefix<=20 or not np.isfinite(factor) or factor<=0:
        raise ValueError('Invalid temporal weights')
    weights=np.ones(pad.shape,dtype=np.float32);weights[:,:prefix]=factor;weights[pad]=0
    if not weights.sum()>0:raise ValueError('No valid supervision')
    return weights


def prefix_loss(losses,padding,factor):
    import torch
    if losses.ndim!=3 or losses.shape[-2:]!=(20,32) or padding.shape!=losses.shape[:2] or padding.dtype!=torch.bool:
        raise ValueError('Expected Bx20x32 losses and boolean mask')
    weights=torch.as_tensor(temporal_weights(padding.detach().cpu().numpy(),factor),device=losses.device,dtype=losses.dtype)
    valid=~padding
    # Boolean indexing excludes padding before multiplication (including NaNs).
    return (losses[...,:29][valid]*weights[valid].unsqueeze(-1)).sum()/(29*weights.sum())
