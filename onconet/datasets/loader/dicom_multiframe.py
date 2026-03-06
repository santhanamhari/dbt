import numpy as np
import pydicom
import torch

def load_multiframe_dicom(path, *, force_monochrome=True):
    """
    Load a SINGLE multi-frame DICOM (npz) and return a 3D volume.

    Returns:
        vol: np.ndarray float32 (T, H, W)
        meta: dict
    """

    #obj = torch.load(path, map_location="cpu") 
    obj = np.load(path)
    arr = obj['volume']
    return arr


def normalize_minmax(vol, eps=1e-6):
    """
    Min-max normalize a (T,H,W) volume to [0,1].
    """
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    #vmin = float(torch.min(vol))
    #vmax = float(torch.max(vol))
    return (vol - vmin) / (vmax - vmin + eps)
