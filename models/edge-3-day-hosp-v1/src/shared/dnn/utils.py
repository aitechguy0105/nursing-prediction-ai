import numpy as np 
import torch
import copy


def get_mask_from_sizes(sizes, device='cpu'): 
    # sizes has batch_size elements.  Return a mask of shape batch_size x max_length
    # with 1's and 0's.  
    batch_size = len(sizes)
    max_length = max(sizes)
    mask = np.zeros((max_length, batch_size), dtype=np.float32)
    for i, size in enumerate(sizes): 
        mask[:size, i] = 1
    mask = torch.from_numpy(mask)
    mask = mask.to(device)
    return mask

