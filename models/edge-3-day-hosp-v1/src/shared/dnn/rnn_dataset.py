import os
import numpy as np
import sys
import glob
import pickle as pkl
from pathlib import Path 
import torch 
from torch.utils.data import Dataset

# TODO 
# Pass everything back as tensors if possible... After debugging a bit... 

def _convertForEmbedding(inputs_list): 
    """
       codes_list is a ragged array, with each element a variable length 
       list of arrays of indices.  Return input and offset arrays suitable 
       for use with pytorch's EmbeddingBag module. 
    """
    num_inputs_per_day = np.array([len(el) for el in inputs_list])
    offsets = np.cumsum(num_inputs_per_day)[:-1]
    offsets = np.concatenate([[0], offsets])
    inputs = np.concatenate(inputs_list)
    return (torch.tensor(inputs), torch.tensor(offsets))

def _pad_to_length(inputs, offsets, target_length): 
    """
       Inputs and offsets to target length. For each missing day, add a 0 to input, 
       and increment a ptr in offsets.  
    """
    length = offsets.size()[0]
    if length < target_length: 
        num_to_add = target_length - length
        new_inputs = torch.cat([inputs, torch.tensor([0]*num_to_add)])
        start = inputs.size()[-1]
        new_offsets = torch.cat([offsets, torch.tensor(np.arange(start, start+num_to_add,1))])
        return new_inputs, new_offsets
    else: 
        return inputs, offsets

def _merge_for_embedding(batch, idx, max_length): 
    """
       batch is [(Y0, X1), (Y1, X2), ...]
       - Each Yn is a list of target values.  These get turned into tensors with dimensions
         (max_length x batch_size)
       - Each Xn is a tuple of (X_fixed, X_codes, X_emar, X_pn). 
       - X_fixed is a tensor of fixed length inputs, one per timestep.  This gets padded to 
         max_length, and all turned into a tensor with dims (max_length, batch_size, fixed_length)
       - X_codes, X_emar, X_pn are each tuples of inputs, offsets suitable for input to EmbeddingBag. 

       This function takes an index into the X of each tuple in the batch (designating X_codes,
       X_emar, or X_pn), and creates a single tuple of appropriately padded to max_length, input
       and offset tensors suitable for use with EmbeddingBag for the entire batch. 
    """

    # Each element is a tuple with Y in first pos, X in second pos. 
    # Each X is a tuple of fixed, codes, emar, and pn data. idx indexes this.  
    # Each specific X tuple is itself a tuple of inputs and offsets.  
    inputs_list = [b_i[1][idx][0] for b_i in batch]
    offsets_list = [b_i[1][idx][1] for b_i in batch]
    
    # Pad to max_length if necessary
    new_data = [_pad_to_length(inputs, outputs, max_length) for inputs, outputs in zip(inputs_list, offsets_list)]
    inputs_list = [el[0] for el in new_data]
    offsets_list = [el[1] for el in new_data]
    
    # Concatenate inputs
    new_inputs = torch.cat(inputs_list)
    
    num_inputs = np.array([input.size()[0] for input in inputs_list])
    cumsums = np.cumsum(num_inputs)
    new_offsets_list = [offsets_list[0]] + [(offsets + int(cumsums[i])) for i, offsets in enumerate(offsets_list[1:]) ]    
    new_offsets = torch.cat(new_offsets_list)
    
    return new_inputs, new_offsets

def data_loader_collate_fn(batch): 
    """
       batch is [(Y0, X1), (Y1, X2), ...]
       - Each Yn is a list of target values.  These get turned into tensors with dimensions
         (max_length x batch_size).  
       - max_length is the length of the longest Yn (max sequence length for the batch)
       - Each Xn is a tuple of (X_fixed, X_codes, X_emar, X_pn). 
       - X_fixed gets turned into a tensor with dims (max_length, batch_size, fixed_length). 
       - X_codes, X_emar, X_pn are each turned into tuples of inputs, offsets suitable 
         for input to EmbeddingBag. 

       This function takes an index into the X of each tuple in the batch (designating X_codes,
       X_emar, or X_pn), and creates a single tuple of appropriately padded to max_length, input
       and offset tensors suitable for use with EmbeddingBag for the entire batch. 
    """    

    # Get lengths, etc. 
    sizes = [len(b_i[0]) for b_i in batch]
    max_length = max(sizes)
    batch_size = len(batch)

    # Get the fixed length stuff out of the way first.  
    Y_list = [b_i[0] for b_i in batch]
    Y_batch = torch.nn.utils.rnn.pad_sequence(Y_list)  # Pads to max length. 
    X_fixed_list = [b_i[1][0] for b_i in batch]        # Pads to max length.
    X_fixed_batch = torch.nn.utils.rnn.pad_sequence(X_fixed_list)

    # Now deal with the variable length inputs...  We have inputs and offsets for each item in batch.  
    X_codes_batch = _merge_for_embedding(batch, 1, max_length)
    X_emar_batch = _merge_for_embedding(batch, 2, max_length)
    X_pn_batch = _merge_for_embedding(batch, 3, max_length)
    
    # Make sure we return sizes so we can do loss masking later on.  
    return Y_batch, X_fixed_batch, X_codes_batch, X_emar_batch, X_pn_batch, sizes



class SaivaDataset(Dataset): 
    """
       Dataset utility class for use with PyTorch DataLoader
    """


    def __init__(self, root_dir, max_length=100, max_num=None, filter=True): 
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.npz')))
        if os.path.isfile(os.path.join(root_dir, 'bad_cases.pkl')): 
            print(f"Filtering cases starting with {len(self.files)}...")
            with open(os.path.join(root_dir, 'bad_cases.pkl'), 'rb') as f_in: 
                bad_cases = pkl.load(f_in)
            if len(bad_cases) > 0:
                bad_files = set([self.files[i] for i in bad_cases])
                self.files = [f for f in self.files if f not in bad_files]
        self.N = len(self.files)
        if max_num is not None:  # Use this to create toy datasets we can overfit to... 
          self.files = self.files[:max_num]
          self.N = max_num
        self.max_length = max_length
        print(f"Loaded {self.N} from {root_dir}")

    def __len__(self): 
        return self.N            
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx): 
            idx = idx.tolist()[0]
        
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)
        Y = data['Y']
        X_fixed = data['X_fixed']
        X_codes = data['X_codes']
        X_emar = data['X_emar']
        X_pn = data['X_pn']

        # Truncate if necessary.  note that we always take from 
        # end; can sample randomly too.  
        seq_length = Y.shape[0]
        if seq_length > self.max_length: 
            start_idx = seq_length - self.max_length
            Y = Y[start_idx:]
            X_fixed = X_fixed[start_idx:]
            X_codes = X_codes[start_idx:]
            X_emar = X_emar[start_idx:]
            X_pn = X_pn[start_idx:]
        
        # Convert X_codes, X_emar, X_pn into inputs and offsets arrays... 
        Y = torch.tensor(Y)
        X_fixed = torch.tensor(X_fixed)
        X_codes = _convertForEmbedding(X_codes)
        X_emar = _convertForEmbedding(X_emar)
        X_pn = _convertForEmbedding(X_pn)

        return (Y, (X_fixed, X_codes, X_emar, X_pn))


        

    