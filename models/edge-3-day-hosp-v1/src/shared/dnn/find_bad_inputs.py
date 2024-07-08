import argparse
import os
import sys
import copy
import pickle as pkl

import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

if os.path.isdir('/code'): 
    sys.path.append('/code')
else: 
    sys.path.append("/home/ubuntu/saivahc/models/infinity-3-day-hosp/code")

from edge.dnn import rnn_dataset
from edge.dnn.basic_model import BasicModel 
from edge.dnn.rnn_dataset import data_loader_collate_fn
from edge.dnn import utils 


def is_bad_case(model, optimizer, criterion, Y, X, device): 
    
    total_num_iter = 0
    cumulative_loss = 0.    

    X_fixed = X[0]
    X_codes = X[1]
    X_emar = X[2]
    X_pn = X[3]
    model.train()
    with torch.set_grad_enabled(True):
        
        sizes = [Y.size()[0]]
        num_days = sum(sizes)

        Y = Y.to(device)
        X_fixed = X_fixed.to(device)
        X_codes = (X_codes[0].to(device), X_codes[1].to(device))
        X_emar = (X_emar[0].to(device), X_emar[1].to(device))
        X_pn = (X_pn[0].to(device), X_pn[1].to(device))
        
        X_fixed = X_fixed.view(X_fixed.size()[0], 1, X_fixed.size()[1])

        batch_size = 1
        optimizer.zero_grad()

        mask = utils.get_mask_from_sizes(sizes, device)

        logits, probs = model(X_fixed, X_codes, X_emar, X_pn, sizes, batch_size)
        if num_days > 1: 
            loss = criterion(logits, Y)  # Loss is a Tensor of size (max_length, batch_size)...  
            masked_loss = loss * mask  # This is why mask should be on device I think?  
            scalar_loss = torch.sum(masked_loss) / torch.sum(mask)
            scalar_loss.backward()     # Back propagate gradients. 
            optimizer.step()    # Take the step...      

            return True if np.isnan(scalar_loss.item()) else False
        else: 
            return True if np.isnan(logits.item()) else False



def _compare_stats(stats, which_metric, best_metric): 
    if which_metric == 'loss': 
        return stats[which_metric] < best_metric
    else: 
        return stats[which_metric] > best_metric

if os.path.isdir('/data'): 
    # We are in docker container so root_dir is /
    root_dir = '/'
else: 
    root_dir = '/home/ubuntu/saivahc/models/infinity-3-day-hosp'

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", 
                        type=str, 
                        default=os.path.join(root_dir, "data/processed/rnn/data"))
    parser.add_argument("--max_length", 
                        type=int, 
                        default=200)
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1)
    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=1)
    parser.add_argument('--num_data_workers', 
                        type=int, 
                        default=1)
    parser.add_argument('--rnn_dim', 
                        type=int, 
                        default=200)
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.0001) 
    parser.add_argument('--patience', 
                        type=int, 
                        default=5)
    parser.add_argument("--save_dir", 
                        type=str, 
                        default=os.path.join(root_dir, "data/processed/rnn/models"))
    parser.add_argument("--model_file", 
                        type=str, 
                        default="saved_model.pt")
    FLAGS = parser.parse_args()
    
    max_length = FLAGS.max_length
    data_root_dir = FLAGS.data_root_dir
    batch_size = FLAGS.batch_size
    num_data_workers = FLAGS.num_data_workers
    lr = FLAGS.lr
    num_epochs = FLAGS.num_epochs
    rnn_dim = FLAGS.rnn_dim
    patience = FLAGS.patience

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Using device {device}")

    fixed_size = 83
    if os.path.isdir('/data'): 
        embeddings_dir = '/data/processed/rnn'
    else: 
        embeddings_dir = '/home/ubuntu/saivahc/models/infinity-3-day-hosp/data/processed/rnn'

    codes_file = os.path.join(embeddings_dir, 'code_embeddings.npy')
    emar_file = os.path.join(embeddings_dir, 'emar_word_embeddings.npy')
    pn_file = os.path.join(embeddings_dir, 'pn_word_embeddings.npy')

    train_dataset = rnn_dataset.SaivaDataset(os.path.join(data_root_dir, 'train'), 
                                             max_length=max_length, 
                                             filter=False)
    val_dataset = rnn_dataset.SaivaDataset(os.path.join(data_root_dir, 'val'), 
                                           max_length=max_length, 
                                           filter=False)
    test_dataset = rnn_dataset.SaivaDataset(os.path.join(data_root_dir, 'test'),
                                            max_length=max_length, 
                                            filter=False)
    train_N = len(train_dataset)                                         
    val_N = len(val_dataset)
    test_N = len(test_dataset)
    print(f"Testing {train_N} + {val_N} + {test_N} cases...")

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # TEST
    bad_cases = set()
    for idx in np.arange(test_N):
        print(f"At {idx} / {test_N}", end='\r')
        model = BasicModel(fixed_size=fixed_size, 
                           codes_file=codes_file, 
                           emar_file=emar_file, 
                           pn_file=pn_file, 
                           rnn_dim=rnn_dim)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)       
        Y, X = test_dataset[idx]
        if is_bad_case(model, optimizer, criterion, Y, X, device): 
            bad_cases.add(idx)
    with open("/home/ubuntu/test_bad_cases.pkl", "wb") as f_out: 
        pkl.dump(bad_cases, file=f_out)     

    # VAL
    bad_cases = set()
    for idx in np.arange(val_N):
        print(f"At {idx} / {val_N}", end='\r')
        model = BasicModel(fixed_size=fixed_size, 
                           codes_file=codes_file, 
                           emar_file=emar_file, 
                           pn_file=pn_file, 
                           rnn_dim=rnn_dim)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)       
        Y, X = val_dataset[idx]
        if is_bad_case(model, optimizer, criterion, Y, X, device): 
            bad_cases.add(idx)
    with open("/home/ubuntu/val_bad_cases.pkl", "wb") as f_out: 
        pkl.dump(bad_cases, file=f_out)

    # TRAIN
    bad_cases = set()
    for idx in np.arange(train_N):

        print(f"At {idx} / {train_N}", end='\r')
        model = BasicModel(fixed_size=fixed_size, 
                           codes_file=codes_file, 
                           emar_file=emar_file, 
                           pn_file=pn_file, 
                           rnn_dim=rnn_dim)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)       
        Y, X = train_dataset[idx]
        if is_bad_case(model, optimizer, criterion, Y, X, device): 
            bad_cases.add(idx)
            print(f"Adding {idx}!")
    with open("/home/ubuntu/train_bad_cases.pkl", "wb") as f_out: 
        pkl.dump(bad_cases, file=f_out)



   

    exit()



    
    