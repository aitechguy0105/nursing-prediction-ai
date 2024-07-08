import argparse
import os
import sys
import copy
import math
import logging

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

def setup_logging(log_file): 
    logging.basicConfig(
        level=logging.DEBUG, 
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )                    

def train_epoch(model, optimizer, criterion, data, 
                device, print_every=10, max_iter=100): 
    
    total_num_iter = 0
    cumulative_loss = 0.    

    model.train()
    with torch.set_grad_enabled(True): 
        for batch_i, (Y, X_fixed, X_codes, X_emar, X_pn, sizes) in enumerate(data):  
            min_size = np.min(np.array(sizes))

            Y = Y.to(device)
            X_fixed = X_fixed.to(device)
            X_codes = (X_codes[0].to(device), X_codes[1].to(device))
            X_emar = (X_emar[0].to(device), X_emar[1].to(device))
            X_pn = (X_pn[0].to(device), X_pn[1].to(device))

            batch_size = Y.size()[1]
            num_days = sum(sizes) 
            optimizer.zero_grad()

            mask = utils.get_mask_from_sizes(sizes, device)

            logits, probs = model(X_fixed, X_codes, X_emar, X_pn, sizes, batch_size)
            loss = criterion(logits, Y)  # Loss is a Tensor of size (max_length, batch_size)...  
            masked_loss = loss * mask  
            scalar_loss = torch.sum(masked_loss) / torch.sum(mask)
            scalar_loss.backward()     
            optimizer.step()    

            cumulative_loss += scalar_loss.item() 
            total_num_iter += 1          

            if np.isnan(cumulative_loss): 
                logging.error(f"Caught error!  {batch_i}" + " "*20)
                exit()

            if (batch_i+1) % print_every == 0: 
                loss_right_now = cumulative_loss / total_num_iter
                percent_complete = 100 * (batch_i+1) / len(data)
                print_str = f"\rBatch {batch_i+1} / {len(data)} ({percent_complete:.1f}%) => {loss_right_now:.5f}"
                print_str = print_str + (" " * (80 - len(print_str)))
                print(print_str, end='\r')
                cumulative_loss = 0.
                total_num_iter = 0

    print()
    return cumulative_loss


def evaluate(model, criterion, data, device): 

    Y_all = None
    Yhat_all = None
    total_num_iter = 0
    cumulative_loss = 0.  

    model.eval()
    with torch.set_grad_enabled(False): 
        for batch_i, (Y, X_fixed, X_codes, X_emar, X_pn, sizes) in enumerate(data): 
            Y = Y.to(device)
            X_fixed = X_fixed.to(device)
            X_codes = (X_codes[0].to(device), X_codes[1].to(device))
            X_emar = (X_emar[0].to(device), X_emar[1].to(device))
            X_pn = (X_pn[0].to(device), X_pn[1].to(device))

            batch_size = Y.size()[1] # dims are time x batch
            num_days = sum(sizes)
            
            mask = utils.get_mask_from_sizes(sizes, device)
            
            logits, probs = model(X_fixed, X_codes, X_emar, X_pn, sizes, batch_size)
            loss = criterion(logits, Y) 
            masked_loss = loss * mask  
            scalar_loss = torch.sum(masked_loss) / torch.sum(mask)            

            # Accumulate loss
            cumulative_loss += scalar_loss.item() 
            total_num_iter += 1
            
            # Accumulate Y and Yhat given sizes (i.e., disregarding mask...)
            Y_batch = Y.detach().cpu().numpy()
            Yhat_batch = probs.detach().cpu().numpy()

            if Y_all is None: 
                Y_list, Yhat_list = [], []
            else: 
                Y_list, Yhat_list = [Y_all], [Yhat_all]
                
            for idx in range(Y_batch.shape[1]): 
                Y_list.append(Y_batch[:sizes[idx], idx])
                Yhat_list.append(Yhat_batch[:sizes[idx], idx])
            Y_all = np.concatenate(Y_list)
            Yhat_all = np.concatenate(Yhat_list)

    data_loss = cumulative_loss / total_num_iter
    auroc = roc_auc_score(Y_all, Yhat_all)
    ap = average_precision_score(Y_all, Yhat_all)
    return data_loss, auroc, ap, Y_all, Yhat_all

def get_data_loaders(data_root_dir, max_length, batch_size, num_data_workers): 
    train_dataset = rnn_dataset.SaivaDataset(os.path.join(data_root_dir, 'train'), 
                                             max_length=max_length)
    val_dataset = rnn_dataset.SaivaDataset(os.path.join(data_root_dir, 'val'), 
                                           max_length=max_length)
    test_dataset = rnn_dataset.SaivaDataset(os.path.join(data_root_dir, 'test'),
                                            max_length=max_length)

    train_data = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size, 
                                             num_workers=num_data_workers, 
                                             shuffle=True, 
                                             collate_fn=data_loader_collate_fn)
    val_data = torch.utils.data.DataLoader(val_dataset, 
                                           batch_size=batch_size, 
                                           num_workers=num_data_workers, 
                                           collate_fn=data_loader_collate_fn)
    test_data = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            num_workers=num_data_workers, 
                                            collate_fn=data_loader_collate_fn)    

    return train_data, val_data, test_data


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
    # Dataset and data loader parameters
    parser.add_argument("--data_root_dir", 
                        type=str, 
                        default=os.path.join(root_dir, "data/processed/rnn/data"))
    parser.add_argument('--num_data_workers', 
                        type=int, 
                        default=4)
    parser.add_argument("--max_length", 
                        type=int, 
                        default=100)

    # Training hyperparameters
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=64)
    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=100)
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.0001) 
    parser.add_argument("--pos_weight", 
                        type=float, 
                        help='weight on pos samples', 
                        default=5.0) 
    parser.add_argument("--which_metric", 
                        type=str, 
                        help='one of: loss, roc, ap', 
                        default='roc') 
    parser.add_argument('--patience', 
                        type=int, 
                        help='how many epochs w/o increasing which_metric before stopping',
                        default=5) # Need to be higher?  5 sometimes exits a bit early...   

    # RNN hyperparameters                        
    parser.add_argument('--rnn_dim', 
                        type=int, 
                        default=200)
    parser.add_argument('--rnn_layers', 
                        type=int, 
                        default=1)
    parser.add_argument('--learn_h0', 
                        action='store_true', 
                        help='Learning initial hidden state of RNN',
                        default=False)

    # Embedding hyperparameters
    parser.add_argument("--random_embed", 
                        default=False, 
                        help='Randomly init embeddings instead of using pretrained',
                        action='store_true')
    parser.add_argument("--embed_dim", 
                        type=int, 
                        default=100) 
    parser.add_argument('--embed_mode', 
                        type=str, 
                        help='How to aggregate embeddings - max or mean',
                        default='mean')                        
    parser.add_argument('--ffn', 
                        action='store_true', 
                        default=False, 
                        help='Add dense layers between embeddings and RNN')


    # Save params
    parser.add_argument("--save_dir", 
                        type=str, 
                        default=os.path.join(root_dir, "data/processed/rnn/results/baseline"))
    parser.add_argument("--model_file", 
                        type=str, 
                        default="saved_model.pt")
    parser.add_argument("--log_file", 
                        type=str, 
                        default="log.txt")
    FLAGS = parser.parse_args()
    
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    log_file = os.path.join(FLAGS.save_dir, FLAGS.log_file)
    setup_logging(log_file)

    train_data, val_data, test_data = get_data_loaders(FLAGS.data_root_dir, 
                                                       FLAGS.max_length, 
                                                       FLAGS.batch_size, 
                                                       FLAGS.num_data_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Using device {device}")

    fixed_size = 83
    if os.path.isdir('/data'): 
        embeddings_dir = '/data/processed/rnn'
    else: 
        embeddings_dir = '/home/ubuntu/saivahc/models/infinity-3-day-hosp/data/processed/rnn'

    if FLAGS.random_embed: 
        codes_file = emar_file = pn_file = None
    else: 
        codes_file = os.path.join(embeddings_dir, 'code_embeddings.npy')
        emar_file = os.path.join(embeddings_dir, 'emar_word_embeddings.npy')
        pn_file = os.path.join(embeddings_dir, 'pn_word_embeddings.npy')
    model = BasicModel(fixed_size=fixed_size, 
                       codes_file=codes_file, 
                       emar_file=emar_file, 
                       pn_file=pn_file, 
                       embed_dim=FLAGS.embed_dim, 
                       embed_mode=FLAGS.embed_mode, 
                       rnn_dim=FLAGS.rnn_dim, 
                       rnn_layers=FLAGS.rnn_layers, 
                       learn_h0=FLAGS.learn_h0, 
                       ffn=FLAGS.ffn)
    model.to(device)

    # Set up objective function and optimizer. 
    pos_weight = torch.Tensor([FLAGS.pos_weight]) # Do we want to do this?
    pos_weight = pos_weight.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none', 
                                           pos_weight=pos_weight) 
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)     

    best_i = 0
    best_weights = None
    best_metric = np.Inf if FLAGS.which_metric == 'loss' else -np.Inf
    epochs_since_improvement = 0
    for epoch_i in range(FLAGS.num_epochs): 
        
        train_loss = train_epoch(model, optimizer, criterion, 
                                 train_data, device, print_every=5)
        
        val_loss, val_auroc, val_ap, Y, Yhat = evaluate(model, criterion, 
                                                        val_data, device)
        stats = {'loss': val_loss, 
                 'roc': val_auroc, 
                 'ap': val_ap}
        logging.debug(f"Epoch {epoch_i+1} val loss: {val_loss:.4f} AUROC: {val_auroc:.4f} aP: {val_ap:.4f}")

        if _compare_stats(stats, FLAGS.which_metric, best_metric): 
            best_metric = stats[FLAGS.which_metric]
            best_i = epoch_i + 1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else: 
            epochs_since_improvement += 1
        if epochs_since_improvement == FLAGS.patience: 
            break     
    
    logging.debug(f"Stopped training at epoch {epoch_i+1}")

    # Revert to best weights, and get test set perf.
    model.load_state_dict(best_weights)
    # Verify best model by re-running on val... 
    val_loss, val_auroc, val_ap, Y_val, Yhat_val = evaluate(model, criterion, 
                                                            val_data, device)
    logging.debug(f"Val loss: {val_loss:.4f} AUROC: {val_auroc:.4f} aP: {val_ap:.4f}")

    # Now see how we do on test... 
    test_loss, test_auroc, test_ap, Y_test, Yhat_test = evaluate(model, criterion, 
                                                                 test_data, device)
    logging.debug(f"Test loss: {test_loss:.4f} AUROC: {test_auroc:.4f} aP: {test_ap:.4f}")

    # Save model, Y, Yhat.  
    logging.debug(f"Saving model and outputs to {FLAGS.save_dir}...")
    model_filepath = os.path.join(FLAGS.save_dir, FLAGS.model_file)
    torch.save(best_weights, model_filepath)
    Y_test_filepath = os.path.join(FLAGS.save_dir, 'Y_test.npy')
    np.save(Y_test_filepath, Y_test)
    Yhat_test_filepath = os.path.join(FLAGS.save_dir, 'Yhat_test.npy')
    np.save(Yhat_test_filepath, Yhat_test)

    exit()



    
    