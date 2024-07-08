import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F


def shape_embeddings_for_batch(embeddings, max_length, batch_size, embed_dim): 
    """
       embeddings is the result of calling EmbeddingBag layer with inputs and 
       offsets for a given batch of size batch_size, each padded to max_length.  
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retval = torch.zeros(max_length, batch_size, embed_dim, device=device)
    for i in range(batch_size): 
        start_idx = i*max_length
        end_idx = start_idx + max_length
        retval[:,i] = embeddings[start_idx:end_idx]
    retval = retval.contiguous() # Prob not necessary... 
    return retval

def _embeddings_from_file(filepath): 
    embeddings = torch.Tensor(np.load(filepath))
    print(f"Loaded embeddings from {filepath}: {embeddings.size()}")
    return embeddings

class BasicModel(nn.Module):
    def __init__(self, fixed_size=83, codes_file=None, 
                 emar_file=None, pn_file=None, 
                 code_size=4786, emar_size=26847, 
                 pn_size=97811, embed_dim=100, 
                 embed_mode='mean', 
                 rnn_dim=200, 
                 rnn_layers=1, 
                 learn_h0=False, 
                 ffn=False): 
        super().__init__()
        self.fixed_size = fixed_size
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers 
        self.embed_mode = embed_mode
        self.fixed_size = fixed_size
        self.learn_h0 = learn_h0
        self.ffn = ffn

        if codes_file is not None: 
            code_embeddings = _embeddings_from_file(codes_file)
            self.code_embedding_bag = nn.EmbeddingBag.from_pretrained(code_embeddings, 
                                                                      freeze=False, 
                                                                      mode=embed_mode) # Max?  
        else: 
            self.code_embedding_bag = nn.EmbeddingBag(code_size, embed_dim, mode=embed_mode)

        if emar_file is not None: 
            emar_embeddings = _embeddings_from_file(emar_file)
            self.emar_embedding_bag = nn.EmbeddingBag.from_pretrained(emar_embeddings, 
                                                                      freeze=False, 
                                                                      mode=embed_mode) # Max?  
        else: 
            self.emar_embedding_bag = nn.EmbeddingBag(emar_size, embed_dim, mode=embed_mode)

        if pn_file is not None: 
            pn_embeddings = _embeddings_from_file(pn_file)                                                                  
            self.pn_embedding_bag = nn.EmbeddingBag.from_pretrained(pn_embeddings, 
                                                                    freeze=False, 
                                                                    mode=embed_mode) # Max?  
        else: 
            self.pn_embedding_bag = nn.EmbeddingBag(pn_size, embed_dim, mode=embed_mode)

        self.embed_dim = self.code_embedding_bag.weight.size()[1]
        self.rnn_dim = rnn_dim

        # Create init hidden state with batch size 1; later we'll repeat it. 
        if self.learn_h0:
            self.rnn_h0 = torch.nn.Parameter(torch.zeros(self.rnn_layers, 1, self.rnn_dim), 
                                             requires_grad=True)

        # What about dense layers for an ffn before feeding into the RNN?  Does that help?  
        if self.ffn: 
            # Set up ffn with nonlinearities before feeding into RNN.  Just do 2 layers, PReLU or ReLU. 
            self.ffn_dense_1 = torch.nn.Linear(self.embed_dim*3 + self.fixed_size, 200)
            self.ffn_relu_1 = torch.nn.ReLU()
            self.drop_1 = torch.nn.Dropout(p=0.2)
            self.ffn_dense_2 = torch.nn.Linear(200, 200)
            self.ffn_relu_2 = torch.nn.ReLU()

        # Merge
        if self.ffn: 
            rnn_input_size = 200
        else:
            rnn_input_size = self.embed_dim * 3 + self.fixed_size
        self.rnn = nn.GRU(input_size=rnn_input_size, 
                          hidden_size=self.rnn_dim, 
                          num_layers=rnn_layers) # dropout between RNN layers?  
        self.dense_1 = nn.Linear(self.rnn_dim, 1)
        self.out_prob = nn.Sigmoid()  
        #self.output = output
        self.init_weights()

    def init_weights(self):
        # Do we need to do anything here?...  
        pass

    def forward(self, X_fixed, X_codes, X_emar, X_pn, sizes, batch_size):
        max_length = max(sizes)

        # Set up hidden state - use 0's for now.  But eventually make it learnable.  
        if not self.learn_h0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            rnn_init_hidden = torch.zeros((self.rnn_layers, batch_size, self.rnn_dim))
            rnn_init_hidden = rnn_init_hidden.to(device)
        
        code_embeddings = self.code_embedding_bag(X_codes[0], X_codes[1])
        code_embeddings = shape_embeddings_for_batch(code_embeddings, 
                                                     max_length, 
                                                     batch_size, 
                                                     self.embed_dim)

        emar_embeddings = self.emar_embedding_bag(X_emar[0], X_emar[1])
        emar_embeddings = shape_embeddings_for_batch(emar_embeddings, 
                                                     max_length, 
                                                     batch_size, 
                                                     self.embed_dim)

        pn_embeddings = self.pn_embedding_bag(X_pn[0], X_pn[1])
        pn_embeddings = shape_embeddings_for_batch(pn_embeddings, 
                                                   max_length, 
                                                   batch_size, 
                                                   self.embed_dim)

        embeddings = torch.cat([X_fixed, 
                               code_embeddings, 
                               emar_embeddings, 
                               pn_embeddings], axis=2)

        if self.ffn: 
            embeddings = self.ffn_relu_2(
                self.ffn_dense_2( # Dropout? Doesn't make much if any diff; bigger problem is probably using ReLUs here... 
                    self.ffn_relu_1(
                        self.ffn_dense_1(embeddings)
                    )
                )
            )
        
        if self.learn_h0:
            h0 = self.rnn_h0.repeat(1, batch_size, 1)
            # h0 = self.rnn_h0.expand(self.rnn_h0.size()[0], 
            #                         batch_size, 
            #                         self.rnn_h0.size()[2])
            rnn_output, rnn_hidden = self.rnn(embeddings, h0)
        else: 
            rnn_output, rnn_hidden = self.rnn(embeddings, rnn_init_hidden)

        # Note - we discard rnn_hidden; we default to 0 for initial hidden state.  Arguably, 
        # we can take advantage of this to deal with longer sequences, but these models don't 
        # do particularly well at that anyway so meh... 

        logits = torch.squeeze(self.dense_1(rnn_output))
        probs = torch.squeeze(self.out_prob(logits))

        return logits, probs