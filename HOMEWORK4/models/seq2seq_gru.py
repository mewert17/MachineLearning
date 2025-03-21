import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=2, dropout_p=0.3):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        
        # GRU layer
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, 
                         batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Initialize embedding weights
        nn.init.xavier_uniform_(self.embedding.weight)
        # Set padding tokens to zero
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
    
    def forward(self, input_seq, hidden):
        # input_seq: [batch_size, seq_len]
        # Convert input to embeddings
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, embed_size]
        
        # Pass embeddings through GRU
        output, hidden = self.gru(embedded, hidden)
        # output: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden state
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class DecoderGRU(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=2, dropout_p=0.3):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        
        # GRU layer
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, 
                         batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        
        # Output projection
        self.out = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Initialize embedding weights
        nn.init.xavier_uniform_(self.embedding.weight)
        # Set padding tokens to zero
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        # Initialize output layer
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, input_seq, hidden):
        # input_seq: [batch_size, 1] (single token per time step)
        
        # Convert input to embeddings
        embedded = self.embedding(input_seq)  # [batch_size, 1, embed_size]
        embedded = self.dropout(embedded)
        
        # Pass embeddings through GRU
        output, hidden = self.gru(embedded, hidden)
        # output: [batch_size, 1, hidden_size]
        
        # Apply dropout to output
        output = self.dropout(output)
        
        # Project output to vocabulary size
        output = self.out(output.squeeze(1))  # [batch_size, output_size]
        output = F.log_softmax(output, dim=1)
        
        return output, hidden