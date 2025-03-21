import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=2, dropout_p=0.3):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Embedding layer for the source (English)
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        # GRU layer: outputs shape [batch_size, seq_len, hidden_size]
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        self.init_weights()
    
    def init_weights(self):
        # Initialize embedding weights using Xavier
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_seq, hidden):
        # input_seq: [batch_size, seq_len]
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, embed_size]
        output, hidden = self.gru(embedded, hidden)  # output: [batch_size, seq_len, hidden_size]
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class DecoderGRU(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=2, dropout_p=0.3):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Embedding layer for the target (French)
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        # GRU layer: processes one token at a time
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        # Linear layer to project hidden state to vocabulary size
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # Initialize GRU parameters
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_seq, hidden):
        # input_seq: [batch_size, 1] – one token per time step
        embedded = self.embedding(input_seq)  # [batch_size, 1, embed_size]
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)  # output: [batch_size, 1, hidden_size]
        output = self.dropout(output)
        output = self.out(output.squeeze(1))  # [batch_size, output_size]
        output = F.log_softmax(output, dim=1)
        return output, hidden
