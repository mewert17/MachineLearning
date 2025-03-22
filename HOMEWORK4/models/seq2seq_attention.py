import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=2, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout_p if num_layers>1 else 0)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad(): self.embedding.weight[0].fill_(0)
        for name, p in self.gru.named_parameters():
            if 'weight' in name: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=device)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # hidden: [num_layers, batch, hidden]; use last layer
        dec = hidden[-1].unsqueeze(2)                          # [batch, hidden, 1]
        scores = torch.bmm(encoder_outputs, dec).squeeze(2)    # [batch, seq_len]
        weights = torch.softmax(scores, dim=1)                 # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden]
        return context, weights


class DecoderGRUWithAttention(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=2, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout_p if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad(): self.embedding.weight[0].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for name, p in self.gru.named_parameters():
            if 'weight' in name: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)

    def forward(self, input_step, hidden, encoder_outputs):
        # input_step: [batch,1]
        embedded = self.dropout(self.embedding(input_step))
        context, _ = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        output = torch.cat([output.squeeze(1), context], dim=1)
        return F.log_softmax(self.fc(output), dim=1), hidden, None
