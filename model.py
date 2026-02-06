import torch
import torch.nn as nn

class OrbitLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=6):
        super(OrbitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class OrbitAutoencoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, seq_len=10):
        super(OrbitAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        _, (hn, _) = self.encoder(x)
        # hn shape: [1, batch, hidden_size]
        
        # Repeat hidden state for decoder
        dec_input = hn.repeat(self.seq_len, 1, 1).transpose(0, 1)
        # dec_input shape: [batch, seq_len, hidden_size]
        
        out, _ = self.decoder(dec_input)
        out = self.fc(out)
        return out
