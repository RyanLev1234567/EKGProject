import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        scores = self.attention(lstm_output)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # normalize over time
        context = torch.sum(weights * lstm_output, dim=1)  # weighted sum over time
        return context, weights

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_channels=12, num_classes=71, lstm_hidden=128):
        super(CNN_LSTM_Attention, self).__init__()

        # CNN to extract features per time step
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(lstm_hidden*2)
        self.fc = nn.Linear(lstm_hidden*2, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, leads] -> need [batch, leads, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        context, attn_weights = self.attention(lstm_out)  # [batch, hidden*2]
        out = self.fc(context)
        return out
