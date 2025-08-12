import torch.nn as nn
import torch.nn.functional as F

class AttentiveGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, use_attention=True, dropout_rate=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        if use_attention:
            self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x: [batch, seq, feature]
        gru_out, _ = self.gru(x) # [batch, seq, hidden]
        # Apply dropout to GRU output
        gru_out = F.dropout(gru_out, p=self.dropout_rate, training=self.training)

        if self.use_attention:
            attn_weights = F.softmax(self.attn(gru_out), dim=1) # [batch, seq, 1]
            context = (gru_out * attn_weights).sum(dim=1)       # [batch, hidden]
        else:
            context = gru_out[:, -1, :] # Just use last hidden state

        # Optional: Dropout before the final FC
        context = F.dropout(context, p=self.dropout_rate, training=self.training)

        return self.fc(context).squeeze(-1)
