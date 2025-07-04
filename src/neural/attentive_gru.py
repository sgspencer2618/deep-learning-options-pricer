import torch.nn as nn
import torch.nn.functional as F

class AttentiveGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, use_attention=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x: [batch, seq, feature]
        gru_out, _ = self.gru(x) # [batch, seq, hidden]
        if self.use_attention:
            attn_weights = F.softmax(self.attn(gru_out), dim=1) # [batch, seq, 1]
            context = (gru_out * attn_weights).sum(dim=1)       # [batch, hidden]
        else:
            context = gru_out[:, -1, :] # Just use last hidden state
        return self.fc(context).squeeze(-1)
