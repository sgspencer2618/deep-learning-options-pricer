import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units=[256, 128, 64], dropout_rate=0.1):
        super(MLP, self).__init__()
        
        # Create the input layer
        layers = [nn.Linear(input_dim, hidden_units[0]),
                  nn.LeakyReLU(),
                  nn.BatchNorm1d(hidden_units[0]),
                  nn.Dropout(dropout_rate)
                  ]
        
        # Create hidden layers
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_units[i]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_units[-1], 1))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def build_mlp(input_dim, hidden_units=[128, 64], dropout_rate=0.2):
    """Factory function to create a MLP model"""
    return MLP(input_dim, hidden_units, dropout_rate)
