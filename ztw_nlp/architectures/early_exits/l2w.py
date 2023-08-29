import torch
import torch.nn as nn

from architectures.early_exits.sdn import SDN


class L2W(SDN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TanhWPN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=500, num_layers=1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for l in self.hidden_layers:
            x = torch.relu(l(x))
        x = torch.tanh(self.output_layer(x))
        return x
