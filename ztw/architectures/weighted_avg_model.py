import torch
from torch import nn

import model_funcs as mf


class WeightedAverage(nn.Module):
    def __init__(self, args, params):
        super().__init__()
        self.input_dim = int(params['input_dim'])
        self.output_dim = int(params['output_dim'])
        self.head_idx = int(params['head_idx'])
        self.input_type = params['input_type']
        self.train_logits = params['train_logits']
        self.train_last_logits = params['train_last_logits']
        self.train_labels = params['train_labels']
        self.test_logits = params['test_logits']
        self.test_last_logits = params['test_last_logits']
        self.test_labels = params['test_labels']
        self.num_classes = params['num_classes']
        self.weight = nn.Parameter(
            torch.normal(0, 0.01, size=(self.input_dim // self.num_classes,), requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(size=(self.output_dim,)))
        self.num_heads = self.input_dim // self.num_classes
        self.softmax = params['softmax']
        self.ensemble_mode = params['ensemble_mode']
        self.train_func = mf.run_ensb_train

    def forward(self, x):
        # x shape is [batch_size, num_heads, input_dim]
        if self.ensemble_mode == 'additive':
            resized_weight = self.weight.view(1, -1, 1).exp()
            x = (x * resized_weight).sum(1) + self.bias.view(1, -1).exp()
            x = x / x.sum(1, keepdim=True)
            return (x + 1e-6).log()
        elif self.ensemble_mode == 'geometric':
            resized_weight = torch.exp(self.weight.view(1, -1, 1))
            if self.softmax:
                resized_weight = resized_weight.softmax(1)
            x = x * resized_weight
            return x.mean(1) + self.bias
        elif self.ensemble_mode == 'standard':
            return x.mean(1).log()
