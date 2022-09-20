import torch


class BertPooler(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        index = torch.tensor(0, device=hidden_states.device)
        first_token_tensor = torch.index_select(hidden_states, dim=-2, index=index)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
