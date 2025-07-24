import torch
from einops import einsum

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta # Θ value for the RoPE
        self.d_k = d_k # dimension of query and key vectors
        self.max_seq_len = max_seq_len # Maximum sequence length that will be inputted
        self.device = device # Device to store the buffer on

        self.register_buffer(persistent=False)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return einsum(x, token_positions, " ... sequence_length d_k, ... sequence_length -> ")