import torch
from einops import einsum, rearrange

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even.")

        # Create position indices
        positions = torch.arange(max_seq_len, device=device).float()

        # Create frequency indices
        # The formula uses k from 1 to d/2. With 0-based indexing for k, this corresponds
        # to 2k/d for k from 0 to d/2-1, which is what torch.arange(0, d_k, 2) creates.
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # Calculate angles (the m * theta_k part)
        # Shape: (max_seq_len, d_k / 2)
        freqs = einsum(positions, inv_freq, "m, theta_k -> m theta_k")

        # Register sin and cos values as non-learnable buffers
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., sequence_length, d_k)
            token_positions (torch.Tensor): Positions of tokens, shape (..., sequence_length)

        Returns:
            torch.Tensor: Rotated tensor of the same shape as x.
        """
        # Retrieve sin and cos values for the given token positions
        # Shapes: (..., sequence_length, d_k / 2)
        sin = self.sin_cached[token_positions]
        cos = self.cos_cached[token_positions]

        # Reshape x to treat pairs of features
        # x: (..., sequence_length, d_k) -> (..., sequence_length, d_k/2, 2)
        # x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_reshaped = rearrange(x, "... sequence_length (d_k_half p) -> ... sequence_length d_k_half p", p=2)
        x1, x2 = x_reshaped.unbind(-1)

        # Apply the rotation to pairs of features
        # y1 = x1*cos - x2*sin
        # y2 = x1*sin + x2*cos
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Stack and reshape back to original dimension
        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1)
        return rotated_x.flatten(start_dim=-2)