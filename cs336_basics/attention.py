from math import sqrt
from jaxtyping import Float, Int
import torch
from torch import Tensor
from einops import einsum
from .softmax import softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    attention_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(Q.shape[-1])
    
    # Apply mask if provided (add large negative values to masked positions)
    if mask is not None:
        mask_float = torch.where(mask, 0.0, float('-inf'))
        attention_scores = attention_scores + mask_float
    
    # Apply softmax along the keys dimension (last dimension)
    attention_weights = softmax(attention_scores, dim=-1)
    
    # Apply attention weights to values
    output = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    
    return output

class MultiHeadAttention(torch.nn.Module):
    """
    
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

    
    def forward():
        """
        
        """
