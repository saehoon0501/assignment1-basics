from math import sqrt
from jaxtyping import Float, Int
import torch
from torch import Tensor
from einops import einsum, rearrange
from .softmax import softmax
from .positional_embedding import RotaryPositionalEmbedding

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
    Multi-head attention module for transformer models.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,        
        q_proj_weight: (Float[Tensor, "d_k d_in"]),
        k_proj_weight: (Float[Tensor, "d_k d_in"]),
        v_proj_weight: (Float[Tensor, "d_k d_in"]),
        o_proj_weight: (Float[Tensor, "d_model d_v"]),
        max_seq_len: int | None = None,
        theta: float | None= None,
        device: torch.device | None  = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.multi_head_q_proj_weight = rearrange(q_proj_weight, "(h d_k_h) d_in -> h d_k_h d_in", h=num_heads)
        self.multi_head_k_proj_weight = rearrange(k_proj_weight, "(h d_k_h) d_in -> h d_k_h d_in", h=num_heads)
        self.multi_head_v_proj_weight = rearrange(v_proj_weight, "(h d_v_h) d_in -> h d_v_h d_in", h=num_heads)
        self.o_proj_weight = o_proj_weight
        self.device = device
        self.dtype = dtype
        self.rope = None
        if theta:
            self.rope = RotaryPositionalEmbedding(theta, self.multi_head_k_proj_weight.shape[1], max_seq_len=max_seq_len, device=device)
        
    
    def forward(
        self, 
        in_features: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        """
        Given the input sequence, proceed self-multi_head_attention.

        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        # Project inputs to Q, K, V for all heads
        q = einsum(in_features, self.multi_head_q_proj_weight, "... sequence_length d_in, h d_k_h d_in -> ... h sequence_length d_k_h")
        k = einsum(in_features, self.multi_head_k_proj_weight, "... sequence_length d_in, h d_k_h d_in -> ... h sequence_length d_k_h")
        v = einsum(in_features, self.multi_head_v_proj_weight, "... sequence_length d_in, h d_v_h d_in -> ... h sequence_length d_v_h")
        
        # RoPE should be applied to the query and key vectors, but not the value vectors
        if self.rope and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create causal mask
        sequence_length = in_features.shape[-2]
        mask = ~torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=in_features.device), diagonal=1)
        
        # Apply scaled dot product attention
        result = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # Project back to d_model dimensions
        output_rearranged = rearrange(result, "... h queries d_v_h -> ... queries (h d_v_h)")
        return einsum(output_rearranged, self.o_proj_weight, "... queries d_v, d_model d_v -> ... queries d_model")
