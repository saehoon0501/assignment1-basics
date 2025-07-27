from torch import Tensor
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import RMSNorm
from jaxtyping import Float
import torch
from .token_embedding import Embedding
from .linear import Linear

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_head: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            d_model, 
            num_head, 
            weights['attn.q_proj.weight'],
            weights['attn.k_proj.weight'],
            weights['attn.v_proj.weight'],
            weights['attn.output_proj.weight'],
            max_seq_len=max_seq_len, 
            theta=theta
        )
        self.ff = FeedForward(d_model, d_ff)
        
        w1,w2,w3 = (weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'])
        self.ff.weights_1 = torch.nn.Parameter(w1)
        self.ff.weights_2 = torch.nn.Parameter(w2)
        self.ff.weights_3 = torch.nn.Parameter(w3)

        self.rms_norm_1 = RMSNorm(d_model)
        self.rms_norm_1.weights = torch.nn.Parameter(weights['ln1.weight'])
        
        self.rms_norm_2 = RMSNorm(d_model)
        self.rms_norm_2.weights = torch.nn.Parameter(weights['ln2.weight'])

    def forward(self, in_features:Float[Tensor, "batch sequence_length d_model"]) -> Tensor:
        # y= x + MultiHeadSelfAttention(RMSNorm(x))
        token_positions = torch.arange(in_features.shape[1], device=in_features.device).unsqueeze(0)
        residual = self.multi_head_attention.forward(self.rms_norm_1.forward(in_features), token_positions=token_positions)
        multi_attention = in_features + residual

        # y = x + FeedForward(RMSNorm(x))
        residual = self.ff.forward(self.rms_norm_2.forward(multi_attention))
        ff = multi_attention + residual

        return ff
    
class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.token_embeddings.embedding_mat = torch.nn.Parameter(weights['token_embeddings.weight'])

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_head=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                weights={k.split('.', 2)[-1]: v for k, v in weights.items() if k.startswith(f"layers.{i}.")}
            ) for i in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.ln_final.weights = torch.nn.Parameter(weights['ln_final.weight'])
        
        self.lm_head = Linear(d_model, vocab_size)
        self.lm_head.weights = torch.nn.Parameter(weights['lm_head.weight'])

    def forward(self, in_indices:Float[Tensor, "batch_size sequence_length"]) -> Tensor:
        x = self.token_embeddings.forward(in_indices)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        logits = self.lm_head.forward(x)
        return logits
    