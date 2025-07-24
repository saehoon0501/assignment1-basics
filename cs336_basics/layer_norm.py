import torch
from einops import einsum

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model:int,
        eps:float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = torch.nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )

        torch.nn.init.trunc_normal_(self.weights)

    def _calculate_rms_a(self, a_i: float) -> float:        
        return (a_i ** 2 + self.eps)
    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # upcast to prevent overflow when square the input.

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms
        
        return einsum(result, self.weights, "... d_model, ... d_model -> ... d_model").to(in_dtype)