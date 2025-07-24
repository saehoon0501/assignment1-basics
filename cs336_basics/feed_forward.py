import torch
from einops import einsum

class FeedForward(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None  = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model # Dimensionality of the feedforward input and output.
        self.d_ff = d_ff # Dimensionality of the up-project happening internally to your swiglu.
        self.device = device
        self.dtype = dtype
                
        self.weights_1 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.weights_2 = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.weights_3 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        
        for w in [self.weights_1, self.weights_2, self.weights_3]:
            torch.nn.init.trunc_normal_(w)

    def _silu(self, x:torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


    # def _glu(self, x:torch.Tensor) -> torch.Tensor:
        # torch.sigmoid_


    def _swi_glu(self, x:torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.weights_1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_x = einsum(self.weights_3, x, "d_ff d_model, ... d_model -> ... d_ff")

        return einsum(self.weights_2, self._silu(w1_x) * w3_x, "d_model d_ff, ... d_ff -> ... d_model")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self._swi_glu(x)