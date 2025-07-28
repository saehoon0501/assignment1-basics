import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        weight_decay,
        betas,
        eps,
        lr
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps":eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            b_1, b_2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data # Get the gradient of loss with respect to p.

                m = b_1 * m + (1-b_1) * grad # Update the first moment estimate
                v = b_2 * v + (1-b_2) * (grad**2)
                
                lr_t = lr * math.sqrt(1-b_2**t) / (1 - b_1 ** t)
                p.data -= lr_t * (m / (torch.sqrt(v) + eps)) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        
        return loss