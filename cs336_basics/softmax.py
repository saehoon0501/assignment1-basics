import torch

def softmax(x:torch.Tensor, dim: int) -> torch.Tensor:
    max_i = x.max(dim=dim, keepdim=True)[0] # this gets a tuple(max_vals, the indexes of each max_val)
    # subtracting the maximum value in the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues
    x_stable = x - max_i
    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return  exp_x / sum_exp