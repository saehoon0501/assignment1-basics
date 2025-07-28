import torch

def softmax(x:torch.Tensor, dim: int) -> torch.Tensor:
    max_i = x.max(dim=dim, keepdim=True)[0] # this gets a tuple(max_vals, the indexes of each max_val)
    # subtracting the maximum value in the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues
    x_stable = x - max_i
    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return  exp_x / sum_exp

def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes log(softmax(x)) in a numerically stable way.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension to apply log_softmax over.

    Returns:
        torch.Tensor: The log-softmax of the input tensor.
    """
    max_val = x.max(dim=dim, keepdim=True)[0]
    x_stable = x - max_val
    # This is the log-sum-exp trick. <- Underflow stability
    # log(softmax(x_i)) = log(exp(x_i) / sum(exp(x_j)))
    #                  = x_i - log(sum(exp(x_j)))
    # To make it stable, we use x_i - m - log(sum(exp(x_j - m)))
    log_sum_exp = torch.log(torch.exp(x_stable).sum(dim=dim, keepdim=True))
    return x_stable - log_sum_exp