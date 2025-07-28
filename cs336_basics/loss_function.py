from .softmax import log_softmax
from torch import Tensor
import torch
from jaxtyping import Float, Int
from einops import reduce

def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    neg_log_probs_all = -log_softmax(inputs, -1)
    num_examples = inputs.shape[0]
    loss_per_example = neg_log_probs_all[torch.arange(num_examples), targets]

    loss = reduce(loss_per_example, 'batch_size -> ()', 'mean')
    return loss