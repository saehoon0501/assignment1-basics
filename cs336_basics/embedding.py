import torch

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None  = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings # Size of the vocabulary
        self.embedding_dim = embedding_dim # Dimension of the embedding vectors, i.e., dmodel
        self.device = device
        self.dtype = dtype

        self.embedding_mat = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        torch.nn.init.trunc_normal_(self.embedding_mat)    

    #Lookup the embedding vectors for the given token IDs.
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_mat[token_ids]
