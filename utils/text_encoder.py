import torch

def encode_text(text, embedding_dim=50):
    torch.manual_seed(len(text))
    return torch.rand(1, embedding_dim)
