import torch

def get_dummy_data(batch_size=1):
    images = torch.randn(batch_size, 784)
    texts = ["a simple object"]
    return images, texts

