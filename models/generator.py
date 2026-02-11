import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, text_dim=50):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        return self.model(x)
