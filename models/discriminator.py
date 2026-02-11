import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, text_dim=50):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784 + text_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        x = torch.cat((image, text_embedding), dim=1)
        return self.model(x)
