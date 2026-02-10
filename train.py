import torch
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import get_dummy_data
from utils.text_encoder import encode_text

generator = Generator()
discriminator = Discriminator()

criterion = torch.nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

for epoch in range(1):
    images, texts = get_dummy_data()
    text_embedding = encode_text(texts[0])

    # Train Discriminator
    noise = torch.randn(1, 100)
    fake_image = generator(noise, text_embedding)

    real_label = torch.ones(1, 1)
    fake_label = torch.zeros(1, 1)

    d_real = discriminator(images, text_embedding)
    d_fake = discriminator(fake_image.detach(), text_embedding)

    d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # Train Generator
    g_loss = criterion(discriminator(fake_image, text_embedding), real_label)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    print("Training completed successfully")
