import torch
import matplotlib.pyplot as plt
from models.generator import Generator
from utils.text_encoder import encode_text

generator = Generator()
text = "a simple object"
text_embedding = encode_text(text)

noise = torch.randn(1, 100)
generated_image = generator(noise, text_embedding)
image = generated_image.view(28, 28).detach().numpy()

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.savefig("outputs/sample.png")
plt.show()
