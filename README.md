# Text-to-Image GAN Pipeline

This project demonstrates a basic Text-to-Image generation pipeline using
Generative Adversarial Networks (GANs).

The model learns to generate simple images conditioned on text embeddings.
This project is designed for academic demonstration and viva purposes.

## Technologies Used
- Python
- PyTorch
- GANs
- NLP (Text Embeddings)

## Project Structure
- models/: Generator and Discriminator models
- utils/: Dataset and text encoding utilities
- train.py: Model training script
- generate.py: Image generation from text

## How to Run
1. Install dependencies  
   `pip install -r requirements.txt`
2. Train the model  
   `python train.py`
3. Generate image  
   `python generate.py`

## Output
Generated images are saved in the `outputs/` folder.

## Author
Arushi Das
