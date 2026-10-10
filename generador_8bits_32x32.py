import clip  # pip install git+https://github.com/openai/CLIP.gi
import torch
import torch.nn as nn


class TextToPixel32Generator(nn.Module):
    def __init__(self, text_embed_dim=512, noise_dim=100):
        super(TextToPixel32Generator, self).__init__()
        # Proyecta el vector de texto + ruido a una característica espacial base (4x4)
        self.fc = nn.Linear(text_embed_dim + noise_dim, 512 * 4 * 4)
        # Red de convolución que escala progresivamente de 4x4 -> 8x8 -> 16x16 -> 32x32
        self.deconv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Capa final de color RGB (3 canales, 32x32)
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Salida de píxeles entre [-1, 1]
        )

    def forward(self, text_embeddings, noise):
        # Concatenar el embedding del texto con un vector de ruido aleatorio
        x = torch.cat([text_embeddings, noise], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)
        pixel_art_32x32 = self.deconv_layers(x)
        return pixel_art_32x32


# --- Ejemplo de Uso ---
# 1. Cargar CLIP para codificar la palabra
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)
prompt = "puerta de madera laberinto 8 bits"
text_token = clip.tokenize([prompt]).to(device)
with torch.no_grad():
    text_embedding = clip_model.encode_text(text_token).float()
# 2. Generar Sprite 32x32
generator = TextToPixel32Generator().to(device)
noise = torch.randn(1, 100).to(device)
generated_sprite = generator(text_embedding, noise)  # Dimensiones: [1, 3, 32, 32]
