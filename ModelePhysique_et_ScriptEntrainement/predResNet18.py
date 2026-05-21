import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Configuration
IMAGE_PATH = "nouvelle_image.jpg"   # ← ton image dans le même dossier
MODEL_PATH = "vc_model2_retrained.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture (identique à l'entraînement sur l'application CosmerAnnotator) 
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#  Pré-traitement
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = Image.open(IMAGE_PATH).convert("RGB")
x = tf(img).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

# Prédiction
with torch.no_grad():
    vc_pred = model(x).item()

print(f"Image  : {IMAGE_PATH}")
print(f"→ Vc estimée : {vc_pred:.2f} cm/s")