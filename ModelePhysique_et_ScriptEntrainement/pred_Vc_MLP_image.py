# Script complet pour prédire la vitesse du courant à partir d'une image.
# Pipeline : Image → YOLO (masque) → Squelettisation → PCA (angle θ) → MLP → Vc


import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
from skimage.morphology import skeletonize

#
# Chargement du modèle MLP (θ → Vc)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Charger le modèle et les paramètres de normalisation
checkpoint = torch.load("mlp_theta_to_vc.pth", map_location='cpu', weights_only=False)

model_mlp = MLP()
model_mlp.load_state_dict(checkpoint['model_state_dict'])
model_mlp.eval()
X_mean = checkpoint['X_mean']
X_std = checkpoint['X_std']

# Prédit Vc (cm/s) à partir d'un angle θ (degrés).
def predire_vc_depuis_theta(theta_deg):
    
    theta_norm = (theta_deg - X_mean) / X_std
    theta_tensor = torch.tensor([[theta_norm]], dtype=torch.float32)
    with torch.no_grad():
        vc = model_mlp(theta_tensor).item()
    return vc


# 2. Extraction de l'angle θ depuis une image (via YOLO + PCA)
    # Retourne l'angle θ (degrés) du câble sur l'image.
    # Pipeline : YOLO segmentation → squelette → PCA → angle.
def extraire_angle_depuis_image(image_path, yolo_model_path="best.pt"):
    
    #  Étape A : YOLO segmentation
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"Modèle YOLO introuvable : {yolo_model_path}")
    
    yolo = YOLO(yolo_model_path)
    results = yolo(image_path, verbose=False)
    
    if len(results) == 0 or results[0].masks is None:
        raise ValueError("Aucun masque de câble détecté dans l'image.")
    
    # Prendre le masque du premier objet détecté
    mask = results[0].masks.data[0].cpu().numpy()
    mask_bin = (mask > 0.5).astype(np.uint8)

    #  Étape B : Squelettisation 
    skeleton = skeletonize(mask_bin)  # scikit-image (stable)
    skeleton = skeleton.astype(np.uint8) * 255

    #  Étape C : Récupérer les points du squelette 
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 5:
        raise ValueError("Pas assez de points sur le squelette (câble trop court ou mal segmenté).")

    #  Étape D : PCA pour obtenir l'angle principal 
    pts = np.column_stack((xs, ys))
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    #  Étape E : Angle par rapport à la verticale (axe y) 
    angle_rad = np.arctan2(np.abs(principal[0]), np.abs(principal[1]))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


# Fonction principale : Image → Vc

def predire_vc_depuis_image(image_path, yolo_model_path="best.pt"):
    """
    Pipeline complet :
        Image → YOLO → masque → squelette → PCA → angle θ → MLP → Vc
    """
    try:
        theta = extraire_angle_depuis_image(image_path, yolo_model_path)
        vc = predire_vc_depuis_theta(theta)
        return theta, vc
    except Exception as e:
        print(f" Erreur : {e}")
        return None, None


# Exécution principale (test sur une image)

if __name__ == "__main__":
    # Remplace ce chemin par le chemin de ta vraie image 
    image_a_tester = "nouvelle_image.jpg"
    
    # Vérifier que l'image existe
    if not os.path.exists(image_a_tester):
        print(f" L'image '{image_a_tester}' n'existe pas.")
        exit()
    
    theta, vc = predire_vc_depuis_image(image_a_tester)
    
    if theta is not None:
        print(f" Angle mesuré : {theta:.2f}°")
        print(f" Vitesse estimée : {vc:.2f} cm/s")
    else:
        print(" Impossible d'estimer la vitesse.")
        