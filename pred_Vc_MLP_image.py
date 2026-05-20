# Script de prédiction Vc depuis une image inconnue — méthode MLP
# Pipeline : Image → YOLO (segmentation câble) → Squelettisation → PCA (angle θ) → MLP → Vc
#
# INSTRUCTIONS :
# 1. Assurez-vous d'avoir mlp_theta_to_vc.pth (généré par entrainement_mlp.py)
# 2. Assurez-vous d'avoir best.pt (modèle YOLO entraîné sur vos câbles)
# 3. Installez les dépendances : pip install torch torchvision ultralytics scikit-image opencv-python numpy
# 4. Lancez : python pred_Vc_MLP_image.py --image nouvelle_image.jpg

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize

# --- ARCHITECTURE MLP (identique à l'entraînement) ---
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


def load_mlp(model_path: str):
    """Charge le MLP et les paramètres de normalisation."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle MLP introuvable : {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = MLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    X_mean = checkpoint['X_mean']
    X_std  = checkpoint['X_std']
    print(f"MLP chargé : {model_path} | X_mean={X_mean:.3f}, X_std={X_std:.3f}")
    return model, X_mean, X_std


def predire_vc_depuis_theta(theta_deg: float, model, X_mean: float, X_std: float) -> float:
    """Prédit Vc (cm/s) à partir d'un angle θ (degrés)."""
    theta_norm = (theta_deg - X_mean) / X_std
    t = torch.tensor([[theta_norm]], dtype=torch.float32)
    with torch.no_grad():
        return model(t).item()


def extraire_angle_depuis_image(image_path: str, yolo_path: str = "best.pt") -> float:
    """
    Extrait l'angle θ du câble dans une image.
    Pipeline : YOLO segmentation → masque binaire → squelettisation → PCA → angle vs verticale.
    """
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"Modèle YOLO introuvable : {yolo_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    # Étape A — YOLO segmentation
    yolo = YOLO(yolo_path)
    results = yolo(image_path, verbose=False)

    if len(results) == 0 or results[0].masks is None:
        raise ValueError("Aucun masque de câble détecté dans l'image.")

    mask = results[0].masks.data[0].cpu().numpy()
    mask_bin = (mask > 0.5).astype(np.uint8)

    # Étape B — Squelettisation (scikit-image)
    skeleton = skeletonize(mask_bin).astype(np.uint8) * 255

    # Étape C — Récupération des points du squelette
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 5:
        raise ValueError("Squelette trop court (câble mal segmenté ou trop petit).")

    # Étape D — PCA pour l'axe principal
    pts = np.column_stack((xs, ys)).astype(float)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Étape E — Angle par rapport à la verticale (axe y image)
    angle_deg = np.degrees(np.arctan2(np.abs(principal[0]), np.abs(principal[1])))
    return float(angle_deg)


def predire_vc_depuis_image(
    image_path: str,
    mlp_path: str = "mlp_theta_to_vc.pth",
    yolo_path: str = "best.pt"
) -> tuple[float, float]:
    """
    Pipeline complet : Image → Vc.
    Retourne (theta_deg, vc_cm_s).
    """
    model, X_mean, X_std = load_mlp(mlp_path)
    theta = extraire_angle_depuis_image(image_path, yolo_path)
    vc = predire_vc_depuis_theta(theta, model, X_mean, X_std)
    return theta, vc


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédit Vc (cm/s) depuis une image de câble via YOLO + PCA + MLP."
    )
    parser.add_argument("--image",  "-i", type=str, required=True,  help="Chemin vers l'image")
    parser.add_argument("--mlp",    "-m", type=str, default="mlp_theta_to_vc.pth", help="Fichier .pth du MLP")
    parser.add_argument("--yolo",   "-y", type=str, default="best.pt",             help="Fichier .pt du modèle YOLO")
    args = parser.parse_args()

    try:
        theta, vc = predire_vc_depuis_image(args.image, args.mlp, args.yolo)
        print(f"\n{'='*45}")
        print(f"  Image         : {args.image}")
        print(f"  Angle θ (PCA) : {theta:.2f} °")
        print(f"  → Vc estimée  : {vc:.2f} cm/s")
        print(f"{'='*45}\n")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
