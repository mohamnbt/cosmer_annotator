import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


# Configuration

CSV_PATH = "dataset_summary.csv"        # ton fichier CSV
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0045
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Chargement et préparation des données

df = pd.read_csv(CSV_PATH)

# On garde les lignes avec un angle valide et une vitesse valide
df = df.dropna(subset=["cable_angle_deg", "current_speed_cm_s"])
print(f"Nombre d'échantillons : {len(df)}")

# On sépare en train/val (80/20) pour validation, mais on utilisera tout pour l'apprentissage final
X = df[["cable_angle_deg"]].values.astype(np.float32)
y = df["current_speed_cm_s"].values.astype(np.float32).reshape(-1, 1)

# Normalisation
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

# Split train/val
n = len(X)
n_train = int(0.8 * n)
indices = np.random.permutation(n)
train_idx, val_idx = indices[:n_train], indices[n_train:]
X_train, X_val = X_norm[train_idx], X_norm[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Dataset Pytorch

class ThetaVcDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ThetaVcDataset(X_train, y_train)
val_ds   = ThetaVcDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# Modèle MLP

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

model = MLP().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Entraîtement

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * Xb.size(0)

    train_loss /= len(train_ds)
    val_loss /= len(val_ds)

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


# 6. Sauvegarde du modèle et paramètres de normalisation

torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean,
    'X_std': X_std
}, "mlp_theta_to_vc.pth")
print("- Modèle et paramètres de normalisation sauvegardés : mlp_theta_to_vc.pth")


# Visualisation Rapide

model.eval()
with torch.no_grad():
    y_pred_train = model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    y_pred_val = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()

plt.figure(figsize=(8,5))
plt.scatter(y_train, y_pred_train, alpha=0.5, label='Train')
plt.scatter(y_val, y_pred_val, alpha=0.5, label='Val')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Vc réel (cm/s)")
plt.ylabel("Vc prédit (cm/s)")
plt.title("Prédictions MLP (θ → Vc)")
plt.legend()
plt.grid(True)
plt.show()

print(" - Entraînement terminé. Le modèle est prêt à l'emploi.")

# Calcul du MAE sur l'ensemble de validation
model.eval()
with torch.no_grad():
    # Prédictions sur les données normalisées
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_pred_val = model(X_val_tensor).cpu().numpy()

# Renormaliser les prédictions pour les comparer aux vraies valeurs (y_val)
# On inverse la normalisation : y = y_norm * std + mean
# Ici, y_val est déjà en cm/s, mais les prédictions sont sur les données normalisées.
# On doit prédire sur les données normalisées et comparer aux y_val (en cm/s).
# C'est déjà bon car y_pred_val est la sortie du modèle, et y_val est la vérité terrain en cm/s.
# On calcule le MAE entre les deux :
from sklearn.metrics import mean_absolute_error
mae_val = mean_absolute_error(y_val, y_pred_val)
print(f" MAE sur l'ensemble de validation : {mae_val:.2f} cm/s")