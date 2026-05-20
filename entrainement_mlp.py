# Script d'entraînement MLP : θ → Vc
# Pipeline : dataset_summary.csv → normalisation → MLP(1→32→32→1) → mlp_theta_to_vc.pth
#
# INSTRUCTIONS :
# 1. Exportez vos données depuis l'application (Export Global ou Session).
# 2. Placez ce script à côté du fichier dataset_summary.csv.
# 3. Installez les dépendances : pip install torch pandas numpy matplotlib scikit-learn
# 4. Lancez le script : python entrainement_mlp.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os

# --- CONFIGURATION ---
CSV_PATH = "dataset_summary.csv"
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0045
MODEL_SAVE_PATH = "mlp_theta_to_vc.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Entraînement MLP (θ → Vc) ---")
print(f"Device : {DEVICE}")

# --- CHARGEMENT DES DONNÉES ---
if not os.path.exists(CSV_PATH):
    print(f"ERREUR : Fichier {CSV_PATH} introuvable.")
    exit(1)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["cable_angle_deg", "current_speed_cm_s"])
print(f"Nombre d'échantillons valides : {len(df)}")

X = df[["cable_angle_deg"]].values.astype(np.float32)
y = df["current_speed_cm_s"].values.astype(np.float32).reshape(-1, 1)

# --- NORMALISATION DE L'ENTRÉE ---
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

# --- SPLIT TRAIN / VAL (80/20) ---
n = len(X)
n_train = int(0.8 * n)
indices = np.random.permutation(n)
train_idx, val_idx = indices[:n_train], indices[n_train:]
X_train, X_val = X_norm[train_idx], X_norm[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print(f"Train samples : {len(X_train)}")
print(f"Val samples   : {len(X_val)}")


# --- DATASET PYTORCH ---
class ThetaVcDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(ThetaVcDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(ThetaVcDataset(X_val, y_val),   batch_size=BATCH_SIZE, shuffle=False)


# --- MODÈLE MLP ---
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

# --- BOUCLE D'ENTRAÎNEMENT ---
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            val_loss += criterion(model(Xb), yb).item() * Xb.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss   /= len(val_loader.dataset)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'X_mean': float(X_mean),
            'X_std':  float(X_std)
        }, MODEL_SAVE_PATH)

print(f"\nMeilleur Val Loss : {best_val_loss:.4f}")
print(f"Modèle sauvegardé : {MODEL_SAVE_PATH}")

# --- MÉTRIQUES FINALES ---
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    y_pred_val = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()

mae = mean_absolute_error(y_val, y_pred_val)
print(f"MAE validation : {mae:.2f} cm/s")

# --- VISUALISATION ---
with torch.no_grad():
    y_pred_train = model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).cpu().numpy()

plt.figure(figsize=(8, 5))
plt.scatter(y_train, y_pred_train, alpha=0.5, label='Train')
plt.scatter(y_val,   y_pred_val,   alpha=0.5, label='Val')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Idéal')
plt.xlabel("Vc réel (cm/s)")
plt.ylabel("Vc prédit (cm/s)")
plt.title("Prédictions MLP (θ → Vc)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_resultats.png", dpi=120)
plt.show()
print("Graphique sauvegardé : mlp_resultats.png")
