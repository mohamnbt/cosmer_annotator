import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Paramètres Modèle Sphérique  (config câble 25 cm)

rho   = 1020      # kg/m³
g     = 9.81        # m/s²
H     = 0.295      # hauteur d'eau (m)
L     = 0.24        # longueur du câble soumis au courant (m)
d     = 0.004       # diamètre du câble (m)
Sc    = np.pi * (d/2)**2
mu    = 0.0491      # masse linéique du câble (kg/m)

R     = 0.08        # rayon de la bouée sphérique (m) ou rayon équivalent
m_b   = 0.493       # masse de la bouée (kg)
Cdb   = 0.56      # coeff de traînée bouée (élevé car bas de la bouée en forme de cône et )
Cdc   = 0.8    # coeff de traînée câble


# FONCTIONS GÉOMÉTRIE + RÉSOLUTION θ = f(v)

def geometrie(theta):
    """Retourne V_imm, S_imm et z_cp (centre de poussée) pour un angle theta (rad)."""
    h = np.clip(H - L * np.cos(theta), 0.0, 2*R)
    V_imm = np.pi * h**2 / 3 * (3*R - h)
    S_imm = np.pi * h * (2*R - h)
    z_bas = L * np.cos(theta)
    # centre de poussée simplifié (centre de la calotte ≈ h/2 depuis le bas)
    z_cp = z_bas + h/2   # approximation acceptable
    return V_imm, S_imm, z_cp

def theta_from_v(v_ms, theta0=15.0, alpha=0.3, tol=1e-10, max_iter=1000):
    """
    Calcule θ (degrés) pour une vitesse v (m/s) en résolvant l'équation implicite :
    tanθ = N(θ) / D(θ)
    avec amortissement.
    """
    theta = np.radians(theta0)
    for _ in range(max_iter):
        V, S, z_cp = geometrie(theta)
        ct = np.cos(theta)
        # Numérateur (traînée)
        N = 0.5 * rho * v_ms**2 * (Cdb * S * (L + R) + Cdc * d * ct**2 * L**2 / 2)
        # Dénominateur (rappel)
        D = g * ((rho*V - m_b) * z_cp + (rho*Sc - mu) * ct * L**2 / 2)
        if D <= 0:
            return None
        theta_new = (1 - alpha) * theta + alpha * np.arctan(N / D)
        if abs(theta_new - theta) < tol:
            return np.degrees(theta_new)
        theta = theta_new
    return None

def vitesse_locale(theta_deg, Li=0.01):
    """Retourne la vitesse locale (m/s) pour un angle theta (degrés) et une longueur de segment Li."""
    theta = np.radians(theta_deg)
    ct = np.cos(theta)
    # Formule simplifiée (câble fin)
    num = 2 * g * (rho*Sc - mu) * np.tan(theta)
    denom = rho * Cdc * d * ct
    v_ms = np.sqrt(num / denom)  # si denom > 0
    return v_ms * 100  # en cm/s

# Modele etendu a chaque segment d'un catenaire grace a la methode PBD (approximation)
def vitesse_locale_sans_bouee(theta_deg):
    """
    Calcule la vitesse locale (cm/s) sur un segment de câble sans bouée.
    Input : theta_deg (degrés) de ce segment.
    """
    theta = np.radians(theta_deg)
    ct = np.cos(theta)
    num = 2 * g * np.abs(rho*Sc - mu) * np.tan(theta)  
    den = rho * Cdc * d * ct**2
    if den <= 0:
        return None
    v_ms = np.sqrt(num / den)
    return v_ms * 100  # conversion en cm/s

# Exemple d'application pour un profil de vitesse
angles_locaux = [5.2, 7.8, 10.1, 12.3, 15.0, 17.2, 20.5]  # à remplacer par tes theta extraits de l'image
profil_vitesse = [vitesse_locale_sans_bouee(a) for a in angles_locaux]

print("Profil de vitesse (cm/s) :", profil_vitesse) 

# CHARGEMENT DES DONNÉES EXPÉRIMENTALES

df = pd.read_csv("dataset_summary.csv")   
df = df.dropna(subset=["cable_angle_deg", "current_speed_cm_s"])
print(f"{len(df)} points chargés")


# CALCUL DE θ THÉORIQUE POUR CHAQUE VITESSE MESURÉE

theta_theo = []
for v_cms in df["current_speed_cm_s"]:
    v_ms = v_cms / 100.0
    th = theta_from_v(v_ms)
    theta_theo.append(th)

df["theta_theo_deg"] = theta_theo
# Supprimer les lignes où le calcul a échoué
df_ok = df.dropna(subset=["theta_theo_deg"])
print(f"{len(df_ok)} points avec solution physique")


# GRAPHIQUE 1 : courbe théorique θ(v) + points mesurés

v_range = np.linspace(5, 35, 100)   # cm/s
theta_range = [theta_from_v(v/100) for v in v_range]

plt.figure(figsize=(8,5))
plt.plot(v_range, theta_range, 'r-', label="Modèle sphérique")
plt.scatter(df_ok["current_speed_cm_s"], df_ok["cable_angle_deg"],
            alpha=0.6, label="Mesures (angle PCA)")
plt.xlabel("Vitesse courant (cm/s)")
plt.ylabel("Angle θ (degrés)")
plt.title("Validation du modèle physique : θ = f(Vc)")
plt.legend()
plt.grid(True)
plt.show()


# GRAPHIQUE 2 : θ théorique vs θ mesuré

plt.figure(figsize=(6,6))
plt.scatter(df_ok["cable_angle_deg"], df_ok["theta_theo_deg"], alpha=0.7)
plt.plot([0,40],[0,40], 'k--')
plt.xlabel("θ mesuré (PCA) [°]")
plt.ylabel("θ théorique (modèle) [°]")
plt.title("Corrélation angle théorique / angle mesuré")
plt.grid(True)
plt.show()


# MÉTRIQUES

erreur = df_ok["cable_angle_deg"] - df_ok["theta_theo_deg"]
mae = np.mean(np.abs(erreur))
print(f"MAE angle : {mae:.2f}°")