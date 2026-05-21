# COSMER Annotator — Documentation complète

> **Outil d'annotation et d'apprentissage automatique pour l'estimation de la vitesse de courant marin**  
> par analyse visuelle de câbles de mouillage sous-marins.  
> Développé au **Laboratoire COSMER — Université de Toulon**  
> Auteur : **Mohamed-Amine Boutahri** (stagiaire L3 Physique, 2025–2026)

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Architecture générale](#2-architecture-générale)
3. [Outils utilisés et justification](#3-outils-utilisés-et-justification)
4. [Prérequis](#4-prérequis)
5. [Installation](#5-installation)
6. [Démarrage de l'application](#6-démarrage-de-lapplication)
7. [Guide d'utilisation pas à pas](#7-guide-dutilisation-pas-à-pas)
8. [Calculs effectués dans le backend](#8-calculs-effectués-dans-le-backend)
9. [Scripts Python standalone](#9-scripts-python-standalone)
10. [Modèle physique du câble](#10-modèle-physique-du-câble)
11. [API REST — référence complète](#11-api-rest--référence-complète)
12. [Structure des fichiers de données](#12-structure-des-fichiers-de-données)
13. [Entraînement IA et modèles](#13-entraînement-ia-et-modèles)
14. [Auto-annotation YOLO](#14-auto-annotation-yolo)
15. [Export des données](#15-export-des-données)
16. [Étendre / modifier l'application](#16-étendre--modifier-lapplication)
17. [FAQ et dépannage](#17-faq-et-dépannage)

---

## 1. Contexte et objectif

Lorsqu'un câble de mouillage est soumis à un courant marin, il se défléchit et forme un angle par rapport à la verticale. Cet angle est directement corrélé à la vitesse du courant : plus le courant est fort, plus le câble est incliné.

L'objectif de ce travail est d'**estimer automatiquement la vitesse de courant (Vc) à partir d'images de câbles sous-marins**, sans capteur de courant dédié. Deux approches complémentaires ont été développées :

- **Approche deep learning — ResNet-18** : le réseau prédit Vc directement à partir de l'image entière (pixels → Vc).
- **Approche physique + MLP** : YOLO détecte le câble → ACP calcule l'angle θ → un petit réseau MLP prédit Vc à partir de θ seul.

L'application **COSMER Annotator** centralise toute la chaîne de travail :

1. **Annoter** des images (centerline ou contour du câble) avec les conditions expérimentales associées.
2. **Calculer automatiquement** l'angle de déflexion (ACP), l'angle accordal et l'indice de courbure.
3. **Construire un dataset** au format YOLO pour entraîner des modèles de segmentation.
4. **Entraîner** les modèles ResNet-18 ou MLP directement depuis l'interface graphique.
5. **Visualiser** en temps réel les courbes d'entraînement, les métriques et les graphiques Vc = f(θ).
6. **Prédire** Vc sur de nouvelles images depuis l'application.

L'application est conçue pour être utilisée par des chercheurs sans compétences en développement logiciel.

---

## 2. Architecture générale

```
cosmer_annotator/
├── backend/
│   ├── main.py                     ← API FastAPI — toute la logique métier
│   ├── requirements.txt            ← dépendances Python backend
│   └── best.pt                     ← (optionnel) modèle YOLO pré-entraîné pour auto-annotation
├── frontend/
│   ├── src/                        ← Interface React + TypeScript (Vite)
│   │   ├── pages/                  ← Pages principales (annotation, global, statistiques...)
│   │   ├── components/             ← Composants réutilisables
│   │   └── lib/api.ts              ← Appels API centralisés
│   └── package.json
├── ModelePhysique_et_ScriptEntrainement/
│   ├── entrainementResNet18.py     ← Script standalone entraînement ResNet-18
│   ├── entrainementMLP.py          ← Script standalone entraînement MLP (θ → Vc)
│   ├── pred_Vc_MLP_image.py        ← Script de prédiction MLP sur image (via YOLO + ACP)
│   ├── predResNet18.py             ← Script de prédiction ResNet-18 sur image
│   └── modelephysique.py           ← Modèle physique analytique du câble (équation d'équilibre)
├── data/
│   ├── sessions/                   ← données générées (images, annotations, labels)
│   └── models/                     ← modèles PyTorch (.pth) entraînés
├── start.sh                        ← script de démarrage tout-en-un
└── requirements.txt                ← dépendances Python (racine)
```

**Flux de données :**

```
Navigateur (React) ──HTTP/REST──► FastAPI (Python)
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼             ▼
                     data/sessions  OpenCV/NumPy  PyTorch/YOLO
                     (JSON + images) (ACP, calculs) (ResNet18, MLP)
```

Le **backend** tourne sur `http://localhost:8000` et expose une API REST.  
Le **frontend** tourne sur `http://localhost:5173` et communique exclusivement via cette API.

---

## 3. Outils utilisés et justification

### Backend — Python / FastAPI

| Outil | Rôle | Pourquoi ce choix |
|-------|------|-------------------|
| **FastAPI** | Framework API REST | Rapide, moderne, génère automatiquement la doc Swagger (`/docs`). Supporte les tâches d'arrière-plan (`BackgroundTasks`) indispensables pour l'entraînement non bloquant. |
| **Uvicorn** | Serveur ASGI | Serveur léger compatible FastAPI, supporte le rechargement à chaud en dev (`--reload`). |
| **OpenCV** (`cv2`) | Traitement d'images | Rasterisation des masques polygonaux YOLO, squelettisation avec `skimage`. |
| **NumPy** | Calcul matriciel | ACP (matrice de covariance, valeurs propres), rééchantillonnage équidistant des points. |
| **PyTorch + TorchVision** | Deep learning | Entraînement ResNet-18 et MLP. Supporte CUDA, Apple Silicon (MPS) et CPU automatiquement. |
| **Ultralytics YOLO** | Détection/segmentation | YOLOv8 pour l'auto-annotation des câbles. Utilisé à la fois pour créer le dataset d'entraînement et pour la prédiction dans le pipeline MLP. |
| **scikit-image** | Morphologie | Squelettisation (`skeletonize`) du masque de câble pour extraire la centerline. |
| **FFmpeg** (subprocess) | Extraction vidéo | Extraction de frames depuis des vidéos DJI/GoPro. Outil standard, très rapide, appelé via `subprocess`. |

### Frontend — React / TypeScript

| Outil | Rôle | Pourquoi ce choix |
|-------|------|-------------------|
| **React 18 + TypeScript** | Interface utilisateur | Composants réactifs, typage fort, écosystème riche. |
| **Vite** | Build tool | Démarrage instantané, HMR (hot module reload) en développement. |
| **Recharts** | Graphiques | Bibliothèque de visualisation declarative intégrée nativement avec React. Utilisée pour les courbes de loss, scatter plots, histogrammes. |
| **Wouter** | Routing | Routeur léger (alternative à React Router), suffit pour ce projet single-page. |
| **Tailwind CSS** | Style | Utilitaires CSS, cohérence visuelle rapide, thème sombre/clair via variables CSS. |

### Pourquoi deux approches IA (ResNet-18 vs MLP) ?

| | ResNet-18 | MLP (θ → Vc) |
|---|---|---|
| **Entrée** | Image RGB 224×224 | Angle θ (1 scalaire) |
| **Sortie** | Vc (cm/s) | Vc (cm/s) |
| **Avantage** | Pas besoin de détection préalable, capture les informations visuelles fines | Modèle très léger, interprétable, cohérent avec la physique |
| **Inconvénient** | Nécessite beaucoup de données, boîte noire | Dépend de YOLO pour extraire θ sur une image inconnue |
| **Usage recommandé** | Dataset large, robustesse | Dataset limité, pipeline explicable, validation du modèle physique |

Les deux approches sont disponibles dans l'application et dans les scripts standalone.

---

## 4. Prérequis

### Système

| Outil | Version minimale | Usage |
|-------|-----------------|-------|
| Python | 3.10+ | Backend FastAPI |
| Node.js | 18+ | Frontend React/Vite |
| npm | 9+ | Gestion des paquets frontend |
| FFmpeg | toute version récente | Extraction de frames vidéo |

### Installation de FFmpeg

```bash
# macOS (avec Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Télécharger depuis https://ffmpeg.org/download.html et ajouter au PATH
```

---

## 5. Installation

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/mohamnbt/cosmer_annotator.git
cd cosmer_annotator
```

### Étape 2 — Installer les dépendances Python

```bash
pip install -r requirements.txt
```

Dépendances Python installées :

| Paquet | Rôle |
|--------|------|
| `fastapi` | Framework web API |
| `uvicorn` | Serveur ASGI |
| `opencv-python-headless` | Traitement d'images (centerline, masques) |
| `python-multipart` | Upload de fichiers |
| `scikit-image` | Squelettisation morphologique |
| `ultralytics` | Auto-annotation YOLO (optionnel) |
| `torch`, `torchvision` | Entraînement ResNet-18 et MLP (optionnel) |

### Étape 3 — Installer les dépendances frontend

```bash
cd frontend
npm install
cd ..
```

### Étape 4 (optionnel) — Modèle YOLO pré-entraîné

Pour utiliser l'auto-annotation et le pipeline MLP :

```bash
# Placer un modèle YOLO segmentation pré-entraîné sur les câbles
cp votre_modele.pt backend/best.pt
```

> Sans `best.pt`, toutes les fonctions d'annotation manuelle, d'export et de statistiques restent disponibles. Seules l'auto-annotation et la prédiction MLP sont désactivées.

---

## 6. Démarrage de l'application

### Méthode rapide (recommandée)

```bash
chmod +x start.sh
./start.sh
```

Ce script démarre le backend FastAPI sur le port **8000** et le frontend Vite sur le port **5173**, et gère l'arrêt propre des deux processus avec `Ctrl+C`.

**Ouvrir dans le navigateur :** `http://localhost:5173`

### Méthode manuelle (deux terminaux)

**Terminal 1 — Backend :**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend :**
```bash
cd frontend
npm run dev
```

### Vérification

- API backend + documentation interactive : `http://localhost:8000/docs`
- Frontend : `http://localhost:5173`

---

## 7. Guide d'utilisation pas à pas

### 7.1 Créer une session

Une **session** est un ensemble d'images liées à une même expérience (même câble, mêmes conditions générales).

1. Cliquer sur **« Nouvelle session »**
2. Saisir un nom (sera automatiquement nettoyé : espaces → `_`, accents supprimés)
3. Optionnel : ajouter une description et un dossier de classement
4. Valider → la session apparaît dans la liste

### 7.2 Importer des images

**Option A — Upload direct :**  
Glisser-déposer des fichiers `.jpg`, `.jpeg` ou `.png` dans la session.

**Option B — Extraction depuis une vidéo :**  
1. Cliquer sur **« Importer une vidéo »**
2. Choisir le fichier vidéo (DJI, GoPro, etc.)
3. Régler l'intervalle d'extraction (par défaut : 1 frame toutes les 240 frames ≈ 1 frame/8s à 30fps)
4. L'extraction se fait **en arrière-plan** via FFmpeg — une barre de progression s'affiche
5. Les frames extraites apparaissent automatiquement dans la session

### 7.3 Annoter une image

1. Cliquer sur une image dans la liste (statut : **"À annoter"**)
2. L'éditeur s'ouvre avec l'image à pleine taille
3. Renseigner les **conditions expérimentales** :
   - `current_speed_cm_s` — vitesse de courant mesurée (cm/s) ← **champ clé pour l'IA**
   - `current_direction` — direction du courant
   - `camera_angle` — angle de la caméra
   - `wave_amplitude_cm`, `wave_length_cm`, `wave_speed_cm_s` — paramètres de houle
   - `water_depth_m`, `cable_tension_n` — paramètres physiques
   - `annotator_name`, `notes`

4. Tracer l'annotation :
   - **Mode centerline** : cliquer le long de l'axe central du câble, du haut vers le bas
   - **Mode contour** : tracer les bords gauche et droit séparément

5. Cliquer **« Enregistrer »** → l'image passe au statut **"Annotée"**

> **Auto-annotation YOLO** : si `backend/best.pt` est présent, cliquer sur **« Prédire »** pour obtenir une annotation automatique à corriger manuellement.

### 7.4 Entraîner un modèle

Depuis l'onglet **🌐 Modèle global** :

1. Sélectionner la méthode : **ResNet-18** (image → Vc) ou **MLP** (angle θ → Vc)
2. Sélectionner les sessions à utiliser (tout si vide)
3. Régler le nombre d'époques (défaut : 50 pour ResNet, 200 pour MLP)
4. Nommer le modèle et cliquer **« Entraîner »**
5. Suivre la progression en temps réel : epochs, loss train/val
6. À la fin : MAE, RMSE, courbes de loss et scatter Vc_prédit vs Vc_réel

Le modèle est sauvegardé dans `data/models/`.

### 7.5 Prédire Vc sur une nouvelle image

Depuis la même page, section **« Prédire Vc »** :
1. Sélectionner le modèle souhaité
2. Glisser ou charger une image
3. Cliquer **« Analyser »** → résultat en cm/s

---

## 8. Calculs effectués dans le backend

### 8.1 Angle du câble — ACP (`calc_cable_angle`)

La direction principale du câble est calculée par **Analyse en Composantes Principales** sur les coordonnées des points de la centerline :

1. Calcul de la matrice de covariance des coordonnées centrées
2. Décomposition en valeurs propres → le vecteur propre associé à la plus grande valeur propre donne la direction principale
3. Angle de déflexion par rapport à la verticale : `θ = arctan(|dx| / |dy|)` en degrés

Trois valeurs sont calculées et stockées dans chaque annotation :
- `cable_angle_deg` — angle ACP (régression sur tous les points)
- `cable_angle_chord_deg` — angle de la corde (premier → dernier point)
- `cable_curvature_index` — `|θ_ACP - θ_chord|` (indicateur de courbure)

### 8.2 Rééchantillonnage équidistant

Pour garantir la cohérence du dataset YOLO, tous les câbles d'une session ont le **même nombre de points** `n`. Ce nombre est déterminé automatiquement comme le minimum des annotations existantes. Si une nouvelle annotation est ajoutée avec moins de points, **toutes les annotations existantes sont rééchantillonnées rétroactivement** par interpolation curviligne.

### 8.3 Extraction de centerline depuis masque YOLO

Quand YOLO détecte un câble en mode segmentation :
1. Le masque polygonal est rasterisé en image binaire
2. **Squelettisation morphologique** (`skimage.morphology.skeletonize`) → ligne centrale 1 pixel
3. Les pixels du squelette sont ordonnés par projection sur l'axe ACP
4. 40 points équidistants sont sélectionnés

---

## 9. Scripts Python standalone

Le dossier `ModelePhysique_et_ScriptEntrainement/` contient des scripts Python utilisables **indépendamment de l'application**, en ligne de commande. Ils sont utiles pour :
- Reproduire les entraînements hors de l'application
- Adapter les hyperparamètres plus finement
- Tester rapidement sur un dataset local

### 9.1 `entrainementResNet18.py` — Entraînement ResNet-18

Script complet d'entraînement du modèle ResNet-18 pour prédire Vc à partir d'images.

**Ce que fait le script :**
- Charge les images et les annotations depuis le dataset exporté par l'application (format CSV + dossier `images/`)
- Construit un `Dataset` PyTorch personnalisé
- Entraîne un ResNet-18 modifié (tête de régression FC 512→128→1)
- Applique augmentations de données (flip, jitter couleur)
- Sauvegarde le meilleur modèle (`best_model.pth`) selon la val loss
- Affiche les courbes de loss et le scatter Vc prédit vs réel

**Utilisation :**
```bash
cd ModelePhysique_et_ScriptEntrainement

# Adapter les chemins en tête de script :
# DATASET_PATH = "chemin/vers/dataset"
# CSV_PATH = "chemin/vers/annotations.csv"

python entrainementResNet18.py
```

**Dépendances :** `torch`, `torchvision`, `Pillow`, `pandas`, `numpy`

**Hyperparamètres modifiables en tête de fichier :**
```python
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
```

---

### 9.2 `entrainementMLP.py` — Entraînement MLP (θ → Vc)

Script d'entraînement d'un réseau MLP (Multi-Layer Perceptron) qui prédit Vc à partir du **seul angle ACP θ** extrait des annotations.

**Architecture du MLP :**
```
θ (1 scalaire) → FC(1→64) → ReLU → FC(64→64) → ReLU → FC(64→1) → Vc
```

**Ce que fait le script :**
- Lit les paires `(θ, Vc)` depuis les fichiers `.json` d'annotation de l'application
- Normalise θ et Vc (z-score)
- Entraîne le MLP sur ces paires
- Sauvegarde le modèle (`.pth`) et les paramètres de normalisation
- Affiche la courbe Vc = f(θ) : annotations vs prédictions MLP vs régression linéaire

**Utilisation :**
```bash
python entrainementMLP.py
# Modifier ANNOTATIONS_DIR en tête de script pour pointer vers data/sessions/
```

**Pourquoi ce modèle est intéressant :**  
Le MLP est très léger (quelques centaines de paramètres). Son entrée θ est directement liée à la physique du câble. Sur un dataset limité, il peut surpasser ResNet-18 car il ne souffre pas du sur-apprentissage sur les détails visuels non pertinents.

---

### 9.3 `pred_Vc_MLP_image.py` — Prédiction MLP sur image inconnue

Script de prédiction complet du pipeline MLP sur une nouvelle image, en utilisant YOLO pour détecter le câble et calculer son angle.

**Pipeline complet :**
```
Image → YOLOv8 (segmentation) → Masque → Squelettisation → ACP → θ → MLP → Vc (cm/s)
```

**Utilisation :**
```bash
python pred_Vc_MLP_image.py \
  --image chemin/vers/image.jpg \
  --mlp chemin/vers/mlp_theta_to_vc.pth \
  --yolo chemin/vers/best.pt
```

**Paramètres :**

| Argument | Description |
|----------|-------------|
| `--image` | Chemin vers l'image à analyser |
| `--mlp` | Chemin vers le modèle MLP `.pth` entraîné |
| `--yolo` | Chemin vers le modèle YOLO `best.pt` |
| `--conf` | Seuil de confiance YOLO (défaut : 0.5) |

**Sortie :** affiche θ calculé (°) et Vc estimée (cm/s).

---

### 9.4 `predResNet18.py` — Prédiction ResNet-18 sur image

Script de prédiction simple : charge un modèle ResNet-18 `.pth` et prédit Vc sur une image.

**Utilisation :**
```bash
python predResNet18.py \
  --image chemin/vers/image.jpg \
  --model chemin/vers/modele.pth
```

---

### 9.5 `modelephysique.py` — Modèle physique analytique

Script de modélisation physique du câble basé sur l'**équation d'équilibre mécanique** d'un câble soumis à un courant uniforme.

**Principe physique :**  
Pour un câble de longueur `L`, de masse linéique `m`, soumis à une force de traînée hydrodynamique, l'angle de déflexion à la base θ est lié à la vitesse du courant Vc par la relation :

```
Fd = ½ · ρ · Cd · D · L · Vc²
tan(θ) = Fd / (T₀ + m·g·L)
```

où `ρ` est la densité de l'eau, `Cd` le coefficient de traînée, `D` le diamètre du câble, `T₀` la tension initiale.

**Ce que fait le script :**
- Calcule θ = f(Vc) pour une gamme de vitesses selon les paramètres physiques du câble
- Trace la courbe θ = f(Vc) et Vc = f(θ) du modèle physique
- Superpose les données d'annotation réelles pour comparer modèle physique vs mesures
- Permet de **valider** que les annotations sont cohérentes avec la physique

**Utilisation :**
```bash
python modelephysique.py
# Adapter les paramètres physiques en tête de script (masse, diamètre, Cd, profondeur...)
```

**Pourquoi ce script est important :**  
Le modèle physique sert de **baseline** et de validation. Si la courbe Vc = f(θ) apprise par le MLP ou ResNet-18 s'écarte fortement du modèle physique, cela indique soit un problème dans les annotations, soit des conditions expérimentales non prises en compte (houle, inclinaison caméra, etc.).

---

## 10. Modèle physique du câble

Le câble de mouillage est modélisé comme une structure flexible soumise à :
- Son poids propre dans l'eau (masse linéique apparente `m_a = m - ρ·V`)
- La force de traînée hydrodynamique (traînée de forme + frottement visqueux)
- La tension de rappel à l'ancrage

La relation entre l'angle de déflexion θ et la vitesse de courant Vc dépend des paramètres :
- Longueur du câble `L` (m)
- Diamètre `D` (m)
- Masse linéique dans l'eau (kg/m)
- Coefficient de traînée `Cd` (typiquement 1.0–1.2 pour un câble cylindrique)
- Tension initiale `T₀` (N)

Le script `modelephysique.py` permet de tracer cette courbe théorique et de la comparer aux données expérimentales collectées via l'application.

---

## 11. API REST — référence complète

L'API complète est consultable via Swagger : `http://localhost:8000/docs`

### Sessions

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/sessions` | Liste toutes les sessions |
| `POST` | `/api/sessions` | Crée une session |
| `GET` | `/api/sessions/{name}` | Détails d'une session |
| `PATCH` | `/api/sessions/{name}` | Modifie description/dossier |
| `DELETE` | `/api/sessions/{name}` | Supprime une session |
| `POST` | `/api/sessions/batch-move` | Déplace plusieurs sessions dans un dossier |

### Images & Annotation

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/sessions/{name}/images` | Upload d'images |
| `GET` | `/api/sessions/{name}/images/{filename}` | Récupère une image |
| `DELETE` | `/api/sessions/{name}/images/{filename}` | Supprime image + annotation |
| `POST` | `/api/sessions/{name}/images/{filename}/ignore` | Marque comme ignorée |
| `GET` | `/api/sessions/{name}/images/{filename}/predict` | Auto-annotation YOLO |
| `GET` | `/api/sessions/{name}/annotations/{stem}` | Récupère une annotation |
| `POST` | `/api/sessions/{name}/annotations/{stem}` | Sauvegarde une annotation |
| `GET` | `/api/sessions/{name}/last-conditions` | Dernières conditions saisies |

### Vidéo

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/sessions/{name}/video` | Upload vidéo + extraction frames |
| `GET` | `/api/sessions/{name}/video/progress` | Progression extraction |

### Statistiques & Export

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/sessions/{name}/statistics` | Statistiques complètes |
| `GET` | `/api/sessions/{name}/export/download` | Dataset ZIP (session) |
| `GET` | `/api/export/global/download` | Dataset ZIP global |
| `GET` | `/api/sessions/{name}/angle-vc-data` | Points θ/Vc |

### Entraînement IA

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/sessions/{name}/train` | Entraînement ResNet-18 (session) |
| `GET` | `/api/sessions/{name}/train/progress` | Progression |
| `POST` | `/api/train/global` | Entraînement global (ResNet ou MLP) |
| `GET` | `/api/train/global/progress` | Progression entraînement global |
| `GET` | `/api/train/global/angle-vc-data` | Points θ/Vc globaux |
| `GET` | `/api/models` | Liste les modèles `.pth` |
| `GET` | `/api/models/{filename}/download` | Télécharge un modèle |
| `GET` | `/api/models/{model_name}/visualize` | Préd. vs réel sur dataset |
| `POST` | `/api/predict` | Prédit Vc sur une image |

---

## 12. Structure des fichiers de données

```
data/
├── sessions/
│   └── {session_name}/
│       ├── session.json              ← métadonnées de la session
│       ├── images/
│       │   ├── frame_000001.jpg
│       │   └── ...
│       ├── annotations/
│       │   ├── frame_000001.json     ← annotation complète par image
│       │   └── ...
│       └── labels/
│           ├── frame_000001.txt      ← label YOLO (format polygone normalisé)
│           └── ...
└── models/
    ├── {session_name}_vc_model.pth   ← modèle ResNet-18 par session
    ├── global_vc_model.pth           ← modèle ResNet-18 global
    └── mlp_global_model.pth          ← modèle MLP global
```

### Format `{stem}.json` (annotation)

```json
{
  "points": [{"x": 512.3, "y": 45.1}, {"x": 514.7, "y": 89.3}, "..."],
  "n_points_normalized": 40,
  "image_width": 1920,
  "image_height": 1080,
  "annotation_mode": "centerline",
  "cable_angle_deg": 12.453,
  "cable_angle_chord_deg": 12.107,
  "cable_curvature_index": 0.346,
  "conditions": {
    "current_speed_cm_s": 10.5,
    "current_direction": "est",
    "camera_angle": "90",
    "wave_amplitude_cm": 2.0,
    "wave_length_cm": 50.0,
    "wave_speed_cm_s": 30.0,
    "water_depth_m": 1.5,
    "cable_tension_n": 25.0,
    "annotator_name": "Amine",
    "notes": "Bonne visibilité"
  },
  "saved_at": "2025-06-01T10:15:00"
}
```

---

## 13. Entraînement IA et modèles

### Depuis l'application

1. Aller dans **🌐 Modèle global**
2. Choisir la méthode : **ResNet-18** ou **MLP**
3. Sélectionner les sessions, régler les époques, nommer le modèle
4. Cliquer **« Entraîner »** → suivi en temps réel
5. Résultats : MAE, RMSE, courbes de loss, scatter, graphique Vc = f(θ)

### Depuis les scripts standalone

Voir section [9. Scripts Python standalone](#9-scripts-python-standalone).

### Prédiction via l'API

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "model_name=global_vc_model" \
  -F "file=@mon_image.jpg"
# → {"vitesse_estimee": 12.34, "model_used": "global_vc_model"}
```

---

## 14. Auto-annotation YOLO

Si `backend/best.pt` est présent (modèle YOLOv8 segmentation entraîné sur câbles) :

1. Ouvrir une image dans l'éditeur → cliquer **« Prédire »**
2. Le backend applique YOLO → extrait le masque du câble
3. Le masque est squelettisé → 40 points de centerline
4. Les points s'affichent et peuvent être corrigés manuellement

**Entraîner son propre modèle YOLO :**
```bash
# Exporter le dataset depuis l'application (Export ZIP)
# Décompresser → dossier dataset/

yolo segment train \
  data=dataset/dataset.yaml \
  model=yolov8n-seg.pt \
  epochs=100 \
  imgsz=640

# Le modèle entraîné se trouve dans runs/segment/train/weights/best.pt
cp runs/segment/train/weights/best.pt backend/best.pt
```

Vérifier le statut YOLO :
```bash
curl http://localhost:8000/api/yolo/status
# {"model_path": "...", "model_exists": true, "model_loaded": true}
```

---

## 15. Export des données

### Export par session

Génère un ZIP avec images + labels YOLO + annotations JSON + CSV :
```
GET /api/sessions/{name}/export/download
```

### Export global

Combine toutes les sessions (noms préfixés `{session}__` pour éviter collisions) :
```
GET /api/export/global/download
```

---

## 16. Étendre / modifier l'application

### Ajouter un champ de condition

1. Dans `backend/main.py` : ajouter dans la liste `CSV_FIELDNAMES`
2. Dans `frontend/src/` : ajouter le champ dans le formulaire de l'éditeur
3. Le champ est automatiquement sauvegardé dans `conditions` du JSON et exporté dans le CSV

### Changer le backbone IA

Dans `run_train_global()` dans `main.py` :
```python
# Remplacer ResNet-18 par ResNet-50 :
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
```

### Modifier l'architecture MLP

Dans `entrainementMLP.py` ou dans `run_train_global()` pour la méthode MLP :
```python
# Exemple : ajouter une couche cachée
nn.Linear(1, 64), nn.ReLU(),
nn.Linear(64, 128), nn.ReLU(),
nn.Linear(128, 64), nn.ReLU(),
nn.Linear(64, 1)
```

### Modifier les ports

- Backend : changer `--port 8000` dans `start.sh`
- Frontend : modifier `vite.config.ts` et l'URL de l'API dans `frontend/src/lib/api.ts`

---

## 17. FAQ et dépannage

**L'application ne démarre pas — erreur `uvicorn`**  
→ Vérifier Python 3.10+ : `python --version`  
→ Réinstaller : `pip install -r requirements.txt`

**Le frontend ne s'ouvre pas**  
→ Vérifier Node.js ≥ 18 : `node --version`  
→ Relancer `cd frontend && npm install`

**L'extraction vidéo ne fonctionne pas**  
→ Vérifier FFmpeg : `ffmpeg -version`  
→ Installer via `brew install ffmpeg` (macOS) ou `apt install ffmpeg` (Linux)

**L'entraînement IA échoue — "PyTorch non installé"**  
→ `pip install torch torchvision`

**L'auto-annotation YOLO retourne "modèle non disponible"**  
→ Placer `best.pt` dans `backend/best.pt`  
→ Vérifier : `curl http://localhost:8000/api/yolo/status`

**Les annotations existantes changent de nombre de points**  
→ Normal. Le rééchantillonnage rétroactif garantit la cohérence du dataset YOLO. Toutes les annotations d'une session ont toujours le même nombre de points.

**Où sont stockées les données ?**  
→ Dans `data/sessions/` à la racine. Ce dossier n'est pas versionné (`.gitignore`). **Sauvegarder ce dossier séparément avant toute manipulation.**

**Le modèle MLP donne de mauvais résultats**  
→ Vérifier que `cable_angle_deg` est bien renseigné dans toutes les annotations.  
→ Comparer avec la courbe du modèle physique (`modelephysique.py`) : si les annotations s'en écartent fortement, les conditions expérimentales sont peut-être non homogènes (houle, angle caméra variable).

---


Développé par **Mohamed-Amine Boutahri** (stagiaire L3 Physique, Université de Toulon) pour le **Laboratoire COSMER**.  
Usage interne recherche. Contact : mohamed-amine-bouthari@etud.univ-tln.fr
