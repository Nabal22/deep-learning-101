# Fiche Résumé — Classification d'images & Transfer Learning

**TP :** Dogs vs Cats (Kaggle) avec VGG16 pré-entraîné sur ImageNet

---

## 1. Objectif

Classifier des images de **chats et chiens** (2 classes) en réutilisant un réseau de neurones profond (VGG16) déjà entraîné sur ImageNet (1000 classes).

**Idée clé :** Plutôt qu'entraîner un réseau from scratch (long, coûteux), on réutilise les features apprises sur un grand dataset → **Transfer Learning**.

---

## 2. Dataset

| Split | Taille |
|-------|--------|
| Train | 23 000 images |
| Valid | 2 000 images |
| Test  | 12 500 images (non labelisées) |

Structure de dossiers exploitée par `datasets.ImageFolder` :
```
dogscats/
├── train/
│   ├── cats/   (11 500 imgs)
│   └── dogs/   (11 500 imgs)
└── valid/
    ├── cats/   (1 000 imgs)
    └── dogs/   (1 000 imgs)
```

---

## 3. Préparation des données

Les modèles pré-entraînés ImageNet attendent :
- Images **224×224** en RGB
- Normalisées avec `mean=[0.485, 0.456, 0.406]` et `std=[0.229, 0.224, 0.225]`

```python
imagenet_format = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**DataLoader** : charge les images en mini-batches, en parallèle (`num_workers`), avec mélange aléatoire (`shuffle=True` pour l'entraînement).

---

## 4. Modèle : VGG16

```
VGG16
├── features   : 13 couches de convolution  → extraction de patterns visuels
├── avgpool    : réduction de taille
└── classifier : 3 couches Dense (FC)
                 └── Linear(4096, 1000)  ← à remplacer !
```

| Type de couche | Rôle |
|----------------|------|
| Convolution | Détecte des patterns locaux (bords, textures…) |
| Pooling | Réduit la taille, améliore l'invariance |
| Dense (FC) | Combine les patterns globalement pour décider |

---

## 5. Stratégie Transfer Learning

```
① Charger VGG16 pré-entraîné (ImageNet)
② Geler tous les paramètres  →  requires_grad = False
③ Remplacer la dernière couche  →  Linear(4096, 1000)  ⟹  Linear(4096, 2)
④ Entraîner SEULEMENT cette nouvelle couche
```

```python
for param in model_vgg.parameters():
    param.requires_grad = False

model_vgg.classifier._modules['6'] = nn.Linear(4096, 2)
```

**Paramètres entraînés :** seulement `4096 × 2 + 2 = 8 194` (sur ~138M au total) !

---

## 6. Entraînement

**Loss :** `CrossEntropyLoss` (classification multi-classes)

**Optimiseur :** `SGD(lr=0.001)`

**Boucle d'entraînement** (pour chaque batch) :

```
Pour chaque batch :
  1. Forward  : outputs = model(inputs)
  2. Loss     : loss = criterion(outputs, classes)
  3. Backward : loss.backward()
  4. Update   : optimizer.step()
```

$$\hat{y} = f_\theta(x) \xrightarrow{\mathcal{L}(\hat{y}, y)} \nabla_\theta \mathcal{L} \xrightarrow{\text{SGD}} \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

---

## 7. Évaluation

**Résultat :** ~97.9% de précision sur le jeu de validation après seulement **2 époques** !

Analyses réalisées :

| Analyse | Description |
|---------|-------------|
| Prédictions correctes | Images bien classifiées |
| Prédictions incorrectes | Cas d'erreur du modèle |
| + Confiantes correctes | Probabilité max, bonne réponse |
| + Confiantes incorrectes | Probabilité max, mauvaise réponse |
| + Incertaines | Probabilité ≈ 0.5 (hésitation chien/chat) |

---

## 8. Concepts clés à retenir

| Concept | Définition |
|---------|------------|
| **Transfer Learning** | Réutiliser un modèle pré-entraîné sur une nouvelle tâche |
| **Fine-tuning** | Ré-entraîner seulement certaines couches |
| `requires_grad=False` | Gèle les poids, empêche leur mise à jour |
| **DataLoader** | Charge les données en batches de façon efficace |
| **Softmax** | Convertit les sorties brutes en probabilités (somme = 1) |
| **CrossEntropyLoss** | Loss standard pour la classification |

---

## 9. Pourquoi ça marche si bien ?

VGG16 a été entraîné sur ImageNet qui **contient déjà des milliers de photos de chats et chiens**. Les couches de convolution ont donc déjà appris des features très discriminantes pour cette tâche. Il suffit d'apprendre à les combiner pour distinguer les 2 classes → tâche triviale pour une seule couche Dense.

> *« On a tué une mouche avec un marteau-pilon »* — mais c'est instructif pour apprendre le pipeline complet d'un projet deep learning !
