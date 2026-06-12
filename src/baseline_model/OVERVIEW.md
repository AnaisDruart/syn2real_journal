# baseline_model — Overview

TL;DR — Module de baseline pour évaluer la transferabilité domaine syn→real. Entraîne ResNet50 sur données synthétiques et teste sur données réelles pour mesurer le domain gap.

## Rôle et contexte

Le module `baseline_model` établit une baseline de performance en effectuant un **benchmark domain gap** :
- Entraîne ResNet50 sur images **synthétiques** (GTA, SYNTHIA, etc.)
- Évalue le modèle sur images **réelles** (Cityscapes validation)
- Mesure la dégradation de performance (syn→real gap)
- Fournit une métrique de référence pour évaluer les améliorations apportées par les modèles de génération d'images.

## Structure & fichiers clés

- `main.py` : script d'orchestration principal (`benchmark()`).
  - Parse les arguments CLI (`--train_path`, `--val_path`, `--model_path`).
  - Charge, entraîne, évalue et sauvegarde le modèle.
  
- `data_tools.py` : gestion des données.
  - `CustomImageDataset` : classe Dataset torch pour charger images et labels.
  - `split_train_val()` : divise dataset synthétique en train/val.
  - `preprocess_resnet50()` : normalisation et augmentations ResNet50.
  
- `model_tools.py` : manipulation du modèle.
  - `load_resnet50(num_classes)` : charge ResNet50 pré-entraîné, adapte FC layer.
  - `train()` : boucle d'entraînement avec gradient scaler (mixed precision).
  - `evaluate_model()` : inférence sur validation/test set.
  
- `eval_metrics_tools.py` : métriques d'évaluation.
  - `conf_mtx()` : matrice de confusion + rapport de classification (précision, recall, F1).
  - `GradCamInspector` : visualisations Grad-CAM pour comprendre les prédictions.
  
- `config.py` : tous les paramètres (chemins, hyperparamètres, device, logging).
- `Dockerfile` : conteneurisation.
- `pyproject.toml` : dépendances.
- `tests/` : tests unitaires pour `data_tools.py`, `eval_metrics_tools.py`, `model_tools.py`.

## Workflow complet

```
1. Charger données synthétiques (GTA, SYNTHIA)
   ↓
2. Splitter en train (85%) / val (15%)
   ↓
3. Charger ResNet50 pré-entraîné, adapter FC layer → num_classes
   ↓
4. Entraîner sur données synthétiques
   ↓
5. Évaluer sur données synthétiques (validation set)
   ↓
6. Charger données réelles (Cityscapes)
   ↓
7. Évaluer sur données réelles
   ↓
8. Comparer performances → mesurer domain gap
   ↓
9. Sauvegarder : model weights, confusion matrix, classification report
```

## Utilisation

### Configuration via `config.py`

Paramètres clés (avant exécution ou ligne de commande) :

```python
TRAIN_PATH = "/data/syn2real/train/"     # Dataset synthétique (GTA, SYNTHIA)
VAL_PATH = "/data/syn2real/validation/"  # Dataset réel (Cityscapes)
MODEL_PATH = "/models/resnet50_latest.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 1 (recommandé: 10+ pour entraînement réel)
LR = 0.001
OPTIMIZER = optim.Adam
CRITEREON = nn.CrossEntropyLoss()
DEVICE = auto-détecté (cuda:0 ou cpu)
```

### Exécuter le benchmark

```bash
cd src/baseline_model

# Avec chemins par défaut (définis dans config.py)
python main.py

# Avec chemins personnalisés
python main.py --train_path /path/to/syn/data --val_path /path/to/real/data --model_path /path/to/save/model.pth
```

Ou en conteneur Docker :
```bash
docker build -t baseline_model .
docker run --rm -v /data:/data -v /models:/models -v /out:/out baseline_model
```

### Résultats

Fichiers générés dans `/out/` (selon `config.py`) :

- **log.txt** : historique loss/epoch pendant entraînement.
- **confusion_matrix.png** : heatmap de la matrice de confusion (test réel).
- **classification_report.txt** : rapport détaillé (precision, recall, F1 par classe + moyenne).

Sortie console : matrices de confusion et rapports pour val (syn) et test (réel).

## Format des données d'entrée

Le module attend une **structure arborescente** :
```
/data/syn2real/
├── train/              # Images synthétiques avec sous-dossiers par classe
│   ├── class0/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class1/
│   └── ...
│
└── validation/         # Images réelles (même structure)
    ├── class0/
    ├── class1/
    └── ...
```

La fonction `split_train_val()` scanne les sous-dossiers et assigne les labels entiers (0, 1, 2, ...) à chaque classe. L'ordre des dossiers est important pour la cohérence.

**Par défaut** : 12 classes (hardcodé dans `split_train_val(classes=list(range(12)))`). À adapter si dataset a nombre différent.

## Hyperparamètres recommandés

| Paramètre | Valeur par défaut | Recommandé | Notes |
|-----------|------------------|-----------|-------|
| `NUM_EPOCHS` | 1 | 10–50 | Plus d'epochs = meilleure convergence. |
| `BATCH_SIZE` | 32 | 32–64 | Ajuster selon GPU VRAM. |
| `LR` (learning rate) | 0.001 | 0.0001–0.001 | Baisser si overfitting. |
| `NUM_WORKERS` | 2 | 4–8 | Augmenter pour accélération I/O. |

## Optimisations

- **Mixed Precision** : le code utilise `GradScaler` et `autocast` pour accélérer entraînement et réduire usage mémoire.
- **Gradient Accumulation** : si BATCH_SIZE > VRAM disponible, adapter le code.
- **Early Stopping** : pas implémenté; à ajouter si overfitting sur validation set.

## Intégration avec syn2real

`baseline_model` sert de **point de référence** pour :
- Mesurer le domain gap initial (avant amélioration).
- Évaluer si améliorations (GAN, diffusion, etc.) réduisent bien le gap.
- Permettre comparaisons quantitatives et reproductibilité.

Typiquement, le flux est :
1. Préparer dataset synthétique (ex: SYNTHIA via `prepare_synthia.py`).
2. Exécuter `baseline_model` → noter accuracy/F1 sur real data.
3. Appliquer améliorations (GAN, diffuseurs, etc.) → régénérer dataset augmenté.
4. Réexécuter `baseline_model` avec dataset augmenté → comparer perfs.

## Améliorations / TODOs

- [ ] Ajouter Early Stopping basé sur validation loss.
- [ ] Support multi-GPU (DataParallel ou DistributedDataParallel).
- [ ] Logging structuré (MLflow, TensorBoard, Weights & Biases).
- [ ] Paramètres CLI pour tous les hyperparamètres (learning rate, batch size, epochs).
- [ ] Support de modèles autres que ResNet50 (EfficientNet, ViT, etc.).
- [ ] Sauvegarde checkpoints intermédiaires (résumer entraînement si interrompu).
- [ ] Visualisation des courbes loss/accuracy au cours du temps.

---

Le module est conçu pour être simple, reproductible et servir de baseline. Toute modification ou ajout peut être implémenté sur demande.
