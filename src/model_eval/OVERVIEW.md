# model_eval — Overview

TL;DR — Module d'évaluation pour mesurer le domain gap syn→real. Entraîne modèles de segmentation sémantique (DeepLab, Segformer) sur données synthétiques améliorées et évalue sur données réelles (Cityscapes) pour quantifier l'amélioration.

## Rôle et contexte

Le module `model_eval` fournit les outils pour évaluer quantitativement l'impact des techniques d'amélioration (GAN, diffusion, etc.) :
- **Entraîner** modèles de segmentation sémantique sur données synthétiques (ou augmentées).
- **Évaluer** sur données réelles pour mesurer accuracy, F1, IoU.
- **Comparer** performances baseline vs. données améliorées.
- **Quantifier** domain gap et progrès (métrique objective).

Démarche :
1. Entraîner segmenter sur syn (baseline ou augmentées).
2. Tester sur real → mesurer F1/IoU.
3. Comparer baseline F1 vs. augmentée F1 → gain quantifié.

## Structure & fichiers clés

### Entraînement et modèles

- **`train_deeplab.py`** : entraîne DeepLab v3+ pour segmentation sémantique.
  - Charge dataset (Cityscapes, GTA, SYNTHIA).
  - Fine-tune DeepLabHead sur labels réduits.
  - Évalue F1, mIoU, roc_auc sur validation/test.
  - Sauvegarde modèle entraîné.
  - Supports mixed precision (fp16) et DataParallel.

- **`train_segmenter.py`** : entraîne SegformerForSemanticSegmentation (Hugging Face).
  - Alternative légère à DeepLab.
  - Utilise Trainer API (Hugging Face) pour entraînement unifié.
  - Logs automatiques (wandb, tensorboard).
  - Supports gradient accumulation et multi-GPU via Accelerate.

- **`labels.py`** : définitions labels Cityscapes.
  - `labels` : liste complète (34 labels).
  - `reduced_labels` : labels évalués seulement (~10-20, excluding stuff classes).
  - Mappages : trainIds, colors, names.

### Utilitaires

- **`clip_embeddings.py`** : extrait embeddings CLIP pour évaluation perceptuelle.
  - Utile pour comparer qualité images real vs. générées.

- **`arniqa_embeddings.py`** : qualité image via ARNIQA (blind image quality assessment).
  - Évalue niveaux artefacts/blur/bruit.

- **`mmd.py`** : Maximum Mean Discrepancy.
  - Mesure distance distribution entre domaines.
  - Indicateur domain gap.

- **`patent/`** : fichiers spécialisés (si présents).

- **`diffusers/`** : intégration diffusers (si utilisée).

- **`Dockerfile`** : conteneurisation.
- **`pyproject.toml`** : dépendances.
- **`tests/`** : tests unitaires.

## Workflow complet

```
1. Préparer données
   ├─ Images synthétiques entraînement : /data/train/
   │  (GTA, SYNTHIA, ou données augmentées par GAN/ControlNet)
   ├─ Images réelles test : /data/real/
   │  (Cityscapes validation set)
   └─ Labels (segmentation maps) : /data/labels/

2. Entraîner modèle segmentation
   ├─ Charger dataset avec augmentations
   ├─ Fine-tune DeepLab ou Segformer
   ├─ Valider sur test réel tous les N steps
   └─ Sauvegarder checkpoint meilleur F1
   
3. Évaluer
   ├─ Charger modèle entraîné
   ├─ Prédire sur real images
   ├─ Calculer F1, mIoU, precision, recall par classe
   ├─ Comparer vs. baseline
   └─ Résumer gain quantifié
   
4. Reporter résultats
   ├─ CSV avec métriques par classe
   ├─ Courbes loss/F1 vs epochs
   ├─ Matrice confusion real predictions
   └─ Conclure si améliorations significatives
```

## Format d'entrée/sortie

### Entrée 1 : Dataset entraînement (synthétique)

Structure arborescente Cityscapes-compatible :

```
/data/train/
├── leftImg8bit/                # Images RGB
│   └── train/
│       ├── image_0.png
│       ├── image_1.png
│       └── ...
└── gtFine/                     # Segmentation maps
    └── train/
        ├── image_0_trainIds.png  (uint8, labels 0-19)
        ├── image_1_trainIds.png
        └── ...
```

Format labels : uint8, trainIds (0-18 = classe, 255 = ignore).

### Entrée 2 : Dataset test (réel Cityscapes)

Même structure pour `/data/real/` ou `/data/cityscapes/`:

```
/data/real/
├── leftImg8bit/
│   └── val/
│       └── *.png
└── gtFine/
    └── val/
        └── *_trainIds.png
```

### Sortie : Métriques évaluation

Fichiers générés dans `--exp_folder` :

```
/output/experiment1/
├── model.pth                   # Checkpoint entraîné
├── metrics.csv                 # F1, mIoU, precision, recall par classe
├── loss_curve.png             # Courbe loss vs epochs
├── f1_curve.png               # F1 vs epochs
├── confusion_matrix.png       # Matrice confusion real predictions
└── summary.txt                # Rapport texte synthétique
```

## Utilisation

### 1. Entraîner avec DeepLab

```bash
cd src/model_eval

python train_deeplab.py \
  --in_train_dataset /path/to/synthetic/train \
  --in_test_dataset /path/to/real/val \
  --exp_folder /path/to/output \
  --batch_size 8 \
  --epochs 20 \
  --learning_rate 1e-4 \
  --num_workers 8
```

### 2. Entraîner avec Segformer (Hugging Face)

```bash
cd src/model_eval

python train_segmenter.py \
  --in_train_dataset /path/to/synthetic/train \
  --in_test_dataset /path/to/real/val \
  --exp_folder /path/to/output \
  --batch_size 4 \
  --epochs 12 \
  --learning_rate 5e-5
```

Supports logging via `--report_to wandb` ou `tensorboard`.

### 3. En Docker

```bash
docker build -t model_eval .
docker run --rm --gpus all \
  -v /data:/data \
  -v /output:/output \
  model_eval python train_deeplab.py \
  --in_train_dataset /data/train \
  --in_test_dataset /data/real \
  --exp_folder /output
```

## Modèles segmentation

### DeepLab v3+

- **Base** : ResNet50 ou ResNet101 (backbone).
- **Tête** : DeepLabHead (ASPP + decoder).
- **Entrée** : images RGB (3 canaux).
- **Sortie** : logits pour N classes.
- **Avantages** : bien établi, stable, large réceptif.
- **Inconvénients** : plus gros que Segformer.

### Segformer

- **Base** : HierarchicalVisionTransformer (SegformerImageProcessor + model).
- **Avantages** : plus léger, vision transformer moderne, logique robuste.
- **Inconvénients** : moins matures que DeepLab en 2024.

## Labels & mapping

### Labels Cityscapes

34 labels total (void + stuff + thing) :
- **Stuff** (non-instanciable) : road, sidewalk, sky, building, wall, fence, vegetation, terrain, sky.
- **Thing** (instanciable) : person, rider, car, truck, bus, train, motorcycle, bicycle, etc.

### Reduced Labels

Sous-ensemble (~10-18 labels) pour évaluation simplifiée (exclut stuff rares) :
- Utilisé par défaut dans entraînement.
- Mapping : `reduced_labels` dans `labels.py`.

### Remapping labels

Fonction `reduce_labels()` dans `train_segmenter.py` :
- Convertit labels complets → reduced labels.
- Labels non-évalués → 0 (ignore/background).

## Métriques d'évaluation

| Métrique | Formule | Usage |
|----------|---------|-------|
| **F1-Score** | 2 × (precision × recall) / (precision + recall) | Moyenne harmonic precision/recall. |
| **mIoU** | moyenne IoU par classe | Métrique standard segmentation. |
| **Precision** | TP / (TP + FP) | % prédictions positives correctes. |
| **Recall** | TP / (TP + FN) | % vrais positifs détectés. |
| **ROC-AUC** | Area Under Curve ROC | Discrimination classe vs. non-classe. |

Calculées par classe puis moyennées.

## Intégration avec syn2real

`model_eval` est la **métrique finale** d'évaluation :

```
Flux complet :
1. prepare_synthia.py (convertir + captions)
   ↓
2. controlnet_sdxl/train.py (entraîner ControlNet)
   ↓
3. controlnet_sdxl/inference.py (générer images augmentées)
   ↓
4. deepfloyd_scaler/scale.py (upscaler optionnel)
   ↓
5. model_eval/train_deeplab.py ← ÉVALUATION
   ├─ Entraîner segmenter sur images augmentées
   └─ Évaluer F1/mIoU sur real Cityscapes
   ↓
6. Comparer F1 baseline vs. augmentées
   └─ Quantifier amélioration (ex: +5% F1)
```

## Améliorations / TODOs

- [ ] Support loss functions personnalisés (focal loss, etc.).
- [ ] Validation set séparé avec early stopping.
- [ ] Métriques panoptic (instance-aware, stuff-aware).
- [ ] Visualization per-class performance breakdown.
- [ ] Ensemble methods (multiple models, voting).
- [ ] Knowledge distillation (teacher-student).
- [ ] Adversarial evaluation robustness.
- [ ] Explainability (Grad-CAM, attention maps).

---

Le module fournit une évaluation objective et quantifiée du domain gap. Meilleur pour comparer approches systematiquement.
