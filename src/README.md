# Source code organization

TL;DR — Le projet syn2real fournit un pipeline complet pour **réduire le domain gap synthétique→réaliste** en utilisant des modèles de génération d'images (ControlNet, CycleGAN-Turbo, DeepFloyd) et en mesurant objectivement l'amélioration via segmentation sémantique.

## 📋 Table des matières

- [Structure & Modules](#structure--modules)
- [Ordre d'utilisation (Workflow recommandé)](#ordre-dutilisation-workflow-recommandé)
- [Dépendances inter-modules](#dépendances-inter-modules)
- [Détails par module](#détails-par-module)
- [Tracking expériences (MLflow)](#tracking-expériences-mlflow)

---

## Structure & Modules

```
src/
├── baseline_model/          # Établir baseline domain gap (ResNet50 benchmark)
├── blip_caption/            # Générer captions BLIP pour images
├── controlnet_finetuning/   # Fine-tuner ControlNet sur données paires
├── controlnet_sdxl/         # Entraîner/utiliser ControlNet SDXL (principal)
├── deepfloyd_scaler/        # Upscaler images haute résolution (DeepFloyd IF-II + SD x4)
├── gan/                     # Alternative GAN : CycleGAN-Turbo (données non appairées)
├── model_eval/              # Évaluer domain gap via segmentation sémantique
└── README.md                # Ce fichier
```

**Voir aussi** (à la racine du projet) :
- `mlflow/` — Tracking automatique expériences + métriques MLflow.
- `.gitignore` — Ignorer datasets et sorties.
- `docs/` — Documentation générale (acceleration, ControlNet, SOTA, etc.).

---

## Ordre d'utilisation (Workflow recommandé)

### 🔴 Phase 0 : Setup & Baseline

**Objectif** : établir performance baseline sur données synthétiques brutes.

1. **`baseline_model/`** — Benchmark domain gap initial
   - **Rôle** : Entraîner ResNet50 sur données synthétiques (GTA/SYNTHIA).
   - **Sortie** : F1/accuracy sur données réelles (Cityscapes) — baseline pour comparaison.
   - **Commande** :
     ```bash
     python src/baseline_model/main.py \
       --train_path /path/to/synthetic/data \
       --val_path /path/to/real/data
     ```
   - **Voir** : [baseline_model/OVERVIEW.md](baseline_model/OVERVIEW.md)

**Résultat attendu** : F1 baseline (ex: 45%) sur real data.

---

### 🟡 Phase 1 : Préparation données

**Objectif** : préparer dataset synthétique avec labels et captions textuelles.

#### Étape 1a : Convertir dataset synthétique (si SYNTHIA)

2. **`controlnet_sdxl/prepare_synthia.py`** — Convertir SYNTHIA → format Cityscapes
   - **Rôle** : SYNTHIA (uint16 labels) → Cityscapes (trainIds uint8 + visualisations).
   - **Sortie** : `{output_root}/leftImg8bit/` + `gtFine/` + `metadata.jsonl`.
   - **Commande** :
     ```bash
     python src/controlnet_sdxl/prepare_synthia.py \
       --synthia_root /path/to/raw/synthia \
       --output_root /path/to/synthia_prepared \
       --device cpu
     ```
   - **Voir** : [controlnet_sdxl/OVERVIEW.md](controlnet_sdxl/OVERVIEW.md#flux-de-données-exemple-synthia)

**Résultat attendu** : `synthia_prepared` avec images RGB + trainIds + metadata.jsonl.

#### Étape 1b : Générer captions BLIP

3. **`blip_caption/`** — Générer captions textuelles (si manquantes)
   - **Rôle** : Générer descriptions textuelles des images via BLIP.
   - **Entrée** : JSON avec liste images `{"image": "...", "dataset": "synthetic"}`.
   - **Sortie** : JSON enrichi avec `"text": "a synthetic picture of ..."`.
   - **Commande** :
     ```bash
     cd src/blip_caption
     python main.py  # (configure DATA_PATH/FILE_NAME dans config.py)
     ```
   - **Voir** : [blip_caption/OVERVIEW.md](blip_caption/OVERVIEW.md)

**Note** : `prepare_synthia.py` inclut déjà génération BLIP intégrée → optionnel.

**Résultat attendu** : `metadata.jsonl` avec captions BLIP.

---

### 🟢 Phase 2 : Génération d'images améliorées

**Objectif** : générer nouvelles images réalistes à partir des synthétiques.

#### Option A : Diffusion-based (ControlNet)

4. **`controlnet_sdxl/train_controlnet_sdxl.py`** — Entraîner ControlNet SDXL
   - **Rôle** : Fine-tuner ControlNet SDXL sur paires synthétiques appairées.
   - **Entrée** : `--train_data_dir` pointant vers dataset préparé (SYNTHIA, GTA, etc.).
   - **Sortie** : Modèle ControlNet fine-tuné `.safetensors`.
   - **Commande** :
     ```bash
     python src/controlnet_sdxl/train_controlnet_sdxl.py \
       --train_data_dir /path/to/synthia_prepared \
       --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
       --max_train_steps 500 \
       --train_batch_size 1
     ```
   - **Voir** : [controlnet_sdxl/OVERVIEW.md](controlnet_sdxl/OVERVIEW.md)

**Résultat attendu** : Modèle ControlNet entraîné, checkpoint sauvegardé.

5. **`controlnet_sdxl/inference.py`** — Générer images réalistes via ControlNet
   - **Rôle** : Utiliser ControlNet fine-tuné pour transformer synthétiques → réalistes.
   - **Entrée** : Dataset synthétique + modèle entraîné.
   - **Sortie** : Images réalistes générées (`--output_dir`).
   - **Commande** :
     ```bash
     python src/controlnet_sdxl/inference.py \
       --train_data_dir /path/to/synthia_prepared \
       --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
       --controlnet_model_name_or_path /path/to/trained_controlnet.safetensors \
       --output_dir /path/to/generated
     ```
   - **Voir** : [controlnet_sdxl/OVERVIEW.md](controlnet_sdxl/OVERVIEW.md)

**Résultat attendu** : Images réalistes haute résolution (512×512).

#### Option B : Fine-tuning ControlNet classique

6. **`controlnet_finetuning/`** — Fine-tuner ControlNet sur paires
   - **Rôle** : Alternative : fine-tune ControlNet v2.1 si pas SDXL.
   - **Entrée** : Paires syn→real appairées + captions.
   - **Commande** :
     ```bash
     python src/controlnet_finetuning/train.py --config_path config.yml
     ```
   - **Voir** : [controlnet_finetuning/OVERVIEW.md](controlnet_finetuning/OVERVIEW.md)

#### Option C : GAN-based (CycleGAN-Turbo)

7. **`gan/train_cyclegan.py`** — Entraîner CycleGAN-Turbo
   - **Rôle** : Alternative GAN : pas paires requises, plus rapide (single-step).
   - **Entrée** : Dossier images synthétiques + dossier images réelles (non appairées).
   - **Commande** :
     ```bash
     python src/gan/train_cyclegan.py \
       --syn_folder /path/to/synthetic \
       --real_folder /path/to/real \
       --output_dir /path/to/output
     ```
   - **Voir** : [gan/OVERVIEW.md](gan/OVERVIEW.md)

8. **`gan/inference.py`** — Générer via CycleGAN-Turbo
   - **Commande** :
     ```bash
     python src/gan/inference.py \
       --input_image /path/to/image.png \
       --model_path /path/to/checkpoint.pth \
       --prompt "a real photo" \
       --direction a2b
     ```
   - **Voir** : [gan/OVERVIEW.md](gan/OVERVIEW.md)

---

### 🟣 Phase 3 : Post-traitement (Optionnel)

**Objectif** : améliorer qualité images générées.

9. **`deepfloyd_scaler/`** — Upscaler images en haute résolution
   - **Rôle** : 512×512 → 1024×1024 via DeepFloyd IF-II + SD x4.
   - **Entrée** : Images générées basse résolution + captions BLIP.
   - **Commande** :
     ```bash
     python src/deepfloyd_scaler/scale.py --config_path config.yml
     ```
   - **Voir** : [deepfloyd_scaler/OVERVIEW.md](deepfloyd_scaler/OVERVIEW.md)

**Résultat attendu** : Images ultra haute résolution (1024×1024).

---

### 🔵 Phase 4 : Évaluation

**Objectif** : mesurer objectivement amélioration domain gap.

10. **`model_eval/train_deeplab.py`** — Entraîner segmenter sur données augmentées
    - **Rôle** : Fine-tune DeepLab v3+ sur images générées.
    - **Entrée** : Images augmentées (ControlNet/GAN output) + labels segmentation.
    - **Commande** :
      ```bash
      python src/model_eval/train_deeplab.py \
        --in_train_dataset /path/to/augmented/images \
        --in_test_dataset /path/to/real/cityscapes \
        --exp_folder /path/to/output
      ```
    - **Voir** : [model_eval/OVERVIEW.md](model_eval/OVERVIEW.md)

11. **Comparer résultats**
    - Comparer F1 baseline (étape 1) vs. F1 augmentées (étape 10).
    - Gain quantifié = F1_augmented - F1_baseline.
    - Exemple : baseline 45% → augmented 52% = **+7% improvement**.

**Résultat attendu** : Métriques F1/mIoU sur real data, comparaison vs baseline.

---

## Dépendances inter-modules

Diagramme de flux :

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 0 : Setup                                                     │
│ baseline_model/main.py ──────────────────┬─────────────────────────│
│   (Benchmark F1 baseline sur real data)  │                         │
└──────────────────────────────────────────┼─────────────────────────┘
                                           │
                                   F1_baseline (ex: 45%)
                                           │
┌──────────────────────────────────────────▼─────────────────────────┐
│ Phase 1 : Data Prep                                                 │
│ controlnet_sdxl/prepare_synthia.py ─────────────────────────────── │
│   (SYNTHIA → Cityscapes format + metadata.jsonl)                   │
│   [optional: blip_caption/main.py ────────────────────────────────]│
│   (Generate BLIP captions if missing)                              │
└──────────────────────────────────────┬───────────────────────────┬─┘
                                       │                           │
                    ┌──────────────────┴──────────────────┐         │
                    │                                     │         │
┌───────────────────▼──────────────────┐  ┌──────────────▼────────┐│
│ Phase 2A : ControlNet (Diffusion)    │  │ Phase 2B : CycleGAN   ││
│ controlnet_sdxl/train.py             │  │ gan/train_cyclegan.py ││
│   (Train ControlNet SDXL)            │  │   (Train CycleGAN)    ││
│   ↓                                  │  │   ↓                   ││
│ controlnet_sdxl/inference.py         │  │ gan/inference.py      ││
│   (Generate images 512×512)          │  │   (Generate images)   ││
└───────────────────┬──────────────────┘  └──────────────┬────────┘│
                    │                                     │
        ┌───────────┴─────────────────────────────────────┘
        │
        │ (Optional: Upscale to 1024×1024)
┌───────▼──────────────────────────────────┐
│ Phase 3 : Post-processing                │
│ deepfloyd_scaler/scale.py                │
│   (512×512 → 1024×1024)                  │
└───────┬──────────────────────────────────┘
        │
        │ (Augmented images)
┌───────▼──────────────────────────────────────────────┐
│ Phase 4 : Evaluation                                 │
│ model_eval/train_deeplab.py                          │
│   (Train segmenter on augmented data)                │
│   ↓                                                  │
│ Compute F1/mIoU on real Cityscapes                   │
│   ↓                                                  │
│ Compare F1_augmented vs F1_baseline                  │
│   ↓                                                  │
│ Report improvement (ex: +7% F1)                      │
└──────────────────────────────────────────────────────┘
```

---

## Détails par module

| Module | Rôle | Entrée | Sortie | Ref |
|--------|------|--------|--------|-----|
| `baseline_model` | Benchmark baseline domain gap | Images syn + real | F1/accuracy | [OVERVIEW](baseline_model/OVERVIEW.md) |
| `blip_caption` | Générer captions textuelles | Images + dataset type | JSON captions | [OVERVIEW](blip_caption/OVERVIEW.md) |
| `controlnet_finetuning` | Fine-tune ControlNet classique | Paires syn↔real appairées | Modèle fine-tuné | [OVERVIEW](controlnet_finetuning/OVERVIEW.md) |
| `controlnet_sdxl` | Entraîner ControlNet SDXL (principal) | Dataset syn + captions | Modèle + images générées | [OVERVIEW](controlnet_sdxl/OVERVIEW.md) |
| `deepfloyd_scaler` | Upscaler haute résolution | Images 512×512 + captions | Images 1024×1024 | [OVERVIEW](deepfloyd_scaler/OVERVIEW.md) |
| `gan` | Entraîner CycleGAN-Turbo | Images syn + real (non appairées) | Modèle + images générées | [OVERVIEW](gan/OVERVIEW.md) |
| `model_eval` | Évaluer domain gap via segmentation | Images augmentées + real | F1/mIoU/métriques | [OVERVIEW](model_eval/OVERVIEW.md) |

---

## Tracking expériences (MLflow)

### 📊 MLflow — Logging automatique d'expériences

Le dossier `mlflow/` fournit un système de logging centralisé pour tracker toutes les expériences.

**Fichier clé** : `mlflow/auto_log_exp_mlflow.py`
- Logs automatiques : paramètres, métriques, modèles.
- Intégration avec modules (baseline_model, model_eval, etc.).
- Sauvegarde des résultats dans une base MLflow locale.

### Utilisation

#### 1. Démarrer MLflow server local

```bash
cd mlflow
mlflow ui --host 127.0.0.1 --port 5000
```

Accéder à l'interface : `http://localhost:5000`

#### 2. Logger une expérience baseline

```bash
python src/baseline_model/main.py \
  --train_path /path/to/syn \
  --val_path /path/to/real
# MLflow log automatiquement : F1, loss, hyperparamètres
```

#### 3. Logger une expérience ControlNet

Adapter le code pour intégrer MLflow logging dans `train_controlnet_sdxl.py` :

```python
import mlflow

mlflow.start_run(run_name="controlnet_sdxl_synthia_v1")
mlflow.log_params({
    "model": "controlnet_sdxl",
    "dataset": "synthia",
    "batch_size": 1,
    "learning_rate": 1e-4,
})
# ... entraînement ...
mlflow.log_metrics({"loss": final_loss, "f1": f1_score})
mlflow.end_run()
```

#### 4. Comparer expériences

Dans MLflow UI :
- Voir tous les runs (baseline, controlnet, gan, etc.).
- Comparer métriques (F1, loss) entre runs.
- Télécharger modèles et artifacts.

### Configuration

Voir `mlflow/pyproject.toml` pour dépendances (mlflow, pyyaml, etc.).

Tests : `mlflow/tests/test_auto_log_exp_mlflow.py`

---

## ⚡ Quick start

### Scenario 1 : Évaluer baseline rapidement

```bash
# 1. Setup baseline benchmark
python src/baseline_model/main.py \
  --train_path /data/gta/train \
  --val_path /data/cityscapes/val

# → Résultat : baseline F1 (ex: 45%)
```

### Scenario 2 : Améliorer avec ControlNet (flux complet)

```bash
# 1. Préparer SYNTHIA
python src/controlnet_sdxl/prepare_synthia.py \
  --synthia_root /data/raw/synthia \
  --output_root /data/synthia_prepared

# 2. Entraîner ControlNet
python src/controlnet_sdxl/train_controlnet_sdxl.py \
  --train_data_dir /data/synthia_prepared \
  --max_train_steps 500

# 3. Générer images améliorées
python src/controlnet_sdxl/inference.py \
  --train_data_dir /data/synthia_prepared \
  --controlnet_model_name_or_path /path/to/trained_controlnet.safetensors

# 4. Évaluer amélioration
python src/model_eval/train_deeplab.py \
  --in_train_dataset /data/generated_images \
  --in_test_dataset /data/cityscapes/val

# → Résultat : F1 augmentée (ex: 52%) → +7% improvement
```

### Scenario 3 : GAN-based approach (CycleGAN-Turbo)

```bash
# 1. Entraîner CycleGAN (données non appairées)
python src/gan/train_cyclegan.py \
  --syn_folder /data/gta/images \
  --real_folder /data/cityscapes/images

# 2. Générer images via CycleGAN
python src/gan/inference.py \
  --input_image /path/to/gta_image.png \
  --model_path /path/to/cyclegan_checkpoint.pth

# 3. Évaluer
python src/model_eval/train_deeplab.py \
  --in_train_dataset /data/gan_generated \
  --in_test_dataset /data/cityscapes/val
```

---

## 📖 Ressources supplémentaires

- **Docs générales** : voir `docs/` (acceleration.md, general_approach.md, SOTA.md, etc.).
- **Tests** : chaque module inclut `tests/` avec tests unitaires.
- **Configuration Docker** : Dockerfiles individuels dans chaque module.
- **Dependencies** : `pyproject.toml` dans src/ et modules.

---

## 🤝 Contribution & Support

Pour adapter, étendre ou déboguer :
- Consulter les OVERVIEW.md correspondants.
- Vérifier les tests existants.
- Utiliser MLflow pour tracker les expériences.
- Documenter les modifications pour reproductibilité.