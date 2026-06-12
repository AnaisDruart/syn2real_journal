# controlnet_finetuning — Overview

TL;DR — Module pour fine-tuner un ControlNet (Stable Diffusion v2.1/v1.5) sur données synthétiques (Cityscapes, GTA, SYNTHIA) afin de générer des images réalistes à partir de contrôles (cartes de segmentation + Canny edges).

## Rôle et contexte

Le module `controlnet_finetuning` implémente un pipeline complet pour :
- **Fine-tuner** un ControlNet pré-entraîné sur une base de données synthétiques + réelles mixtes.
- **Apprendre** le mapping synthétique→réaliste via paires d'images + captions textuelles.
- **Générer** des images réalistes en inférence à partir de :
  - **Contrôles** : segmentation maps (optionnel) + Canny edges.
  - **Prompts textuels** : descriptions du contenu attendu (ex: `"a real picture of a street with cars"`).
- Fournir un **modèle amélioré** par rapport au ControlNet générique pour la tâche syn→real.

## Architecture & fichiers clés

### Fichiers principaux

- **`train.py`** : orchestrateur d'entraînement.
  - Fonction `train()` : charge dataset, modèle, lance PyTorch Lightning trainer (multi-GPU).
  - Fonction `main()` : parse config YAML, prépare expérience MLflow, exécute entraînement.
  - Configuration : `resume_path`, `model_path`, `learning_rate`, `accumulate_grad_batches`, `max_epochs`, etc.

- **`evaluate.py`** : inférence et génération d'images réalistes.
  - Fonction `eval()` : transforme images synthétiques en réalistes via ControlNet.
  - Utilise DDIM sampler pour débruitage itératif.
  - Prend en entrée : image synthétique, contrôle (edges/seg), prompt texte.
  - Sortie : image réaliste générée.

- **`dataset.py`** : deux classes Dataset PyTorch.
  - `MyDataset` : format simple JSON avec `source`, `target`, `prompt`.
  - `CityDataset` : format enrichi pour Cityscapes/GTA/SYNTHIA, avec augmentations :
    - Suppression aléatoire des segmentation maps (apprentissage des edges seules).
    - Canny edges appliquées dynamiquement (seuils différents pour syn vs real).
    - Noise ajouté optionnellement.

- **`config.yml`** : configuration centrale en YAML.
  - `exp_config` : nom expérience, run, tracking MLflow.
  - `model_config` : chemin modèle, chemin checkpoint, locked SD, mid-control only.
  - `data_config` : chemin données JSON, résolution images, canny, noise.
  - `train_config` : batch size, learning rate, accumulation, epochs, GPUs, precision.

- **`ControlNet/` (submodule)** : implémentation ControlNet (source externe, ex: lllyasviel/ControlNet).
  - Contient modèle `cldm_v21.yaml`, utilitaires annotateurs (Canny).

- **`Dockerfile`** : conteneurisation pour exécution reproductible.
- **`pyproject.toml`** : dépendances (pytorch-lightning, torch, diffusers, transformers, gradio, etc.).
- **`tests/`** : tests unitaires pour dataset.

## Workflow complet

```
1. Préparer dataset
   ├─ Générer paires (synthétique → réaliste)
   ├─ Créer JSON avec {"source": "...", "target": "...", "caption": "..."}
   └─ Placer JSON + images à DATA_PATH
   
2. Configurer entraînement (config.yml)
   ├─ Spécifier checkpoint resume_path (ControlNet pré-entraîné)
   ├─ Paramètres LR, batch size, epochs, GPUs
   └─ Activer Canny, noise, ou autres augmentations
   
3. Lancer entraînement
   └─ python train.py --config_path config.yml --to_log True
   
4. PyTorch Lightning trainer
   ├─ Charge modèle ControlNet + Stable Diffusion
   ├─ Entraîne multi-GPU avec accumulation gradient
   ├─ Log images d'inférence test pendant entraînement
   └─ Sauvegarde checkpoint final
   
5. Évaluation / Inférence
   ├─ Charger modèle fine-tuné
   ├─ Générer images réalistes à partir synthétiques
   └─ Evaluer qualité via baseline_model ou métriques (LPIPS, FID, etc.)
```

## Format d'entrée/sortie

### Entrée (DATA_PATH)

Fichier JSON contenant liste de paires d'images :

```json
[
  {
    "source": "/path/to/synthia/trainIds.png",
    "target": "/path/to/real_cityscape/RGB.jpg",
    "caption": "a real picture of a street scene with buildings"
  },
  {
    "source": "/path/to/gta/segmentation.png",
    "target": "/path/to/real_cityscape/RGB.jpg",
    "caption": "a real picture of urban area with cars"
  }
]
```

**Chemins** : peuvent être absolus ou relatifs à `DATA_PATH`.

**Caption** : générée par BLIP (voir `blip_caption` module) ou manuel. Format : `"a real picture of ...` (distingue syn vs real).

### Checkpoint initial (resume_path)

Pré-trained ControlNet checkpoint (`.ckpt`), ex:
- `/models/controlnet_v21_canny_ep23.ckpt` (pré-entraîné sur Canny).
- Ou checkpoints HuggingFace : `lllyasviel/control_v21_canny`.

### Sortie (weights_file)

Modèle fine-tuné sauvegardé en `.ckpt` :
```
/models/{exp_name}_{run_name}/trained_model_weights.ckpt
```

Logs d'entraînement + images de validation :
```
/out/{exp_name}_{run_name}/config.yml
/out/{exp_name}_{run_name}/lightning_logs/  (images + metrics)
```

## Utilisation

### 1. Préparer la configuration (config.yml)

```yaml
exp_config: 
  exp_name: my_experiment
  run_name: finetuned_v1

model_config: 
  resume_path: /models/controlnet_v21_canny.ckpt  # ou HF ID
  model_path: /models/cldm_v21.yaml
  sd_locked: True       # Freeze SD layers si True
  only_mid_control: False

data_config: 
  data_path: /path/to/metadata.json  # Liste paires syn→real
  img_w: 896
  img_h: 448
  canny: True           # Appliquer Canny edges
  noise: False          # Ajouter bruit aux contrôles

train_config: 
  batch_size: 1
  logger_freq: 100      # Log images tous les N batches
  learning_rate: 2e-5
  accumulate_grad_batches: 15  # Accumule gradient avant update
  num_devices: 2        # Nombre GPUs
  max_epochs: 50
  precision: 32         # 32-bit float ou 16 (mixed precision)
```

### 2. Lancer l'entraînement

```bash
cd src/controlnet_finetuning

# Avec config.yml dans le dossier courant
python train.py

# Avec config path spécifique
python train.py --config_path /path/to/config.yml --to_log True

# Avec MLflow logging activé
python train.py --config_path config.yml --to_log True
```

Ou en Docker :
```bash
docker build -t controlnet_finetuning .
docker run --rm --gpus all -v /data:/data -v /models:/models -v /out:/out controlnet_finetuning
```

### 3. Évaluation / Inférence

Utiliser `evaluate.py` pour générer images réalistes :

```python
from controlnet_finetuning.evaluate import eval
from cldm.model import create_model, load_state_dict

model = create_model(model_path).cpu()
model.load_state_dict(load_state_dict(checkpoint_path, location="cpu"))

# Générer image réaliste
generated_image = eval(
    image=synthetic_image_rgb,
    control=canny_edge_map,
    prompt="a real picture of a street",
    model=model,
    ddim_sampler=sampler,
    ddim_steps=50,
    strength=0.7
)
```

## Techniques d'entraînement

### Augmentations dans CityDataset

1. **Segmentation Removal** : supprime cartes de segmentation pour 50% images synthétiques.
   - Force le modèle à apprendre uniquement des Canny edges.
   
2. **Canny Edges** :
   - Appliqué à 80% des images (20% gardes segmentation seule).
   - Seuils différents pour syn (`low=100, high=200`) vs real (`low=50, high=120`).
   
3. **Noise** : optionnel, ajoute bruit au contrôle pour robustesse.

### Stratégies d'entraînement

- **Gradient Accumulation** : accumule gradients de N batches avant update → simule batch size larger.
- **Multi-GPU** : PyTorch Lightning distribue batch sur GPUs (data parallel).
- **Mixed Precision** (`precision=16`) : utilise float16 + float32 sélectif pour speed + mémoire.
- **SD Locked** : si `sd_locked=True`, gèle les layers de Stable Diffusion → entraîne seulement ControlNet adapter.

## Intégration avec le pipeline syn2real

Le module s'intègre dans le workflow global :

1. **Préparation** : `prepare_synthia.py` / `blip_caption.py` → dataset JSON + captions.
2. **Fine-tuning** : `controlnet_finetuning/train.py` → modèle amélioré.
3. **Augmentation** : `controlnet_sdxl/train_controlnet_sdxl.py` → générer plus données synthétiques améliorées.
4. **Évaluation** : `baseline_model/main.py` → mesurer amélioration accuracy sur real data.

## Améliorations / TODOs

- [ ] Support pour d'autres architectures ControlNet (ControlNet XL, etc.).
- [ ] Validation set séparé avec métriques (LPIPS, FID) logging.
- [ ] Early stopping basé sur validation loss.
- [ ] Support des données source variables (Canny only, segmentation only, tous).
- [ ] CLI pour tous les hyperparamètres (au lieu de config.yml seul).
- [ ] Optimisation vitesse (batch size augmentation, compilation).
- [ ] Export du modèle pour formats portables (ONNX, TorchScript).

---

Le module est conçu pour être robuste et flexible. Des améliorations peuvent être implémentées sur demande.
