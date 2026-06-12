# gan — Overview

TL;DR — Module implémentant CycleGAN-Turbo (basé sur Stable Diffusion Turbo) pour transformer images synthétiques en réalistes sans paires appairées. Entraîne sur données syn/real non appairées et génère images réalistes haute qualité.

## Rôle et contexte

Le module `gan` utilise **CycleGAN-Turbo**, une architecture efficace et rapide basée sur Stable Diffusion Turbo pour :
- **Entraîner** sans paires appairées (syn ↔ real images ne doivent pas correspondre).
- **Générer** images réalistes à partir d'images synthétiques (a2b) ou inverser (b2a).
- **Fournir** une alternative aux modèles de diffusion classiques : plus rapide, moins VRAM, un seul pas de débruitage.

Contraste avec controlnet_sdxl :
- **CycleGAN-Turbo** : entraîne sans paires appairées, non conditionné (sauf optionnellement par texte).
- **ControlNet** : entraîne sur paires appairées, conditionné par cartes segmentation/Canny.

## Structure & fichiers clés

### Fichiers d'entraînement & modèle

- **`train_cyclegan.py`** : orchestrateur d'entraînement.
  - Charge datasets synthétique et réel (non appairés).
  - Initialise architecture CycleGAN-Turbo (VAE + UNet + Text Encoder).
  - Entraîne avec Accelerate multi-GPU.
  - Évalue avec LPIPS, FID, DinoStructureLoss, Vision-Aided Loss.
  - Utilise LoRA fine-tuning pour efficacité paramétrique.

- **`cyclegan_turbo.py`** : architecture principale.
  - `VAE_encode` / `VAE_decode` : codage/décodage VAE avec skip connections.
  - `initialize_unet()` : initialise UNet 2D avec LoRA.
  - `initialize_vae()` : initialise VAE (Stable Diffusion).
  - `CycleGAN_Turbo` : classe modèle complète.
    - Forward pass : syn → real (a2b) ou real → syn (b2a).
    - Cycle consistency : A → B → A.

- **`model.py`** : utilitaires.
  - `make_1step_sched()` : crée scheduler DDPM à un seul pas.
  - `my_vae_encoder_fwd()` / `my_vae_decoder_fwd()` : forwards VAE customisés.
  - `download_url()` : télécharge checkpoints pré-entraînés.

- **`inference.py`** : génération d'images.
  - CLI pour transformer image synthétique → réaliste.
  - Supporte checkpoints pré-entraînés ou checkpoints locaux.
  - Options : direction (a2b/b2a), image prep, fp16.

### Fichiers utilitaires & données

- **`training_utils.py`** : gestion données et arguments.
  - `UnpairedDataset` : charge images syn et real sans appairage.
  - `build_transform()` : augmentations (resize, crop, flip, color jitter).
  - `parse_args_unpaired_training()` : arguments CLI.

- **`dino_struct.py`** : loss structurelle DinoStructureLoss.
  - Préserve structure contenu via features DINO.
  
- **`Dockerfile`** : conteneurisation.
- **`pyproject.toml`** : dépendances (torch, diffusers, transformers, accelerate, peft, cleanfid, lpips, etc.).
- **`tests/`** : tests unitaires.

## Workflow complet

```
1. Préparer données
   ├─ Images synthétiques (GTA, SYNTHIA, etc.) → dossier A
   ├─ Images réelles (Cityscapes) → dossier B
   └─ Note : pas besoin d'appairage, GAN apprend la distribution

2. Configurer entraînement
   ├─ Datasets paths
   ├─ Learning rate, batch size, num epochs
   ├─ Hyperparamètres perte (λ_cycle, λ_ident, λ_recon, etc.)
   └─ Output dir, logging (wandb/tensorboard)

3. Lancer entraînement
   └─ python train_cyclegan.py [args]
   
4. Entraînement (interne)
   ├─ Charger VAE + UNet + Text Encoder (Stable Diffusion Turbo)
   ├─ Ajouter LoRA adapters pour fine-tuning
   ├─ Multi-GPU via Accelerate
   ├─ Pour chaque batch :
   │  ├─ Encoder images A et B → latents VAE
   │  ├─ Forward UNet : A → B' (generator a2b)
   │  ├─ Forward UNet : B → A' (generator b2a)
   │  ├─ Décoder latents → images pixels
   │  ├─ Cycle loss : ||A - cycle_A|| + ||B - cycle_B||
   │  ├─ Reconstruction loss (L1, perceptual, LPIPS)
   │  ├─ Structural loss (DinoStructureLoss)
   │  ├─ Identity loss (optionnel)
   │  └─ Update avec Accelerate
   ├─ Évaluer FID, LPIPS tous les N steps
   └─ Sauvegarder checkpoints
   
5. Inférence
   ├─ Charger modèle fine-tuné
   ├─ Transformer image synthétique en réaliste
   └─ Exporter résultat
```

## Format d'entrée/sortie

### Entrée 1 : Images synthétiques (domaine A)

Dossier contenant images PNG/JPG :
```
/data/gta/ ou /data/synthia/
├── image_0.png
├── image_1.png
└── ...
```

Format : RGB ou RGBA, n'importe quelle résolution (será redimensionnée selon config).

### Entrée 2 : Images réelles (domaine B)

Dossier contenant images réelles :
```
/data/cityscapes/
├── image_0.jpg
├── image_1.jpg
└── ...
```

**Important** : pas besoin d'appairage ! CycleGAN apprend la transformation de distribution.

### Sortie : Images transformées

```
/output/
├── a2b/
│   ├── image_0.png  (synthétique → réaliste)
│   └── ...
└── b2a/
    ├── image_0.png  (réaliste → synthétique)
    └── ...
```

## Utilisation

### 1. Entraîner CycleGAN-Turbo

```bash
cd src/gan

python train_cyclegan.py \
  --syn_folder /path/to/synthetic/images \
  --real_folder /path/to/real/images \
  --output_dir /path/to/output \
  --epochs 100 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2
```

Ou via configuration fichier (si disponible).

### 2. Faire inférence

```bash
cd src/gan

# Avec modèle pré-entraîné nommé
python inference.py \
  --input_image /path/to/synthetic.png \
  --model_name cyclegan_turbo_gta2cityscapes_v1 \
  --output_dir /path/to/output

# Avec checkpoint local personnalisé
python inference.py \
  --input_image /path/to/synthetic.png \
  --model_path /path/to/checkpoint.pth \
  --prompt "a real photo of a street scene" \
  --direction a2b \
  --output_dir /path/to/output \
  --use_fp16  # Pour plus rapide/moins mémoire
```

### 3. En Docker

```bash
docker build -t gan .
docker run --rm --gpus all \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  gan python train_cyclegan.py --syn_folder /data/syn --real_folder /data/real --output_dir /output
```

## Architecture détails

### CycleGAN-Turbo

Basée sur Stable Diffusion Turbo (single-step diffusion) :

1. **VAE Encoder** : image → latent representation.
   - Downsample spatial, latent dimension 8 (SD standard).
   - Custom forward avec skip connections.

2. **UNet 2D** : transformation latent space.
   - Conditionné par CLIP text embeddings.
   - LoRA fine-tuning sur couches encoder/decoder/autres.
   - Un seul pas de débruitage (vs 20-50 pour diffusion classique).

3. **VAE Decoder** : latent → image.
   - Utilise skip connections de l'encoder.
   - Upsampling avec connexions résiduelles.

### Pertes d'entraînement

| Loss | Rôle | Poids |
|------|------|-------|
| **Cycle Consistency** | A → B' → A ≈ A | λ_cycle |
| **Reconstruction (L1/Perceptual)** | A → B' doit être réaliste | λ_recon |
| **LPIPS** | Similitude perceptuelle img réelle vs générée | λ_lpips |
| **DinoStructureLoss** | Préserve structure DINO | λ_dino |
| **Vision-Aided Loss** | Discriminateur basé features CNN | λ_vai |
| **Identity (optionnel)** | B → B' ≈ B (réel non modifié) | λ_ident |

### LoRA Fine-tuning

Au lieu de fine-tuner tous les poids UNet (~800M params), LoRA ajoute matrices adapters petites :
- Réduit mémoire + compute.
- Entraînable rapide en 100-200 epochs.
- Qualité comparable à full fine-tuning.

## Intégration avec syn2real

CycleGAN-Turbo s'intègre comme **alternative à ControlNet** :

```
Approche ControlNet (segmentation-conditionné) :
prepare_synthia.py → controlnet_sdxl/train → inference → baseline_model

Approche CycleGAN (non-conditionné) :
GTA/SYNTHIA images → gan/train_cyclegan.py → inference → baseline_model
```

Avantages CycleGAN-Turbo :
- **Pas paires appairées** : utilise des datasets disjoints.
- **Rapide** : un seul pas de débruitage (10-100× plus rapide).
- **Moins VRAM** : VAE latent space moins gros.
- **Flexible** : optionnel text-guided.

Limitations :
- **Pas de contrôle fin** : pas de conditional input (segmentation, Canny).
- **Mode unilatéral** : apprend une direction (a2b ou b2a).

## Améliorations / TODOs

- [ ] Configuration file support (YAML/JSON).
- [ ] Validation set séparé avec FID/LPIPS checkpoints.
- [ ] Early stopping sur FID plateau.
- [ ] Support multi-direction (both a2b et b2a simultanément).
- [ ] Logging structuré (MLflow, TensorBoard).
- [ ] Batch inference pour speedup.
- [ ] Export optimisé (ONNX, TorchScript).
- [ ] Techniques avancées (style transfer, attribute editing).

---

Le module fournit une approche GAN moderne et efficace pour domain adaptation. Adaptations et optimisations possibles sur demande.
