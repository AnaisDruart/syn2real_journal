# deepfloyd_scaler — Overview

TL;DR — Module d'upscaling d'images en haute résolution (64×64 → 1024×1024) en utilisant un pipeline à deux étapes (DeepFloyd IF-II + Stable Diffusion x4 Upscaler) guidé par captions textuelles.

## Rôle et contexte

Le module `deepfloyd_scaler` augmente la résolution des images de manière intelligente :
- **Étape 1** : DeepFloyd IF-II-L (64×64 → 256×256) en utilisant embeddings de prompts textuels.
- **Étape 2** : Stable Diffusion x4 Upscaler (256×256 → 1024×1024) pour détails finaux.

Cas d'usage dans syn2real :
- Upscaler les images générées par ControlNet (avant 512×512) vers 1024×1024.
- Améliorer détails et qualité en s'appuyant sur les captions textuelles.
- Générer un dataset de haute résolution pour entraînement ultérieur.

## Structure & fichiers clés

- **`scale.py`** : implémentation principale du pipeline.
  - `flush()` : nettoie la mémoire GPU/CPU après utilisation de pipelines.
  - `encode_prompts(prompts)` : extrait embeddings textuels via T5EncoderModel (DeepFloyd).
  - `transform_img(img_path, size)` : charge, recadre (carré), redimensionne et normalise image.
  - `process(images_list, prompts_list, size, output_dir)` : orchestrateur principal.
    - Charge pipelines DeepFloyd IF-II et SD x4 Upscaler.
    - Itère sur les images, applique upscaling bi-stage.
    - Sauvegarde résultats.

- **`config.yml`** : configuration centralisée.
  - `data_path` : chemin JSON avec captions (format BLIP).
  - `start_idx`, `end_idx` : plage d'images à traiter.
  - `images_path` : dossier contenant images basse résolution à upscaler.
  - `output_path` : dossier de sortie pour images upscalées.

- **`Dockerfile`** : conteneurisation avec support GPU.
- **`pyproject.toml`** : dépendances (diffusers, transformers, torch, torchvision, pillow).
- **`tests/`** : tests unitaires pour `scale.py`.

## Workflow complet

```
1. Préparer données
   ├─ Images basse résolution (ex: 512×512 générées par ControlNet)
   ├─ JSON avec captions BLIP {"caption": "a real picture of..."}
   └─ Spécifier plage : start_idx, end_idx
   
2. Configurer config.yml
   ├─ data_path → JSON captions
   ├─ images_path → dossier images sources
   ├─ output_path → dossier résultats
   └─ start_idx/end_idx → plage à traiter
   
3. Lancer upscaling
   └─ python scale.py --config_path config.yml
   
4. Pipeline d'upscaling (interne)
   ├─ Charger T5EncoderModel pour extraire embeddings prompts
   ├─ Charger DeepFloyd IF-II-L pipeline (fp16, device_map="balanced")
   ├─ Charger Stable Diffusion x4 Upscaler pipeline (fp16, device_map="balanced")
   ├─ Pour chaque image :
   │  ├─ Charger + normaliser image [-1, 1]
   │  ├─ Stage 1 : DeepFloyd 64×64 → 256×256 (guided by prompt embeddings)
   │  ├─ Stage 2 : SD x4 → 1024×1024 (guided by original prompt)
   │  └─ Sauvegarder résultat
   └─ Nettoyer GPU memory
   
5. Résultat
   └─ Images upscalées (1024×1024) dans output_path
```

## Format d'entrée/sortie

### Entrée 1 : Images basse résolution (images_path)

Dossier contenant images PNG/JPG, ex:
```
/out/validation_out_canny_lowres/
├── image_40.png      (64×64 ou variable)
├── image_41.png
└── ...
```

Format : RGB ou RGBA, n'importe quelle résolution (sera recadrée en carré et redimensionnée à `size` param).

### Entrée 2 : Captions textuelles (data_path)

Fichier JSON avec captions BLIP générées (voir `blip_caption` module) :

```json
[
  {
    "image": "/path/to/image40.jpg",
    "caption": "a real picture of a street scene with buildings and cars"
  },
  {
    "image": "/path/to/image41.jpg",
    "caption": "a real picture of an urban area with parked vehicles"
  }
]
```

La plage `[start_idx : end_idx]` est appliquée pour traiter un sous-ensemble.

### Sortie : Images upscalées haute résolution (output_path)

Dossier contenant images 1024×1024 :

```
/out/upscaling/
├── image_0.png   (1024×1024)
├── image_1.png
└── ...
```

Format : PNG/PIL Image.

## Utilisation

### 1. Configurer config.yml

```yaml
# Chemin JSON avec captions BLIP
data_path: "/data/cityscape/data_val_captionned.json"

# Plage d'images à traiter (start_idx inclus, end_idx exclusif)
start_idx: 40
end_idx: 100

# Dossier images source (basse résolution)
images_path: "/out/validation_out_canny_lowres/"

# Dossier résultats upscalés
output_path: "/out/upscaling/"
```

### 2. Lancer l'upscaling

```bash
cd src/deepfloyd_scaler

# Avec config.yml par défaut (dans le dossier courant)
python scale.py

# Avec config path spécifique
python scale.py --config_path /path/to/config.yml
```

Ou en Docker :
```bash
docker build -t deepfloyd_scaler .
docker run --rm --gpus all -v /data:/data -v /out:/out deepfloyd_scaler
```

### 3. Résultats

Vérifier les images dans `output_path` :
```bash
ls -lh /out/upscaling/
# image_0.png  (1024×1024, ~5-10 MB)
# image_1.png
```

## Architecture du pipeline

### Étape 1 : DeepFloyd IF-II-L (64×64 → 256×256)

- **Modèle** : `DeepFloyd/IF-II-L-v1.0` (Stability AI).
- **Entrées** :
  - `image` : tenseur basse résolution [-1, 1].
  - `prompt_embeds` : embeddings T5 du prompt texte.
  - `negative_prompt_embeds` : embeddings pour prompt négatif (ex: "blurry, low quality").
- **Sortie** : image 256×256 en fp16.
- **Optimisations** :
  - `text_encoder=None` : économise mémoire (embeddings pré-extraits).
  - `device_map="balanced"` : distribue model across GPUs.
  - `variant="fp16"` : utilise float16 pour mémoire/vitesse.

### Étape 2 : Stable Diffusion x4 Upscaler (256×256 → 1024×1024)

- **Modèle** : `stabilityai/stable-diffusion-x4-upscaler`.
- **Entrées** :
  - `image` : tenseur 256×256 ([0, 1] normalisé).
  - `prompt` : texte (guide final).
- **Sortie** : image 1024×1024 PIL Image (RGB).
- **Optimisations** : fp16, device_map="balanced".

### Gestion mémoire

- **Pre-encode prompts** : extrait embeddings une seule fois au début.
- **Delete pipelines** : supprime chaque pipeline après utilisation.
- **flush()** : appelle `gc.collect()` + `torch.cuda.empty_cache()`.
- **Generator seed** : fixe aléatoire pour reproductibilité.

## Détails techniques

### Image preprocessing (transform_img)

1. Charger image PIL.
2. **Recadrage carré** : si aspect ratio ≠ 1:1, recadre au carré central.
3. **Redimensionnement** : resize à `size` (par défaut 100 pixels).
4. **Normalisation** : pixels [0, 255] → [-1, 1] pour débruitage.

```python
img_tensor = (img_tensor / 255) * 2 - 1  # [-1, 1]
```

### Encoding prompts

```python
text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/IF-I-XL-v1.0", ...)
pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", ...)
prompt_embeds, negative_embeds = pipe.encode_prompt(text)
```

Produit embeddings de dimension `(1, seq_len, 768)` (pour T5-XL).

### Paramètres clés dans `process()`

| Paramètre | Valeur par défaut | Notes |
|-----------|-------------------|-------|
| `size` | 100 | Taille redimensionnement avant stage 1. |
| `output_dir` | `/out/upscaling/` | Dossier résultats. |
| `device_map` | `"balanced"` | Distribution auto sur GPUs disponibles. |
| `torch_dtype` | `torch.float16` | Mixed precision (fp16). |
| `generator seed` | 2 | Pour reproductibilité. |

## Intégration avec syn2real

Le module s'inscrit dans le pipeline global :

1. **Génération basse résolution** : ControlNet SDXL ou ControlNet fine-tuned → images 512×512.
2. **Upscaling haute résolution** : `deepfloyd_scaler` → images 1024×1024.
3. **Évaluation** : baseline_model sur images upscalées → mesurer amélioration.

Exemple flux complet :
```
prepare_synthia.py (convert + captions)
        ↓
controlnet_sdxl/train_controlnet_sdxl.py (entraîner sur SYNTHIA)
        ↓
controlnet_sdxl/inference.py (générer images basse résolution)
        ↓
deepfloyd_scaler/scale.py (upscaler en haute résolution)
        ↓
baseline_model/main.py (évaluer impact sur real data)
```

## Limitations & Notes

- **Temps d'exécution** : ~30-60 sec/image (selon GPU). Optimiser via batch processing si besoin.
- **VRAM requis** : ~20-30 GB pour fp16 + device_map="balanced". Pour moins de VRAM, tester `device_map="cpu"`.
- **Recadrage carré** : aspect ratio non-carré → perte d'informations. À adapter si images non-carrées importantes.
- **Résolution fixe** : sortie toujours 1024×1024. Pour autres résolutions, modifier pipeline.

## Améliorations / TODOs

- [ ] Support batch processing pour accélérer (traiter N images en parallèle).
- [ ] CLI pour tous les paramètres (size, device_map, output_type).
- [ ] Support upscaling variable (256×256, 512×512, etc. en sortie).
- [ ] Logging structuré avec timestamps.
- [ ] Checkpoints intermédiaires en cas d'interruption.
- [ ] Utilisation d'autres upscalers (ex: Real-ESRGAN, SwinIR).
- [ ] Support video upscaling (frames par frames).
- [ ] Évaluation qualité intégrée (FID, LPIPS, etc.).

---

Le module est conçu pour maximaliser qualité visuelle tout en gérant efficacement la mémoire GPU. Toute optimisation ou variante peut être implémentée sur demande.
