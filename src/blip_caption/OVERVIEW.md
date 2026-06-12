# blip_caption — Overview

TL;DR — Ce module génère automatiquement des captions textuelles pour des images de datasets (Cityscapes, GTA, SYNTHIA, etc.) en utilisant le modèle BLIP de Salesforce.

## Rôle et contexte

Le module `blip_caption` prend une liste d'images et produit des captions textuelles enrichies, essentielles pour entraîner des modèles de diffusion conditionnels (ControlNet, SDXL, etc.). Les captions générées sont stockées avec les métadonnées d'image dans un fichier JSON structuré.

## Structure & fichiers clés

- `main.py` : orchestrateur principal — charge le modèle, traite les images, sauvegarde les résultats.
- `utils.py` : fonctions utilitaires:
  - `load_model(device)` : charge le modèle BLIP et le processor depuis HuggingFace.
  - `load_data(data_path)` : lit la liste des images depuis un fichier JSON.
  - `process(processor, model, paths, prompt)` : génère les captions par batch (20 images/batch).
  - `save_captions(paths, captions, filename, prompt)` : enrichit le JSON avec les captions générées.
- `config.py` : paramètres (DEVICE, DATA_PATH, FILE_NAME, PROMPT).
- `Dockerfile` : image pour exécution en conteneur.
- `pyproject.toml` : dépendances (transformers, torch, Pillow, requests).
- `tests/` : tests unitaires pour `utils.py`.

## Format d'entrée/sortie

### Entrée (DATA_PATH)

Fichier JSON contenant une liste d'objets avec au minimum `image` (chemin absolu) et `dataset` (ex: 'real', 'synthetic'):

```json
[
  {
    "image": "/path/to/image1.jpg",
    "dataset": "real"
  },
  {
    "image": "/path/to/image2.jpg",
    "dataset": "synthetic"
  }
]
```

### Sortie (FILE_NAME)

Même structure enrichie avec le champ `text` contenant la caption générée:

```json
[
  {
    "image": "/path/to/image1.jpg",
    "dataset": "real",
    "text": "a real picture of a street scene with buildings and cars"
  },
  {
    "image": "/path/to/image2.jpg",
    "dataset": "synthetic",
    "text": "a synthetic picture of a city street with parked vehicles"
  }
]
```

## Workflow & utilisation

1. **Préparer le fichier d'entrée** : créer un JSON avec chemins absolus et type de dataset.
   ```bash
   # Exemple : créer data.json avec liste d'images
   ```

2. **Configurer les paramètres** dans `config.py`:
   - `DATA_PATH` : chemin du JSON d'entrée.
   - `FILE_NAME` : chemin du JSON de sortie.
   - `DEVICE` : auto-détecté (`cuda` si GPU disponible, sinon `cpu`).
   - `PROMPT` : prompt initial pour BLIP (par défaut `"a picture of "`).

3. **Exécuter la génération de captions** :
   ```bash
   cd src/blip_caption
   python main.py
   ```
   ou en conteneur:
   ```bash
   docker build -t blip_caption .
   docker run --rm -v /path/to/data:/data blip_caption
   ```

4. **Résultat** : le JSON enrichi est sauvegardé dans `FILE_NAME`.

## Notes d'optimisation & bonnes pratiques

- **Batch size** : actuellement fixé à 20 images par batch (dans `process()`). Ajuste selon la VRAM disponible.
- **Prompt personnalisé** : le champ `PROMPT` est préfixé; la caption complète devient `{prompt} + {caption_generée}`.
- **Dataset type** : le champ `dataset` (real/synthetic) est intégré dans la caption finale pour distinguer les sources. Ex: `"a synthetic picture of ..."` vs `"a real picture of ..."`.
- **GPU/CPU** : détection automatique. Sur GPU faible, baisse le batch size ou utilise `cpu`.
- **Téléchargement du modèle** : première exécution télécharge `Salesforce/blip-image-captioning-base` (~1 GB) dans le cache HuggingFace. Prévoir espace disque.

## Intégration avec controlnet_sdxl

Le module `blip_caption` est souvent utilisé avant `prepare_synthia.py` ou dans des pipelines manuels. Pour intégration automatique dans les workflows `controlnet_sdxl`:
- La script `prepare_synthia.py` inclut déjà la génération de captions BLIP (exécute `process()` et `save_captions()` en interne).
- Si tu dois adapter `blip_caption` pour une interface CLI plus générique, on peut paramètrer `DATA_PATH` / `FILE_NAME` via arguments CLI au lieu de `config.py`.

## Améliorations possibles

- [ ] Ajouter CLI (`argparse`) pour passer `--data-path`, `--output-file`, `--device`, `--prompt`, `--batch-size`.
- [ ] Support des formats d'entrée variables (CSV, directorios d'images sans JSON).
- [ ] Paramètrage du prompt et du modèle BLIP (ex: `blip-image-captioning-large` pour captions plus détaillées).
- [ ] Logging structuré avec timestamps pour suivre les progressions longues.
- [ ] Sauvegarde partielle des résultats en cas d'interruption (checkpoint tous les N batches).

---

Pour toute question ou si tu veux que j'implémente une amélioration, fais-moi signe !
