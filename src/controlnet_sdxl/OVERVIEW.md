# controlnet_sdxl — Overview

TL;DR — Ce dossier contient les outils pour préparer les datasets (Cityscapes/GTA/SYNTHIA), entraîner ControlNet (et variantes Unet), faire de l’inférence et exécuter un pipeline d’active learning.

## Structure & rôles (fichiers clés)

 - `train_controlnet_sdxl.py` : script principal d'entraînement ControlNet.
 - `train_sdxl_unlocked.py` : variante pour entraînement UNet.
 - `dataset.py` : loader & transformations (utilise `metadata.jsonl` / colonnes `image`, `conditioning_image`, `text`, `syn_or_real`).
 - `log.py` : argparse, validation des arguments, configuration de logging/tracking.
 - `model.py` : utilities pour encoder prompts et sauvegarder modèles/card.
 - `inference.py` : génération d'images à partir de conditioning images.
 - `active_train.py` : orchestration multi‑round d'entraînement/évaluation/selection.
 - `prepare_synthia.py` : conversion SYNTHIA → format Cityscapes + génération de captions BLIP + `metadata.jsonl`.
 - `prepare_new_dataset.py` : pipeline pour générer/augmenter nouvelles images via SDXL+ControlNet (prévoir refactor si utilisé).
 - `config.yml` : configuration multi‑round pour `active_train.py`.
 - `pyproject.toml` / `poetry.lock` / `requirements.txt` : dépendances.

## Flux de données (exemple SYNTHIA)

1. Télécharger et dézipper SYNTHIA (ex : `C:\\data\\raw\\synthia`).
2. Lancer la conversion :
	 - `python src\\controlnet_sdxl\\prepare_synthia.py --synthia_root 'C:\\data\\raw\\synthia' --output_root 'C:\\data\\synthia_prepared' --device cpu`
	 - Résultat : `C:\\data\\synthia_prepared\\leftImg8bit\\train\\` (images), `gtFine\\train\\` (`*_trainIds.png`, `*_color.png`) et `metadata.jsonl`.
3. Le loader (`dataset.py`) lit `--train_data_dir <output_root>` ou `--dataset` si support ajouté.
4. Entraînement : `python src\\controlnet_sdxl\\train_controlnet_sdxl.py --train_data_dir 'C:\\data\\synthia_prepared' ...`

## Format attendu des annotations / metadata

 - Pour chaque image `xxxx.png` :
	 - `gtFine/train/xxxx_trainIds.png` → uint8 trainIds (0..18) et 255 ignore.
	 - `gtFine/train/xxxx_color.png` → visualisation couleur.
 - `metadata.jsonl` : lignes JSON avec clés `image`, `conditioning_image`, `text`, `syn_or_real` (1 = synthétique).

## Bonnes pratiques & notes

 - Ne pas committer les données (ajouter `data/`, `C:\\data\\*`, `src/controlnet_sdxl/.venv/` au `.gitignore`).
 - BLIP et modèles lourds se téléchargent dans le cache HF — prévoir espace disque.
 - Pour GPU, installe PyTorch compatible CUDA; pour machines avec peu de VRAM, préférer `--device cpu`.
 - Vérifications rapides :
	 - `np.unique(Image.open(..._trainIds.png))` doit renvoyer valeurs 0..18 ± 255.
	 - `head -n 5 metadata.jsonl` pour vérifier les captions/chemins.
 - `prepare_new_dataset.py` contient des chemins et comportements hardcodés (ex: `/out/sam_annotations/`) — à refactoriser si tu veux l'utiliser pour SYNTHIA.

## Actions recommandées (si tu veux que je fasse)
 - Mettre à jour `log.py` + `dataset.py` pour supporter `--dataset synthia` et `SYNTHIA_DATA_ROOT`.
 - Refactorer `prepare_new_dataset.py` pour paramètres CLI, gestion d’erreurs, et compatibilité avec `metadata.jsonl`.

---

Si tu veux, je peux appliquer les deux dernières actions automatiquement (patchs).

