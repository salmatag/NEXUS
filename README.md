# NEXUS

Projet NEXUS : notebooks d'analyse et interface web locale (`Nexus_site`).

## Contenu

- `Nexus.ipynb`, `Interface.ipynb`, `Analyse_questionnaire_Nexus.ipynb`, `Nexus_Extensions_NER_RAG.ipynb` : notebooks du projet.
- `Nexus_site/` : application Flask (interface locale RAG + NER + Ollama).

## Modèles non inclus (fichiers volumineux)

GitHub refuse les fichiers de plus de 100 Mo. Ces fichiers sont donc exclus par `.gitignore` et doivent être récupérés localement :

- `Nexus_site/models/biomedbert_ner_bc5cdr/model.safetensors` (~415 Mo)
- `Nexus_site/models/nexus_pubmed_big/model.weights.h5` (~91 Mo)
- les vidéos `*.mp4`

Pour versionner ces fichiers, utilisez [Git LFS](https://git-lfs.com/) ou un stockage externe.

## Installation

Voir `Nexus_site/README.md`.
